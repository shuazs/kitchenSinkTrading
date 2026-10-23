"""Pairs trading / statistical arbitrage between crypto assets.

Bar-level research simulator (daily or intraday bars), not a `Strategy`:
pairs need per-pair state (in a trade or not, hedge ratio fixed at entry)
and intraday bars, which the daily-weights engine does not cover.

Building blocks:

* spread of log prices y - beta * x, hedge ratio from a rolling OLS on
  earlier bars (`pairs.rolling_hedge`) or fixed from a formation window;
* `adf_t`: Engle-Granger style ADF t-stat of a spread (no statsmodels);
* `band_positions`: enter at |z| > entry, exit at |z| < exit, stop at
  |z| > stop or after `max_hold` bars;
* `simulate_pair`: signal at bar t, position held from bar t + 1. The legs are
  dollar weights wy = pos / (1 + |b|), wx = -pos * b / (1 + |b|) (gross 1),
  and costs are charged on every weight change;
* `rotation_returns`: the long-only version a US spot account can run without
  shorting: hold the cheap leg, the expensive leg, or a neutral mix;
* `rolling_pairs`: Gatev-style formation/trading windows over a coin universe.

Per-fill costs, in bp of notional (fee + half spread + slippage):
spot taker 55 (Kraken Pro / Coinbase Advanced low tier, = `COSTS["crypto"]`),
spot maker 30, futures 7 (Coinbase Derivatives nano / perp-style futures).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .pairs import half_life, rolling_hedge

COST_BPS = {"spot_taker": 55.0, "spot_maker": 30.0, "futures": 7.0}

# Engle-Granger 5% critical value for two series with a constant
EG_CRIT_5 = -3.34


def adf_t(spread) -> float:
    """t-stat of b in dS = a + b * S(-1) (Dickey-Fuller without lags).
    More negative = more mean-reverting; below `EG_CRIT_5` is cointegrated at
    ~5% when the spread came from an OLS fit on the same window."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        return np.nan
    lag, d = s[:-1], np.diff(s)
    X = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ coef
    dof = len(d) - 2
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(coef[1] / np.sqrt(cov[1, 1]))


def ols_beta(y, x) -> tuple[float, float]:
    """alpha, beta of y = alpha + beta * x."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    b = np.cov(y, x, ddof=0)[0, 1] / np.var(x)
    return float(y.mean() - b * x.mean()), float(b)


def band_positions(z, entry: float = 2.0, exit: float = 0.5, stop: float | None = 4.0,
                   max_hold: int | None = None) -> np.ndarray:
    """+1 = long the spread (z low: buy y, sell x), -1 = short it, 0 = flat.
    The position at bar t is decided from z at bar t only."""
    z = np.asarray(z, float)
    pos = np.zeros(len(z))
    p, held = 0, 0
    for i, v in enumerate(z):
        if not np.isfinite(v):
            p, held = 0, 0
        elif p == 0:
            if v <= -entry and (stop is None or v > -stop):
                p, held = 1, 0
            elif v >= entry and (stop is None or v < stop):
                p, held = -1, 0
        else:
            held += 1
            out = (p == 1 and v >= -exit) or (p == -1 and v <= exit)
            stopped = stop is not None and abs(v) >= stop
            timed = max_hold is not None and held >= max_hold
            if out or stopped or timed:
                p, held = 0, 0
        pos[i] = p
    return pos


def leg_weights(pos, beta) -> tuple[np.ndarray, np.ndarray]:
    """Dollar weights of y and x for spread position `pos` and hedge `beta`,
    gross exposure 1. Beta is frozen at entry so a trade's hedge never drifts."""
    pos = np.asarray(pos, float)
    beta = np.asarray(beta, float) * np.ones(len(pos))
    b = beta.copy()
    for i in range(1, len(pos)):
        if pos[i] != 0 and pos[i] == pos[i - 1]:
            b[i] = b[i - 1]
    b = np.nan_to_num(b)
    wy = pos / (1 + np.abs(b))
    wx = -pos * b / (1 + np.abs(b))
    return wy, wx


def simulate_pair(ry: pd.Series, rx: pd.Series, pos, beta, cost_bps: float,
                  delay: int = 1) -> pd.DataFrame:
    """Per-bar net returns of a dollar pair. `ry`, `rx` are simple returns of
    each bar (close t-1 -> close t); weights decided at bar t earn from bar
    t + delay. Returns gross, cost, net, and the held weights."""
    wy, wx = leg_weights(pos, beta)
    wy = pd.Series(wy, ry.index).shift(delay).fillna(0.0)
    wx = pd.Series(wx, ry.index).shift(delay).fillna(0.0)
    gross = wy * ry.fillna(0) + wx * rx.fillna(0)
    turn = wy.diff().abs().fillna(wy.abs()) + wx.diff().abs().fillna(wx.abs())
    cost = turn * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "wy": wy, "wx": wx})


def rotation_returns(ry: pd.Series, rx: pd.Series, pos, cost_bps: float, neutral: float = 0.5,
                     delay: int = 1) -> pd.DataFrame:
    """Long-only pair: hold y when the spread is cheap (pos +1), x when it is
    rich (pos -1), and `neutral` in y / 1 - neutral in x otherwise (0.5 =
    50/50 hold; None = cash when flat). No shorting, so a spot account can run it."""
    pos = np.asarray(pos, float)
    if neutral is None:
        wy, wx = (pos > 0).astype(float), (pos < 0).astype(float)
    else:
        wy = np.where(pos > 0, 1.0, np.where(pos < 0, 0.0, neutral))
        wx = 1 - wy
    wy = pd.Series(wy, ry.index).shift(delay).fillna(0.0)
    wx = pd.Series(wx, ry.index).shift(delay).fillna(0.0)
    gross = wy * ry.fillna(0) + wx * rx.fillna(0)
    # drift between bars is ignored: weights are rebalanced only when they change
    turn = wy.diff().abs().fillna(wy.abs()) + wx.diff().abs().fillna(wx.abs())
    cost = turn * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "wy": wy, "wx": wx})


def rolling_spread_z(logy: pd.Series, logx: pd.Series, hedge_window: int, z_window: int,
                     ratio: bool = False) -> pd.DataFrame:
    """Spread and z-score with rolling (past-only) hedge and z windows.
    `ratio=True` uses beta = 1 (log price ratio)."""
    if ratio:
        h = pd.DataFrame({"alpha": 0.0, "beta": 1.0, "spread": logy - logx}, index=logy.index)
    else:
        h = rolling_hedge(logy, logx, hedge_window)
    s = h["spread"]
    m = s.rolling(z_window, min_periods=z_window // 2).mean().shift()
    sd = s.rolling(z_window, min_periods=z_window // 2).std().shift()
    h["z"] = (s - m) / sd
    return h


def trade_list(res: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip from `simulate_pair` output: bars held, gross
    and net return (sum of per-bar returns while held)."""
    live = (res["wy"] != 0) | (res["wx"] != 0)
    # band_positions always passes through a flat bar between trades
    tid = (live != live.shift()).cumsum().where(live)
    if tid.notna().sum() == 0:
        return pd.DataFrame(columns=["start", "bars", "gross", "net"])
    g = res.assign(tid=tid).dropna(subset=["tid"])
    # the exit fill's cost lands on the bar after the last held bar
    exit_cost = res["cost"].shift(-1).where(live & ~live.shift(-1, fill_value=False), 0)
    g["cost2"] = g["cost"] + exit_cost.reindex(g.index).fillna(0)
    out = g.groupby("tid").agg(start=("gross", lambda s: s.index[0]), bars=("gross", "size"),
                               gross=("gross", "sum"), cost=("cost2", "sum"))
    out["net"] = out["gross"] - out["cost"]
    return out.reset_index(drop=True)


def stats(net: pd.Series, periods_per_year: float) -> dict:
    """CAGR, vol, Sharpe, max drawdown, exposure-free summary of a return series."""
    net = net.dropna()
    if len(net) == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    eq = (1 + net).cumprod()
    years = len(net) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = net.std() * np.sqrt(periods_per_year)
    sharpe = net.mean() / net.std() * np.sqrt(periods_per_year) if net.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe), "max_dd": float(dd)}


def select_pairs(logp: pd.DataFrame, n_pairs: int, min_hl: float = 1, max_hl: float = 60,
                 method: str = "coint") -> list[dict]:
    """Rank all pairs on a formation window of log prices.

    method "coint": OLS hedge on the window, keep pairs whose spread ADF t is
    below `EG_CRIT_5` and half-life is in [min_hl, max_hl], best ADF first.
    method "distance": Gatev et al. (2006): normalised price paths with the
    smallest sum of squared differences, beta 1 on normalised prices.
    Each pick carries the formation alpha, beta, spread mean and std."""
    logp = logp.dropna(axis=1, how="any")
    rows = []
    for a, b in itertools.combinations(logp.columns, 2):
        y, x = logp[a], logp[b]
        if method == "distance":
            yn, xn = y - y.iloc[0], x - x.iloc[0]
            alpha, beta = 0.0, 1.0
            s = yn - xn
            score = float((s ** 2).sum())
            # spread is measured from each coin's first formation price
            alpha = float(y.iloc[0] - x.iloc[0])
        else:
            alpha, beta = ols_beta(y, x)
            if beta <= 0:
                continue
            s = y - alpha - beta * x
            score = adf_t(s)
            if not score < EG_CRIT_5:
                continue
        hl = half_life(s)
        if not (min_hl <= hl <= max_hl):
            continue
        rows.append({"y": a, "x": b, "alpha": alpha, "beta": beta, "mu": float(s.mean()),
                     "sd": float(s.std()), "score": score, "half_life": hl})
    rows.sort(key=lambda r: r["score"])
    return rows[:n_pairs]


def rolling_pairs(close: pd.DataFrame, formation: int = 180, trading: int = 30, n_pairs: int = 5,
                  entry: float = 2.0, exit: float = 0.5, stop: float | None = 4.0,
                  method: str = "coint", cost_bps: float = COST_BPS["spot_taker"],
                  long_only: bool = False, min_hl: float = 1, max_hl: float = 60,
                  cache: dict | None = None) -> dict:
    """Formation/trading cycle over a coin universe (daily bars).

    Every `trading` bars: pick `n_pairs` on the previous `formation` bars,
    then trade each for the next `trading` bars with z = (spread - mu) / sd
    from the formation window (fixed hedge). Capital is split equally across
    `n_pairs` slots; an empty slot sits in cash. Open trades are closed at the
    end of the trading window. `long_only` buys only the cheap leg (spot,
    no shorting) and sits in cash otherwise. Pass the same `cache` dict to
    calls that share `close` to reuse pair rankings across a parameter sweep.
    Returns {"returns": net Series, "gross": Series, "trades": DataFrame, "picks": DataFrame}."""
    cols = list(close.columns)
    L = np.log(close.to_numpy(float))
    R = np.nan_to_num(close.pct_change().to_numpy(float))
    idx = close.index
    net = np.zeros(len(idx))
    gross = np.zeros(len(idx))
    trades, picks = [], []
    for start in range(formation, len(idx) - 1, trading):
        form = pd.DataFrame(L[start - formation:start], columns=cols)
        key = (method, formation, start, min_hl, max_hl)
        if cache is None or key not in cache:
            ranked = select_pairs(form, len(cols) ** 2, min_hl, max_hl, method)
            if cache is not None:
                cache[key] = ranked
        else:
            ranked = cache[key]
        chosen = ranked[:n_pairs]
        # signals on bars start-1 .. end-1; weights held on bars start .. end-1
        end = min(start + trading, len(idx))
        for c in chosen:
            picks.append({"formed": idx[start - 1], **c})
            iy, ix = cols.index(c["y"]), cols.index(c["x"])
            sl = slice(start - 1, end)
            z = (L[sl, iy] - c["alpha"] - c["beta"] * L[sl, ix] - c["mu"]) / c["sd"]
            pos = band_positions(z, entry, exit, stop)
            pos[-2:] = 0  # flat by the last bar of the window
            if long_only:
                wy, wx = (pos > 0).astype(float), (pos < 0).astype(float)
            else:
                wy, wx = leg_weights(pos, c["beta"])
            wy, wx = np.r_[0.0, wy[:-1]], np.r_[0.0, wx[:-1]]
            g = wy * R[sl, iy] + wx * R[sl, ix]
            cost = (np.abs(np.diff(wy, prepend=0.0)) + np.abs(np.diff(wx, prepend=0.0))) * cost_bps / 1e4
            net[sl] += (g - cost) / n_pairs
            gross[sl] += g / n_pairs
            live = (wy != 0) | (wx != 0)
            edges = np.flatnonzero(np.diff(np.r_[0, live.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                tc = cost[a:b + 1].sum()  # entry fill on bar a, exit fill on bar b
                trades.append({"start": idx[start - 1 + a], "bars": b - a, "gross": g[a:b].sum(),
                               "cost": tc, "net": g[a:b].sum() - tc, "y": c["y"], "x": c["x"]})
    first = formation
    return {
        "returns": pd.Series(net[first:], idx[first:]),
        "gross": pd.Series(gross[first:], idx[first:]),
        "trades": pd.DataFrame(trades),
        "picks": pd.DataFrame(picks),
    }
