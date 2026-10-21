"""Minute-level lead-lag between Kalshi hourly price ladders and the underlying.

Kalshi lists hourly "above K" ladders on the S&P 500 (KXINXU), Nasdaq-100
(KXNASDAQ100U) and Bitcoin (KXBTCD). Each trade at strike K, YES price p and
tau minutes to settlement implies an underlying level

    S_implied = K * exp(sigma * sqrt(tau) * PhiInv(p))

with sigma the trailing 60-minute realised 1-minute vol of the underlying.
Per minute m the trades in [m, m+1) give:

- z: contract-weighted ln(S_implied / close of bar m) in units of sigma*sqrt(tau)
- flow: signed taker contracts (+ = bullish)

Both are known at the end of minute m, so the return from close of bar m to
close of bar m+k is out of sample. `zero_dte_proxy` prices an ATM 0DTE option
on that move with Black-Scholes, a half-spread and commission.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

MIN_PER_YEAR = 390 * 252


def parse_strikes(markets: pd.DataFrame) -> pd.DataFrame:
    """Strike K and a `below` flag from Kalshi tickers like KXINXU-26JAN30H1600-T7049.9999."""
    m = markets.copy()
    m["K"] = m.ticker.str.extract(r"-[TB](-?\d+(?:\.\d+)?)$")[0].astype(float)
    sub = m.yes_sub_title.fillna("").str.lower()
    m["below"] = sub.str.contains(r"below|or less|less than|<")
    return m.dropna(subset=["K"])


def realised_sigma(close: pd.Series, window: int = 60) -> pd.Series:
    lr = np.log(close).diff()
    lr[(close.index.to_series().diff() != pd.Timedelta("1min")).values] = np.nan
    return lr.rolling(window, min_periods=20).std().clip(lower=1e-5)


def trade_implied(trades: pd.DataFrame, markets: pd.DataFrame, close: pd.Series,
                  pmin: float = 0.1, pmax: float = 0.9, max_tau: float = 60) -> pd.DataFrame:
    """Per-minute z, flow, implied level, trade and contract counts. `close` is 1-min, UTC bar-start index."""
    m = parse_strikes(markets)
    t = trades[trades.ticker.isin(m.ticker)].merge(m[["ticker", "K", "below", "close_time"]], on="ticker")
    t["tau"] = (t.close_time - t.created_time).dt.total_seconds() / 60
    t["p"] = np.where(t.below, 1 - t.yes_price / 100, t.yes_price / 100)
    t = t[(t.tau >= 2) & (t.tau <= max_tau) & t.p.between(pmin, pmax)].copy()
    sig = realised_sigma(close)
    t["m"] = t.created_time.dt.floor("min")
    t["S"] = close.reindex(t.m).values
    t["sig"] = sig.reindex(t.m).values
    t = t.dropna(subset=["S", "sig"])
    w = t.sig * np.sqrt(t.tau)
    t["implied"] = t.K * np.exp(w * norm.ppf(t.p))
    t["z"] = np.log(t.implied / t.S) / w
    t["flow"] = np.where(t.taker_side == "yes", 1, -1) * np.where(t.below, -1, 1) * t["count"]
    t["wz"] = t.z * t["count"]
    t["wi"] = t.implied * t["count"]
    g = t.groupby("m")
    n = g["count"].sum()
    return pd.DataFrame({"z": g.wz.sum() / n, "implied": g.wi.sum() / n, "flow": g.flow.sum(),
                         "trades": g.size(), "contracts": n})


def returns_around(f: pd.DataFrame, close: pd.Series, fwd=(1, 2, 3, 5, 10), back=(1, 5)) -> pd.DataFrame:
    """Underlying log returns in sigma units: past{k} = bar m-k -> m, fwd{k} = bar m -> m+k (NaN across gaps)."""
    lp = np.log(close)
    sig = realised_sigma(close).reindex(f.index).values
    pos = close.index.get_indexer(f.index)
    ok = pos >= 0
    out = pd.DataFrame(index=f.index)
    for k in list(back) + [-k for k in fwd]:
        j = np.clip(pos - k, 0, len(close) - 1)
        good = ok & (close.index[j] == f.index - pd.Timedelta(minutes=k))
        r = (lp.values[pos] - lp.values[j]) if k > 0 else (lp.values[j] - lp.values[pos])
        out[f"past{k}" if k > 0 else f"fwd{-k}"] = np.where(good, r / sig, np.nan)
    return out


def leadlag_table(f: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each signal with past and forward returns. Positive fwd corr would mean Kalshi leads."""
    sigs = {"z": f.z, "flow": np.sign(f.flow) * np.log1p(f.flow.abs())}
    return pd.DataFrame({k: {c: s.corr(R[c]) for c in R.columns} for k, s in sigs.items()}).T


def tail_hit_rate(x: pd.Series, fwd: pd.Series, q: float = 0.01) -> dict:
    """Trade the sign of x in its top and bottom q tails: hit rate and mean signed forward move (sigma units)."""
    hi, lo = x > x.quantile(1 - q), x < x.quantile(q)
    s = pd.concat([fwd[hi], -fwd[lo]]).dropna()
    return {"n": len(s), "hit": (s > 0).mean(), "mean": s.mean(), "t": s.mean() / s.std() * np.sqrt(len(s))}


def bs_price(S, K, T, vol, call):
    d1 = (np.log(S / K) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return np.where(call, S * norm.cdf(d1) - K * norm.cdf(d2), K * norm.cdf(-d2) - S * norm.cdf(-d1))


def zero_dte_proxy(signal: pd.Series, close: pd.Series, hold: int = 5, etf_price: float = 500.0,
                   half_spread: float = 0.01, commission: float = 0.0065, iv_mult: float = 1.15) -> pd.DataFrame:
    """Buy an ATM 0DTE call (signal > 0) or put (signal < 0) at the end of minute t, sell `hold` minutes later.

    The underlying is rescaled to an ETF trading at `etf_price` (SPY/QQQ-like), strike rounded to $1, IV =
    iv_mult x trailing realised vol, expiry 16:00 ET. Costs per share: half_spread each side plus commission
    ($0.65/contract = 0.0065/share). One position at a time. Returns one row per trade.
    """
    sig = realised_sigma(close)
    rows, busy = [], pd.Timestamp(0, tz="UTC")
    for t, s in signal.items():
        t1 = t + pd.Timedelta(minutes=hold)
        if t < busy or t not in close.index or t1 not in close.index or not np.isfinite(sig.get(t, np.nan)):
            continue
        et = t.tz_convert("America/New_York")
        left = (et.normalize() + pd.Timedelta(hours=16) - et).total_seconds() / 60
        if left <= hold + 5:
            continue
        scale = etf_price / close[t]
        S0, S1 = close[t] * scale, close[t1] * scale
        vol = iv_mult * sig[t] * np.sqrt(MIN_PER_YEAR)
        K, call = round(S0), s > 0
        p0 = float(bs_price(S0, K, left / MIN_PER_YEAR, vol, call))
        p1 = float(bs_price(S1, K, (left - hold) / MIN_PER_YEAR, vol, call))
        cost = half_spread + commission
        rows.append({"ts": t, "call": call, "premium": p0, "ret": (p1 - cost) / (p0 + cost) - 1})
        busy = t1
    return pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame(columns=["call", "premium", "ret"])


def breakeven_hit_rate(close: pd.Series, hold: int = 5, n: int = 20000, seed: int = 0, **kw) -> float:
    """Hit rate a direction call needs for ATM 0DTE options to break even, from random entry minutes."""
    rng = np.random.default_rng(seed)
    idx = close.index[rng.choice(len(close), size=min(n, len(close)), replace=False)].sort_values()
    up = close.reindex(idx + pd.Timedelta(minutes=hold)).values > close.reindex(idx).values
    win, lose = [], []
    for side in (1, -1):
        r = zero_dte_proxy(pd.Series(side, idx), close, hold=hold, **kw)
        right = pd.Series(up if side > 0 else ~up, idx).reindex(r.index)
        win.append(r.ret[right]); lose.append(r.ret[~right])
    W, L = pd.concat(win).mean(), -pd.concat(lose).mean()
    return L / (W + L)
