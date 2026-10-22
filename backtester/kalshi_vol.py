"""Kalshi S&P ladders as a volatility signal for selling 0DTE iron flies.

Kalshi's end-of-day S&P 500 ladders (KXINXU, INXU, INXD: "will the S&P close
above K today?") price a full distribution for the 16:00 close. The width
between its 25% and 75% points is the crowd's expected move for the rest of
the session. Research in the "Leading Kalshi markets for stocks" thread found
that when this width is narrow relative to VIX1D (the options market's
1-day implied vol), selling volatility for the rest of the day earns more
than usual at 12:30 and 14:30 ET, but not at 10:30.

Pipeline:

1. `range_panel` builds one row per (day, decision time): SPX price, minutes
   to the close, the close, realised variance to the close from 1-minute
   bars, Kalshi's implied quartiles (last trade per strike in the prior 3
   hours) and VIX1D as of the decision time.
2. `add_signal` scales Kalshi's range to an annualised vol, compares it to
   VIX1D and flags rows where the ratio sits in the bottom third of its own
   history (expanding window, earlier days only).
3. `iron_fly_backtest` sells an XSP (1/10 SPX) iron fly at the money with
   wings `wing_pct` away, priced with Black-Scholes at `iv_mult` x VIX1D,
   held to the cash-settled close.

There is no free 0DTE option history, so option prices are modelled.
`fair_iv_mult` finds the VIX1D scale at which short straddles break even,
so a run at that scale tests timing alone, and a run above it models the
variance premium sellers normally collect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .shortterm import bs_price

NY = "America/New_York"
MIN_PER_YEAR = 390 * 252
IQR_TO_SD = 1.349  # interquartile range of a normal distribution in standard deviations
LADDER_SERIES = ("KXINXU", "INXU", "INXD", "INXDU")


# --------------------------------------------------------------------------- Kalshi ladders

def ladder_quantiles(K: np.ndarray, p_above: np.ndarray, levels=(0.25, 0.5, 0.75)) -> list[float]:
    """Quantiles of the close from P(close > K) at each strike.

    Probabilities are forced to fall as K rises; a quantile outside the
    ladder's strikes is NaN rather than extrapolated.
    """
    o = np.argsort(K)
    K, P = np.asarray(K, float)[o], np.minimum.accumulate(np.asarray(p_above, float)[o])
    out = []
    for q in levels:
        target = 1 - q
        if P[0] < target or P[-1] > target:
            out.append(np.nan)
            continue
        j = int(np.argmax(P < target))
        out.append(K[j - 1] + (K[j] - K[j - 1]) * (P[j - 1] - target) / max(P[j - 1] - P[j], 1e-9))
    return out


def end_of_day_ladders(markets: pd.DataFrame) -> pd.DataFrame:
    """Strike markets of the daily S&P ladders that settle at 16:00 ET, with K and a `below` flag."""
    m = markets[markets.event_ticker.str.split("-").str[0].isin(LADDER_SERIES)].copy()
    m["K"] = m.ticker.str.extract(r"-T(-?\d+(?:\.\d+)?)$")[0].astype(float)
    m = m.dropna(subset=["K"])
    close = pd.to_datetime(m.close_time, utc=True).dt.tz_convert(NY)
    m = m[close.dt.hour == 16]
    m["day"] = close[close.dt.hour == 16].dt.tz_localize(None).dt.normalize()
    sub = m.yes_sub_title.fillna("").str.lower()
    m["below"] = sub.str.contains(r"below|or less|less than|<")
    return m


# --------------------------------------------------------------------------- panel

def range_panel(markets: pd.DataFrame, trades: pd.DataFrame, spx: pd.DataFrame, vix1d: pd.Series,
                times=("10:30", "12:30", "14:30"), lookback: str = "3h", min_strikes: int = 4) -> pd.DataFrame:
    """One row per (day, decision time).

    spx: 1-minute SPX bars, UTC bar-start index, column `close`.
    vix1d: VIX1D values indexed by the time they are known (UTC).
    """
    lad = end_of_day_ladders(markets).set_index("ticker")
    t = trades[trades.ticker.isin(lad.index)][["ticker", "yes_price", "created_time"]].copy()
    t["created_time"] = pd.to_datetime(t.created_time, utc=True)
    t["day"] = t.ticker.map(lad.day)
    t = t.sort_values("created_time")
    px = spx["close"].copy()
    px.index = px.index + pd.Timedelta(minutes=1)  # value known at the bar's end
    lr = np.log(px).diff()
    vix1d = vix1d.sort_index()
    rows = []
    for day, g in t.groupby("day"):
        d_px = px[(px.index.tz_convert(NY).tz_localize(None).normalize() == day)]
        if len(d_px) < 300:
            continue
        close_t = d_px.index[-1]
        for hm in times:
            h, mi = map(int, hm.split(":"))
            now = (day + pd.Timedelta(hours=h, minutes=mi)).tz_localize(NY).tz_convert("UTC")
            gg = g[(g.created_time <= now) & (g.created_time > now - pd.Timedelta(lookback))]
            last = gg.groupby("ticker").yes_price.last() / 100
            if len(last) < min_strikes:
                continue
            info = lad.loc[last.index]
            p_above = np.where(info.below, 1 - last.to_numpy(), last.to_numpy())
            q25, q50, q75 = ladder_quantiles(info.K.to_numpy(), p_above)
            S = d_px[:now]
            v = vix1d[:now]
            if not len(S) or not len(v) or not np.isfinite(q25 * q75) or q75 <= q25:
                continue
            rest = lr[(lr.index > now) & (lr.index <= close_t)]
            rows.append({"time": now.tz_convert(NY).tz_localize(None), "price": float(S.iloc[-1]),
                         "close": float(d_px.iloc[-1]), "minutes_left": len(rest), "rv": float((rest ** 2).sum()),
                         "q25": q25, "q50": q50, "q75": q75, "n_strikes": len(last), "vix1d": float(v.iloc[-1])})
    return pd.DataFrame(rows).set_index("time").sort_index() if rows else pd.DataFrame()


def add_signal(P: pd.DataFrame, min_history: int = 60, quantile: float = 1 / 3) -> pd.DataFrame:
    """Kalshi annualised vol, its ratio to VIX1D, and a `low` flag per decision time.

    `adj` divides the ratio by its expanding median of earlier days, so a
    constant gap between the two vol conventions drops out. `low` marks rows
    whose adj is below the `quantile` of earlier days' adj at the same time.
    """
    P = P.copy()
    P["kalshi_vol"] = np.log(P.q75 / P.q25) / IQR_TO_SD / np.sqrt(P.minutes_left / MIN_PER_YEAR)
    P["ratio"] = P.kalshi_vol / (P.vix1d / 100)
    P["hm"] = P.index.strftime("%H:%M")
    out = []
    for _, g in P.groupby("hm"):
        g = g.sort_index().copy()
        g["adj"] = np.log(g.ratio / g.ratio.expanding().median().shift(1))
        g["thr_low"] = g.adj.expanding().quantile(quantile).shift(1)
        g["thr_high"] = g.adj.expanding().quantile(1 - quantile).shift(1)
        n = np.arange(len(g))
        g["low"] = (g.adj < g.thr_low) & (n >= min_history)
        g["high"] = (g.adj > g.thr_high) & (n >= min_history)
        g["live"] = n >= min_history
        out.append(g)
    return pd.concat(out).sort_index()


# --------------------------------------------------------------------------- options

def iron_fly_price(S: float, K: float, wing: float, minutes: float, vol: float) -> float:
    """Credit for selling the K straddle and buying the K-wing put and K+wing call (flat vol)."""
    short = bs_price(S, K, minutes, vol, True) + bs_price(S, K, minutes, vol, False)
    long = bs_price(S, K + wing, minutes, vol, True) + bs_price(S, K - wing, minutes, vol, False)
    return short - long


def iron_fly_payoff(close: float, K: float, wing: float) -> float:
    """What the seller owes at settlement (0 to wing)."""
    return min(abs(close - K), wing)


def fair_iv_mult(P: pd.DataFrame, lo: float = 0.1, hi: float = 3.0) -> float:
    """VIX1D scale at which ATM straddles at the rows' times break even on average."""
    P = P.dropna(subset=["vix1d", "close", "price"])
    K = P.price.round().to_numpy()
    pay = np.abs(P.close.to_numpy() - K).mean()
    args = list(zip(P.price, K, P.minutes_left, P.vix1d))
    for _ in range(40):
        mid = (lo + hi) / 2
        cost = np.mean([bs_price(S, k, mn, mid * v / 100, True) + bs_price(S, k, mn, mid * v / 100, False)
                        for S, k, mn, v in args])
        lo, hi = (lo, mid) if cost > pay else (mid, hi)
    return (lo + hi) / 2


@dataclass
class FlyCosts:
    half_spread: float = 0.02   # XSP dollars per leg per side (0DTE ATM quotes are a few cents wide)
    commission: float = 0.65    # per contract per leg, opening only; held to cash settlement


def iron_fly_backtest(P: pd.DataFrame, iv_mult: float, rule: str = "low", times=("12:30",),
                      wing_pct: float = 0.005, risk_frac: float = 0.10, capital: float = 5_000.0,
                      costs: FlyCosts | None = None, scale: float = 0.1) -> pd.DataFrame:
    """At most one XSP iron fly a day, sold at the first listed time whose rule fires.

    rule: "low" (Kalshi range narrow vs VIX1D), "high" (wide), "always".
    scale: XSP = SPX / 10. Max loss per fly = (wing - credit) x 100; trades
    size so that is at most `risk_frac` of equity, whole contracts only.
    """
    costs = costs or FlyCosts()
    rows = P[P.hm.isin(times) & P.live]
    eq, out = capital, []
    for day, g in rows.groupby(rows.index.normalize()):
        pick = g[g[rule]] if rule in ("low", "high") else g
        if pick.empty:
            continue
        t, r = pick.index[0], pick.iloc[0]
        S, close = r.price * scale, r.close * scale
        K = round(S)
        wing = max(1.0, round(S * wing_pct))
        vol = iv_mult * r.vix1d / 100
        credit = iron_fly_price(S, K, wing, r.minutes_left, vol) - 4 * costs.half_spread
        owe = iron_fly_payoff(close, K, wing)
        max_loss = (wing - credit) * 100 + 4 * costs.commission
        n = int(risk_frac * eq // max_loss) if max_loss > 0 else 0
        if n < 1 or credit <= 0:
            continue
        pnl = n * (100 * (credit - owe) - 4 * costs.commission)
        eq += pnl
        out.append({"time": t, "contracts": n, "credit": credit, "owed": owe, "wing": wing,
                    "pnl": pnl, "equity": eq, "ret_on_risk": pnl / (n * max_loss)})
    cols = ["contracts", "credit", "owed", "wing", "pnl", "equity", "ret_on_risk"]
    return pd.DataFrame(out).set_index("time") if out else pd.DataFrame(columns=cols)


def fly_summary(trades: pd.DataFrame, capital: float = 5_000.0) -> dict:
    if trades.empty:
        return {"trades": 0}
    eq = pd.concat([pd.Series([capital]), trades.equity.reset_index(drop=True)])
    r = trades.ret_on_risk
    return {"trades": len(trades), "win_rate": float((trades.pnl > 0).mean()),
            "avg_ret_on_risk": float(r.mean()), "t_stat": float(r.mean() / r.std() * np.sqrt(len(r))) if len(r) > 1 else np.nan,
            "final_equity": float(trades.equity.iloc[-1]), "max_dd": float((eq / eq.cummax() - 1).min())}
