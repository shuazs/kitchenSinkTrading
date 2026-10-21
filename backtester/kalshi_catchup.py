"""Trading Kalshi index and crypto ladders after the underlying moves.

Kalshi's hourly "above K at HH:00" markets (KXINXU for the S&P 500,
KXNASDAQ100U, KXBTCD, KXETHD) are digital options on the index or coin.
At the hourly scale their prices follow the underlying with a lag, so the
idea is: when the underlying jumps, buy the contract whose last quote has
not caught up yet, and hold it to expiry (or sell it back a few minutes
later).

Kalshi's public trade history has prints, not quotes, so the book is
rebuilt from them: a print whose taker bought YES was at the YES ask, one
whose taker bought NO was at the YES bid. The rule, at each minute where
the underlying moved more than `z` sigma over `k` minutes:

- fair value of each contract of the nearest hourly event is
  Phi(ln(S/K) / (c * sigma * sqrt(tau))), sigma a per-minute EWMA of the
  underlying's 1-minute returns and `c` a per-series multiplier fitted to
  Kalshi prices;
- pick the contract on the side of the move whose last ask is furthest
  below fair value, and only if that gap is above `margin`;
- after `latency` seconds, fill against the prints on our side over the
  next `window` seconds at or below fair value minus `margin`, up to their
  printed size. Someone really traded there, so the size existed, but we
  would have had to beat that trader to it: fills are an upper bound;
- pay Kalshi's taker fee, ceil(rate * C * P * (1 - P)) to the cent, with
  rate 0.07 on crypto and 0.035 on the S&P 500 and Nasdaq-100 markets.

Underlying prices are indexed by the time they are KNOWN (bar end), so a
trigger at minute t only uses information a trader had at t.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEE_RATE = {"KXINXU": 0.035, "KXNASDAQ100U": 0.035, "KXBTCD": 0.07, "KXETHD": 0.07}
VOL_MULT = {"KXINXU": 1.1, "KXNASDAQ100U": 1.1, "KXBTCD": 1.2, "KXETHD": 1.0}
_NS = 1_000_000_000


def taker_fee(contracts: int, price: float, rate: float = 0.07) -> float:
    """Kalshi taker fee in dollars for `contracts` at `price` (0-1), rounded up to the cent."""
    return math.ceil(100 * rate * contracts * price * (1 - price) - 1e-9) / 100


def known_prices(bars: pd.DataFrame, bar: str = "1min") -> pd.Series:
    """Close of each bar, re-indexed to when it is known (bar start + bar length)."""
    s = bars["close"].copy()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(bar)
    return s[~s.index.duplicated(keep="last")].sort_index()


def minute_vol(prices: pd.Series, span: int = 60, max_gap: str = "5min") -> pd.Series:
    """EWMA per-minute volatility of log returns, ignoring returns across session gaps."""
    r = np.log(prices).diff()
    r[(prices.index.to_series().diff() > pd.Timedelta(max_gap)).to_numpy()] = np.nan
    return np.sqrt((r**2).ewm(span=span, min_periods=20).mean())


def digital_fair_value(S, K, sigma, tau_min, mult: float = 1.0):
    """P(underlying > K at expiry) for a driftless lognormal, sigma per minute."""
    x = np.log(S / K) / (mult * sigma * np.sqrt(np.maximum(tau_min, 0.5)))
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, dtype=float) / math.sqrt(2))


def _ns(x) -> np.ndarray:
    return pd.DatetimeIndex(x).as_unit("ns").asi8


def _books(trades: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Per event, per contract arrays of print times, YES prices, taker side and size."""
    t = trades.sort_values("created_time")
    books: dict = {}
    for (ev, tk), g in t.groupby(["event_ticker", "ticker"], sort=False):
        books.setdefault(ev, []).append(
            dict(
                tk=tk,
                K=float(g["K"].iat[0]),
                res=g["result"].iat[0] == "yes",
                ts=_ns(g["created_time"]),
                p=g["yes_price"].to_numpy() / 100,
                yes=(g["taker_side"] == "yes").to_numpy(),
                n=g["count"].to_numpy(),
            )
        )
    ev = t.groupby("event_ticker").agg(close=("close_time", "max"), opn=("created_time", "min")).sort_values("close")
    return books, ev.index.to_numpy(), _ns(ev["close"]), _ns(ev["opn"])


def catchup_backtest(
    prices: pd.Series,
    trades: pd.DataFrame,
    *,
    k: int = 2,
    z: float = 2.5,
    latency: float = 5.0,
    window: float = 60.0,
    margin: float = 0.03,
    risk: float = 250.0,
    min_tau: float = 3,
    max_tau: float = 60,
    vol_mult: float = 1.0,
    fee_rate: float = 0.07,
    exits: tuple[int, ...] = (5, 15),
) -> pd.DataFrame:
    """One row per trigger. Filled rows have qty > 0, `pnl` (held to settlement)
    and `pnl_x{h}` (sold at the first YES-bid print h minutes after the fill).

    prices: underlying indexed by the time each price is known (see `known_prices`).
    trades: Kalshi prints with ticker, event_ticker, created_time, yes_price (cents),
        taker_side, count, plus the market's K, result ('yes'/'no') and close_time.
    """
    prices = prices.copy()
    prices.index = pd.DatetimeIndex(prices.index).as_unit("ns")
    sig = minute_vol(prices)
    past = prices.reindex(prices.index - pd.Timedelta(minutes=k)).to_numpy()
    zs = pd.Series(np.log(prices.to_numpy() / past) / (sig.to_numpy() * np.sqrt(k)), prices.index).dropna()
    zs = zs[zs.abs() > z]
    zs = zs[zs.index.to_series().diff().fillna(pd.Timedelta("1D")) > pd.Timedelta(minutes=k)]  # one per burst
    books, evn, evc, evo = _books(trades)
    rows = []
    for tt, zv in zs.items():
        tn, S0, sg = tt.value, prices.at[tt], sig.at[tt]
        j = np.searchsorted(evc, tn + int(min_tau * 60 * _NS))
        if j >= len(evc) or evc[j] > tn + max_tau * 60 * _NS or evo[j] >= tn:
            continue
        tau, up = (evc[j] - tn) / 60 / _NS, zv > 0
        best = None
        for b in books[evn[j]]:
            F = float(digital_fair_value(S0, b["K"], sg, tau, vol_mult))
            if not 0.05 < F < 0.95:
                continue
            same = np.flatnonzero(b["yes"][: np.searchsorted(b["ts"], tn)] == up)
            if not len(same):
                continue
            last = b["p"][same[-1]] if up else 1 - b["p"][same[-1]]
            fair = F if up else 1 - F
            if best is None or fair - last > best["edge_last"]:
                best = dict(b=b, fair=fair, last_ask=last, edge_last=fair - last, age=(tn - b["ts"][same[-1]]) / _NS)
        if best is None:
            continue
        b = best.pop("b")
        won = float(b["res"] == up)
        row = dict(t=tt, dir=1 if up else -1, tau=tau, ticker=b["tk"], K=b["K"], **best, win=won, qty=0,
                   fill_px=np.nan, fee=0.0, pnl=0.0, **{f"pnl_x{h}": 0.0 for h in exits})
        lo, hi = np.searchsorted(b["ts"], [tn + int(latency * _NS), tn + int((latency + window) * _NS)])
        sel = np.arange(lo, hi)[b["yes"][lo:hi] == up]
        px = b["p"][sel] if up else 1 - b["p"][sel]
        ok = px <= best["fair"] - margin
        if best["edge_last"] > margin and ok.any():
            want, got, cost = int(risk / max(px[ok][0], 0.01)), 0, 0.0
            for p_, c_ in zip(px[ok], b["n"][sel][ok]):
                q = min(int(c_), want - got)
                got, cost = got + q, cost + q * p_
                if got >= want:
                    break
            avg = cost / got
            f = taker_fee(got, avg, fee_rate)
            row.update(qty=got, fill_px=avg, fee=f, pnl=got * (won - avg) - f, model_edge=best["fair"] - avg)
            tf = b["ts"][sel][ok][0]
            for h in exits:
                i = np.searchsorted(b["ts"], tf + h * 60 * _NS)
                opp = np.flatnonzero(b["yes"][i:] != up)
                if len(opp) and b["ts"][i + opp[0]] < evc[j]:
                    xp = b["p"][i + opp[0]] if up else 1 - b["p"][i + opp[0]]
                    row[f"pnl_x{h}"] = got * (xp - avg) - f - taker_fee(got, xp, fee_rate)
                else:  # nobody bid before expiry: hold to settlement
                    row[f"pnl_x{h}"] = row["pnl"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, exits: tuple[int, ...] = (5, 15)) -> dict:
    """Trigger and fill counts, average gross edge per contract (cents) and P&L with t-stats."""
    f = res[res["qty"] > 0] if len(res) else res
    t = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan  # noqa: E731
    out = dict(triggers=len(res), fills=len(f), contracts=int(f["qty"].sum()) if len(f) else 0)
    if len(f):
        out.update(gross_c=100 * (f["win"] - f["fill_px"]).mean(), model_edge_c=100 * f["model_edge"].mean(),
                   pnl=f["pnl"].sum(), t=t(f["pnl"]))
        for h in exits:
            out[f"pnl_x{h}"], out[f"t_x{h}"] = f[f"pnl_x{h}"].sum(), t(f[f"pnl_x{h}"])
    return out
