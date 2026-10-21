"""Pairs trading between a prediction market and the asset it tracks.

Two kinds of pair:

* level pairs: a market price turned into an implied value of the asset (a
  Kalshi ladder's implied close, a Polymarket index) against the asset itself.
  Rolling hedge ratio, z-score of the spread, half-life, asset-leg trades.
* contract pairs: one Kalshi contract's price against its fair value from the
  asset (digital option on the spot price). Fills come from real trade prints
  on the side you would trade against, so the bid-ask bounce in the prints
  is not counted as profit.
"""
from __future__ import annotations

import numpy as np

import pandas as pd

KALSHI_TAKER_FEE = 0.07


def kalshi_fee(p, rate: float = KALSHI_TAKER_FEE):
    """Kalshi trading fee per contract at price p (0-1): rate * p * (1 - p).
    Kalshi rounds the order total up to the cent; per contract that is a
    rounding detail at the sizes a $5k account trades."""
    return rate * np.asarray(p) * (1 - np.asarray(p))


def half_life(spread: pd.Series, same: pd.Series | None = None) -> float:
    """Half-life of mean reversion in bars, from an AR(1) fit dS = a + b*S.
    `same` marks bars whose previous bar belongs to the same segment (same
    contract, no gap); other bars are left out of the fit."""
    s = spread.astype(float)
    d, lag = s.diff(), s.shift()
    ok = d.notna() & lag.notna()
    if same is not None:
        ok &= same.reindex(s.index).fillna(False).astype(bool)
    b = np.polyfit(lag[ok], d[ok], 1)[0]
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else np.inf


def rolling_hedge(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """Rolling OLS y = alpha + beta * x on the previous `window` bars (no
    lookahead: the fit at t uses bars up to t-1). Returns alpha, beta, spread."""
    var = x.rolling(window).var()
    beta = (y.rolling(window).cov(x) / var.where(var > 1e-12)).shift()
    alpha = (y.rolling(window).mean() - beta * x.rolling(window).mean()).shift()
    spread = (y - alpha - beta * x).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"alpha": alpha, "beta": beta, "spread": spread})


def zscore(spread: pd.Series, window: int) -> pd.Series:
    """Spread minus its trailing mean over trailing std, both from earlier bars."""
    m = spread.rolling(window, min_periods=max(2, window // 2)).mean().shift()
    s = spread.rolling(window, min_periods=max(2, window // 2)).std().shift()
    return (spread - m) / s


def asset_leg_trades(log_price: pd.Series, z: pd.Series, entry: float, hold: int,
                     cost_bp: float, delay: int = 1) -> pd.DataFrame:
    """Trade only the asset leg: short when the spread (asset minus market)
    is above +entry z, long below -entry. Enter `delay` bars after the
    signal bar, hold `hold` bars, one position at a time. Returns per-trade
    gross and net returns in basis points."""
    r = log_price.diff()
    fwd = sum(r.shift(-k) for k in range(delay + 1, delay + hold + 1))
    side = -np.sign(z).where(z.abs() > entry, 0)
    rows, busy = [], -1
    for i, (t, d) in enumerate(side.items()):
        if d == 0 or i <= busy or not np.isfinite(fwd.iloc[i]):
            continue
        g = d * fwd.iloc[i] * 1e4
        rows.append({"time": t, "side": int(d), "gross_bp": g, "net_bp": g - cost_bp})
        busy = i + delay + hold - 1
    return pd.DataFrame(rows, columns=["time", "side", "gross_bp", "net_bp"])


def fade_contract_spread(prints: pd.DataFrame, threshold: float, hold_min: int | None,
                         signal: str = "e", latency: str = "30s", fee_rate: float = KALSHI_TAKER_FEE,
                         max_wait: str = "10min") -> pd.DataFrame:
    """Fade a Kalshi contract's spread to fair value, one contract at a time.

    `prints` holds trade prints for one or more contracts with columns
    ticker, created_time, close_time, result ('yes'/'no'), p (YES price 0-1),
    F0 (model fair value at the print), yes (1 if the taker bought YES) and
    the `signal` column (e.g. p - F0).

    When |signal| > threshold: if YES looks rich, sell YES at the next
    NO-taker print after `latency` (that print traded at the bid); if cheap,
    buy YES at the next YES-taker print (the ask). Exit `hold_min` minutes
    later at the next print on the other side, or hold to settlement when
    hold_min is None or no print comes before the close. Taker fee on every
    fill, none at settlement. `hedged_net` subtracts the move in fair value
    between the fills (a perfect delta hedge, its cost not counted)."""
    lat, wait = pd.Timedelta(latency).to_timedelta64(), pd.Timedelta(max_wait).to_timedelta64()
    rows = []
    for tk, g in prints.sort_values("created_time").groupby("ticker", sort=False):
        T = g.created_time.values
        P, F, S = g.p.values, g.F0.values, g[signal].values
        yes = g.yes.values == 1
        ct = g.close_time.iloc[0].to_datetime64()
        res = 1.0 if g.result.iloc[0] == "yes" else 0.0
        iy, ino = np.where(yes)[0], np.where(~yes)[0]
        busy = T[0] - np.timedelta64(1, "s")
        for i in np.where(np.abs(S) > threshold)[0]:
            if T[i] <= busy:
                continue
            d = -1 if S[i] > 0 else 1
            ins, outs = (ino, iy) if d < 0 else (iy, ino)
            k = np.searchsorted(T[ins], T[i] + lat)
            if k >= len(ins) or T[ins[k]] - T[i] > wait:
                continue
            j = ins[k]
            tx = min(T[j] + np.timedelta64(hold_min, "m"), ct) if hold_min else ct
            k2 = np.searchsorted(T[outs], tx) if tx < ct else len(outs)
            if k2 < len(outs) and T[outs[k2]] < ct:
                q = outs[k2]
                px, fx, cost, tq = P[q], F[q], kalshi_fee(P[j], fee_rate) + kalshi_fee(P[q], fee_rate), T[q]
            else:
                px, fx, cost, tq = res, np.nan, kalshi_fee(P[j], fee_rate), ct
            gross = d * (px - P[j])
            rows.append({"ticker": tk, "time": T[i], "side": d, "entry": P[j], "exit": px,
                         "gross": gross, "net": gross - cost,
                         "hedged_net": gross - d * (fx - F[j]) - cost, "settled": tq == ct})
            busy = tq
    return pd.DataFrame(rows)
