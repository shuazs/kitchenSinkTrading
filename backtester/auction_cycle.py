"""Treasury auction cycle (Lou, Yan and Zhang 2013, "Anticipated and
Repeated Shocks in Liquid Markets").

Dealers must absorb each new 10y/30y issue, so long bonds cheapen in the
days before the auction and recover after it. Auctions of the same month
(10y and 30y on consecutive days) form one cycle. The trade: be out of the
long-bond ETF (or short it) from the close `days` trading days before a
cycle's first auction to the close of its last auction day. Dates are known
in advance, so both legs can be market-on-close orders; that is why this is
a standalone simulator rather than a `Strategy` (the engine fills at the
next open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cycles(auction_dates, index: pd.DatetimeIndex, gap: int = 5) -> list[tuple[int, int]]:
    """(first, last) positions in `index` of auction days less than `gap`
    trading days apart. Dates that are not trading days are dropped."""
    pos = sorted({index.get_loc(d) for d in pd.DatetimeIndex(auction_dates) if d in index})
    out: list[list[int]] = []
    for p in pos:
        if out and p - out[-1][1] <= gap:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [tuple(c) for c in out]


def window_mask(index: pd.DatetimeIndex, cyc, days: int = 5) -> pd.Series:
    """True on days whose close-to-close return is earned inside a window
    (the day after the exit close through the last auction day)."""
    m = np.zeros(len(index), bool)
    for first, last in cyc:
        m[max(first - days + 1, 0) : last + 1] = True
    return pd.Series(m, index=index)


def window_returns(close: pd.Series, cyc, days: int = 5, cash: pd.Series | None = None) -> pd.Series:
    """Per-cycle return of the bond from the exit close to the last auction
    close, minus `cash` over the same days (indexed by exit date)."""
    idx = close.index
    rows = {}
    for first, last in cyc:
        a = first - days
        if a < 0 or np.isnan(close.iloc[a]):
            continue
        r = close.iloc[last] / close.iloc[a] - 1
        if cash is not None:
            r -= cash.iloc[last] / cash.iloc[a] - 1
        rows[idx[a]] = r
    return pd.Series(rows, dtype=float)


def simulate(close: pd.DataFrame, auction_dates, bond: str = "TLT", days: int = 5, short: str | None = None,
             short_weight: float = 1.0, cash: str | None = None, cost_bps: float = 3.0) -> pd.Series:
    """Daily returns of: 100% `bond` outside windows; inside them cash (0%,
    or `cash` ETF) or `short_weight` of an inverse ETF `short`. Trades at the
    close, `cost_bps` per side on the value traded."""
    close = close.dropna(subset=[bond])
    idx = close.index
    cyc = cycles(auction_dates, idx)
    inside = window_mask(idx, cyc, days)
    rets = close.pct_change(fill_method=None).fillna(0.0)
    w_bond = (~inside).astype(float)
    r = w_bond * rets[bond]
    w_other = pd.Series(0.0, index=idx)
    if short:
        w_other = inside.astype(float) * short_weight
        r = r + w_other * rets[short]
    elif cash:
        w_other = inside.astype(float)
        r = r + w_other * rets[cash]
    # a position change between day t-1 and t happens at close t-1
    turnover = w_bond.diff().abs().fillna(0.0) + w_other.diff().abs().fillna(0.0)
    return r - turnover * cost_bps / 1e4
