"""Event-window studies for scheduled flows (index rebalances, futures rolls,
tax dates, expiries).

Each test anchors on a calendar date known in advance, takes the asset's (or
spread's) close-to-close return over trading-day offsets [start, end] around
it, and compares fit-period and holdout means. Dates are known ahead, so the
trades are market-on-close orders at the window edges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = {"2008-16": ("2008-01-01", "2016-12-31"), "2017-24H1": ("2017-01-01", "2024-06-30"),
          "holdout": ("2024-07-01", "2100-01-01")}


def anchor_positions(index: pd.DatetimeIndex, dates, how: str = "on_or_after") -> list[int]:
    """Positions in `index` of each date, moved to the next ("on_or_after") or
    previous ("on_or_before") trading day when the date is not one."""
    side = "left" if how == "on_or_after" else "right"
    out = []
    for d in pd.DatetimeIndex(dates):
        p = index.searchsorted(d, side=side)
        if how == "on_or_before":
            p -= 1
        if 0 <= p < len(index):
            out.append(int(p))
    return sorted(set(out))


def nth_business_day(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """The n-th trading day (1-based; negative counts from month end) of each
    month in `index`."""
    s = pd.Series(index, index=index)
    g = s.groupby([index.year, index.month])
    pick = g.nth(n - 1 if n > 0 else n)
    return pd.DatetimeIndex(pick.values)


def event_returns(close: pd.Series, anchors: list[int], start: int, end: int,
                  bench: pd.Series | None = None) -> pd.Series:
    """Return from the close at anchor+start to the close at anchor+end (so
    the first day earned is anchor+start+1), minus `bench` over the same
    days. Indexed by the anchor date."""
    idx = close.index
    rows = {}
    for p in anchors:
        a, b = p + start, p + end
        if a < 0 or b >= len(idx) or np.isnan(close.iloc[a]) or np.isnan(close.iloc[b]):
            continue
        r = close.iloc[b] / close.iloc[a] - 1
        if bench is not None:
            if np.isnan(bench.iloc[a]) or np.isnan(bench.iloc[b]):
                continue
            r -= bench.iloc[b] / bench.iloc[a] - 1
        rows[idx[p]] = r
    return pd.Series(rows, dtype=float)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def split_stats(r: pd.Series, splits: dict = SPLITS) -> pd.DataFrame:
    """Mean (bp), t, n and share of positive events per period."""
    if r.empty:
        r = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    rows = {}
    for name, (a, b) in {"all": ("1900-01-01", "2100-01-01"), **splits}.items():
        x = r[a:b]
        rows[name] = {"mean_bp": x.mean() * 1e4, "t": tstat(x), "n": len(x), "pos": (x > 0).mean()}
    return pd.DataFrame(rows).T


def placebo(close: pd.Series, anchors: list[int], start: int, end: int, shifts=(-15, -10, 10, 15),
            bench: pd.Series | None = None) -> pd.Series:
    """Mean event return (bp) with every anchor moved by each shift."""
    return pd.Series({s: event_returns(close, [p + s for p in anchors], start, end, bench).mean() * 1e4
                      for s in shifts})


def window_mask(index: pd.DatetimeIndex, anchors: list[int], start: int, end: int) -> pd.Series:
    """True on days whose close-to-close return falls inside a window."""
    m = np.zeros(len(index), bool)
    for p in anchors:
        m[max(p + start + 1, 0): min(p + end + 1, len(index))] = True
    return pd.Series(m, index=index)


def last_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - 4) % 7)


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def russell_recon(y: int) -> pd.Timestamp:
    """Last Friday of June, or the Friday before when that is the 29th/30th."""
    d = last_friday(y, 6)
    return d - pd.Timedelta(days=7) if d.day >= 29 and y >= 2008 else d


def vix_expiry(y: int, m: int) -> pd.Timestamp:
    """Wednesday 30 days before the next month's third Friday."""
    ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
    return third_friday(ny, nm) - pd.Timedelta(days=30)
