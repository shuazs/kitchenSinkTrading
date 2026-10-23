"""Spin-off event study with a survivorship bound (alpha hunt #4).

Children come from Form 10-12B registrants (`data.edgar_forms`). A child's
start date is its first Yahoo bar, kept only if it falls within
[-30, +540] days of the first 10-12B. Buy at the open `lag` trading days
later, hold `hold` days, compare to a benchmark (IWM) over the same dates.

Yahoo has no delisted tickers, so registrants that filed 10-Ks/10-Qs but
have no ticker today are "dead" children with unknown returns.
`survivorship_bound` adds them back at assumed excess returns by fate
(acquired, bankrupt, other).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEAD_EXCESS = {"acquired": 0.0, "bankrupt": -1.0, "other": -0.5}


def child_events(first_filing: pd.Series, opens: pd.DataFrame, closes: pd.DataFrame, bench: str = "IWM",
                 lags=(1, 21), holds=(126, 252), min_start="2005-03-01") -> pd.DataFrame:
    """One row per (child, lag, hold): entry date, return, benchmark return, excess.
    `first_filing` maps ticker -> date of its first 10-12B."""
    rows = []
    for t, f in first_filing.items():
        if t not in closes or closes[t].notna().sum() < 30:
            continue
        s = closes[t].dropna()
        d0, f = s.index[0], pd.Timestamp(f)
        if d0 <= pd.Timestamp(min_start) or not (f - pd.Timedelta(days=30) <= d0 <= f + pd.Timedelta(days=540)):
            continue
        for lag in lags:
            if len(s) <= lag + 20:
                continue
            entry = s.index[lag]
            for hold in holds:
                xi = min(lag + hold, len(s) - 1)
                exit_ = s.index[xi]
                ret = s.iloc[xi] / opens[t].loc[entry] - 1
                b = closes[bench].loc[exit_] / opens[bench].loc[entry] - 1
                rows.append(dict(ticker=t, start=d0, lag=lag, hold=hold, entry=entry, days=xi - lag,
                                 ret=ret, bench=b, excess=ret - b))
    return pd.DataFrame(rows)


def dead_fate(acquired: bool, bankrupt: bool) -> str:
    return "bankrupt" if bankrupt else ("acquired" if acquired else "other")


def survivorship_bound(excess: pd.Series, dead_fates: pd.Series, assume: dict | None = None, clip=(-1, 3)) -> dict:
    """Mean excess and t-stat for survivors plus dead children at assumed excess returns."""
    assume = DEAD_EXCESS if assume is None else assume
    x = pd.concat([excess.clip(*clip), dead_fates.map(assume).astype(float)], ignore_index=True).dropna()
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {"n_alive": len(excess), "n_dead": len(dead_fates), "mean_excess": x.mean(), "t": t}
