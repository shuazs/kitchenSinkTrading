"""Kalshi prediction-market features, weekly, for the signal-combination panel.

The daily file comes from the "Leading Kalshi markets for stocks" research
(Kalshi trade history, roughly August 2021 to January 2026): for each market
family, the day-over-day change in the implied value of the nearest live
event, sampled at 20:00 UTC and divided by its trailing median absolute
change, plus the NBER-recession contract's price. Kalshi's API is not
reachable from the cloud environment, so the file is read from disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Families with enough history to use (gas, WTI, 10y note, jobless claims came
# back empty; S&P 500 and TSA ranges have under a year of daily values).
FAMILIES = ["cpi", "cpiyoy", "core", "payrolls", "u3", "gdp", "fed"]
KALSHI_FEATURES = [f"k_{f}" for f in FAMILIES] + ["k_recession_p", "k_recession_chg"]


def load_daily_features(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).astype("datetime64[ns]")
    return df.sort_index()


def weekly_features(daily: pd.DataFrame, weeks: pd.DatetimeIndex, lag_days: int = 1, clip: float = 10.0) -> pd.DataFrame:
    """KALSHI_FEATURES on the panel's week-end dates.

    Each family's feature is the sum of its scaled daily changes over the
    week (each day clipped at +-clip), so a week with no repricing is 0. The
    recession contract gives its last price and its change over the week.
    Everything is lagged `lag_days` so a snapshot taken near the close is
    only used from the next session on.
    """
    d = daily.replace([np.inf, -np.inf], np.nan).shift(lag_days, freq="D")
    weeks = pd.DatetimeIndex(weeks)
    # Bucket each day into the first week end on or after it.
    pos = weeks.searchsorted(d.index)
    ok = pos < len(weeks)
    d, bucket = d[ok], weeks[pos[ok]]
    out = pd.DataFrame(index=weeks)
    for f in FAMILIES:
        col = f"{f}_chg"
        s = d[col].clip(-clip, clip) if col in d else pd.Series(dtype=float)
        out[f"k_{f}"] = s.groupby(bucket).sum(min_count=1).reindex(weeks) if len(s) else np.nan
    p = d["recession_p"].groupby(bucket).last().reindex(weeks).ffill(limit=4)
    out["k_recession_p"] = p
    out["k_recession_chg"] = p.diff()
    # Outside Kalshi's history every feature is unknown, not zero news.
    first, last = daily.index.min(), daily.index.max() + pd.Timedelta(days=lag_days + 7)
    out.loc[(out.index < first) | (out.index > last)] = np.nan
    return out[KALSHI_FEATURES]
