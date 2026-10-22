"""Market-wide sentiment measures that can be built from reachable data.

The classic survey and composite indices (AAII bull-bear, Baker-Wurgler,
CNN Fear & Greed, alternative.me crypto Fear & Greed, Cboe put/call) are
blocked from the cloud environment, so this module rebuilds what it can:

- `mood_index`: a Fear & Greed style composite from Yahoo prices (S&P vs its
  125-day average, VIX vs its 50-day average, VIX term structure, junk-bond
  demand, stocks vs bonds), each ranked against its own trailing year.
- `consumer_sentiment`: University of Michigan index (FRED UMCSENT), lagged
  so a month's value is only used after it was published.
- `funding_rates`: Binance BTC/ETH perpetual funding (Hugging Face mirror,
  May 2021 - Feb 2026), the crypto market's price of leverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import cache_dir

MOOD_SYMBOLS = ["SPY", "^VIX", "^VIX3M", "HYG", "IEF", "TLT"]
FUNDING = "https://huggingface.co/datasets/Torch-Trade/{coin}usdt_perp_funding_8h_05_2021_to_02_2026/resolve/main/data/train-00000-of-00001.parquet"


def mood_components(close: pd.DataFrame) -> pd.DataFrame:
    """Raw components, oriented so higher = greedier. `close` is date x symbol
    on stock-market trading days (no weekend rows) and needs MOOD_SYMBOLS."""
    c = close.ffill()
    vix = c["^VIX"]
    return pd.DataFrame({
        "momentum": c["SPY"] / c["SPY"].rolling(125).mean() - 1,
        "vix_vs_ma": -(vix / vix.rolling(50).mean() - 1),
        "term": -(vix / c["^VIX3M"] - 1),
        "junk": np.log(c["HYG"] / c["IEF"]).diff(20),
        "safe_haven": np.log(c["SPY"]).diff(20) - np.log(c["TLT"]).diff(20),
    })


def trailing_rank(x: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """Percentile (0-1) of each value within its own trailing window, today included."""
    return x.rolling(window, min_periods=min_periods).rank(pct=True)


def mood_index(close: pd.DataFrame, window: int = 252) -> pd.Series:
    """0 = extreme fear, 100 = extreme greed. Uses only data up to each close."""
    ranks = mood_components(close).apply(trailing_rank, window=window)
    return (ranks.mean(axis=1, skipna=False) * 100).dropna().rename("mood")


def consumer_sentiment(lag_months: int = 1) -> pd.Series:
    """Michigan sentiment, indexed by the first day it is safely public.

    FRED dates each value at the start of its survey month; the final number
    comes out near the end of that month, so a 1-month lag is conservative.
    """
    from .fred import load_series

    s = load_series("UMCSENT")
    s.index = s.index + pd.DateOffset(months=lag_months)
    return s.rename("umcsent")


def funding_rates(coin: str = "BTC", refresh: bool = False) -> pd.Series:
    """Daily sum of the three 8-hour funding prints (decimal per day), indexed
    by UTC date. The 16:00 print is known before Yahoo's 00:00 UTC daily close."""
    path = cache_dir("funding") / f"{coin.upper()}.parquet"
    if path.exists() and not refresh:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_parquet(FUNDING.format(coin=coin.lower()))
        raw.to_parquet(path)
    return daily_funding(raw)


def daily_funding(raw: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(raw["timestamp"]).dt.floor("h")
    return raw["funding_rate"].groupby(ts.dt.normalize().values).sum().rename("funding")
