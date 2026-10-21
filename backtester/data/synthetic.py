"""Synthetic prices for testing the engine without network access.

These are random walks. Results on them say nothing about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData


def synthetic_bars(
    n: int = 1500,
    mu: float = 0.08,
    sigma: float = 0.2,
    start_price: float = 100.0,
    start: str = "2015-01-01",
    seed: int | None = 0,
    freq: str = "B",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ppy = 365 if freq == "D" else 252
    rets = rng.normal(mu / ppy - 0.5 * sigma**2 / ppy, sigma / np.sqrt(ppy), n)
    close = start_price * np.exp(np.cumsum(rets))
    gap = rng.normal(0, sigma / np.sqrt(ppy) / 3, n)
    open_ = np.r_[start_price, close[:-1]] * np.exp(gap)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma / np.sqrt(ppy) / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma / np.sqrt(ppy) / 2, n)))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": rng.integers(1e5, 1e6, n)},
        index=idx,
    )


def synthetic_market(symbols=("AAA", "BBB", "CCC"), seed: int = 0, **kw) -> MarketData:
    return MarketData({s: synthetic_bars(seed=seed + i, **kw) for i, s in enumerate(symbols)})
