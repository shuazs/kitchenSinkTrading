"""Market-wide sentiment as a forecaster and as an exposure overlay on the
repo's keeper strategies.

    table = bucket_returns(mood, spy_close, horizon=21, start="2007-06-01", end="2018-12-31")
    mult = threshold_mult(mood, low=10, low_mult=2.0)       # contrarian: lever up in extreme fear
    data.extra["mood_mult"] = mult
    strat = SentimentOverlay(base=VolTarget(...), mult_key="mood_mult", symbols=["UPRO"])

Sentiment is read at close t and the scaled weights trade at open t+1
through the engine, as for every other strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import MarketData, Strategy


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from close t to close t+horizon."""
    return np.log(close.shift(-horizon) / close)


def bucket_returns(signal: pd.Series, close: pd.Series, horizon: int, start=None, end=None, n: int = 5) -> pd.DataFrame:
    """Forward returns by signal quantile (0 = lowest signal).

    `mean_pct` and `hit` use every day; `t` uses non-overlapping samples
    (every `horizon` days), so it is not inflated by overlap. `vol_pct` is
    the per-period standard deviation, to compare return per unit of risk.
    """
    df = pd.DataFrame({"s": signal.reindex(close.index).ffill(), "f": forward_returns(close, horizon)})
    df = df.loc[start:end].dropna()
    df["b"] = pd.qcut(df["s"].rank(method="first"), n, labels=False)
    g = df.groupby("b")
    sub = df.iloc[::horizon].groupby("b")["f"]
    out = pd.DataFrame({
        "from": g["s"].min(),
        "mean_pct": g["f"].mean() * 100,
        "vol_pct": g["f"].std() * 100,
        "hit": g["f"].apply(lambda x: (x > 0).mean()),
        "t": sub.mean() / sub.std() * np.sqrt(sub.count()),
        "days": g.size(),
    })
    out["ret_per_vol"] = out["mean_pct"] / out["vol_pct"]
    return out


def threshold_mult(signal: pd.Series, low: float | None = None, high: float | None = None,
                   low_mult: float = 1.0, high_mult: float = 1.0) -> pd.Series:
    """Exposure multiplier: `low_mult` when signal <= low, `high_mult` when
    signal >= high, 1 otherwise (and 1 before the signal exists)."""
    m = pd.Series(1.0, index=signal.index)
    if low is not None:
        m[signal <= low] = low_mult
    if high is not None:
        m[signal >= high] = high_mult
    return m


class SentimentOverlay(Strategy):
    """Scale a base strategy's weights in `symbols` by a sentiment multiplier.

    base: any Strategy. mult_key: key in `data.extra` of a date-indexed
    multiplier Series (1 = unchanged, 0 = flat, 2 = double). Or set
    signal_key (a sentiment Series in `data.extra`) with low/high/low_mult/
    high_mult and the multiplier is `threshold_mult` of it, which lets
    `research.walk_forward` pick thresholds. symbols: which
    columns to scale (None = all). max_weight caps each scaled weight and
    max_gross the total. Whatever is freed or borrowed moves through
    `risk_off` (e.g. SHY) if given, else cash.
    """

    name = "sentiment_overlay"
    params = {"base": None, "mult_key": "sentiment_mult", "signal_key": None, "low": None, "high": None,
              "low_mult": 1.0, "high_mult": 1.0, "symbols": None, "risk_off": None,
              "max_weight": 1.0, "max_gross": 1.0}

    def multiplier(self, data: MarketData) -> pd.Series:
        if self.p["signal_key"]:
            p = self.p
            return threshold_mult(data.extra[p["signal_key"]], p["low"], p["high"], p["low_mult"], p["high_mult"])
        return data.extra[self.p["mult_key"]]

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        w = self.p["base"].generate_targets(data).reindex(data.index).ffill().fillna(0.0)
        mult = self.multiplier(data).reindex(data.index).ffill().fillna(1.0)
        cols = [c for c in (self.p["symbols"] or w.columns) if c in w.columns]
        before = w[cols].sum(axis=1)
        w[cols] = w[cols].mul(mult, axis=0).clip(-self.p["max_weight"], self.p["max_weight"])
        gross = w.abs().sum(axis=1)
        over = (gross / self.p["max_gross"]).clip(lower=1.0)
        w[cols] = w[cols].div(over, axis=0)
        if self.p["risk_off"]:
            ro = self.p["risk_off"]
            if ro not in w:
                w[ro] = 0.0
            w[ro] = (w[ro] + before - w[cols].sum(axis=1)).clip(lower=0.0)
        return w
