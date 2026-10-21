"""Headline sentiment with FinBERT (ProsusAI/finbert from Hugging Face).

Needs `pip install transformers torch`. Scores are P(positive) - P(negative)
in [-1, 1].
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)


def score(texts: list[str], batch_size: int = 32) -> list[float]:
    out = []
    for res in _pipeline()(list(texts), batch_size=batch_size):
        p = {r["label"].lower(): r["score"] for r in res}
        out.append(p.get("positive", 0.0) - p.get("negative", 0.0))
    return out


def daily_sentiment(headlines: pd.DataFrame, market_close_hour_utc: int = 20) -> pd.DataFrame:
    """Average score per symbol per trading day.

    A headline published after the US close counts toward the next day, so
    the signal is only used once the news was public.
    """
    df = headlines.copy()
    df["score"] = score(df["title"].fillna("").tolist())
    ts = pd.to_datetime(df["published"], utc=True)
    day = ts.dt.tz_convert(None).dt.normalize()
    after_close = ts.dt.hour >= market_close_hour_utc
    df["day"] = day + pd.to_timedelta(after_close.astype(int), unit="D")
    return df.pivot_table(index="day", columns="symbol", values="score", aggfunc="mean")
