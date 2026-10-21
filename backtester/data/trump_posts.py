"""Donald Trump's public posts: first-term tweets and Truth Social.

Sources, all free:
- Tweets, 20 Jan 2017 to 8 Jan 2021: MarkHershey/CompleteTrumpTweetsArchive on
  GitHub. Its "Time" column is in UTC+8 for most rows, so the time is taken
  from the tweet id instead (Twitter ids encode the UTC millisecond).
- Truth Social, Feb 2022 on: CNN's archive (updated every five minutes) when
  reachable, else stiles/trump-truth-social-archive on GitHub, which stopped
  updating in May 2026. CNN's host is blocked from some cloud sandboxes.

There are no posts from 9 Jan 2021 to 13 Feb 2022 (he was off Twitter and
Truth Social had not launched), so any signal built from these is silent then.

Posts are aligned to the trading session whose close is the first one at or
after the post: a post at 15:59 ET counts for that day, one at 16:01 ET or on
a weekend for the next session. A signal formed from these at the close
trades at the next open, as everywhere else in the engine.
"""
from __future__ import annotations

import html
import io
import re

import numpy as np
import pandas as pd
import requests

from .cache import cache_dir

TWEETS_URL = "https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv"
TRUTH_URLS = [
    "https://ix.cnn.io/data/truth-social/truth_archive.csv",
    "https://raw.githubusercontent.com/stiles/trump-truth-social-archive/main/data/truth_archive.csv",
]
TWITTER_EPOCH_MS = 1288834974657

# Keyword sets for daily counts. Lower-cased text, regex.
TARIFF = r"\btariff"
TRADE = r"\btariff|\btrade (?:deal|war|deficit)|\bchina\b|\bpresident xi\b"
MARKET = TRADE + r"|\bfed\b|\bpowell\b|interest rate|stock market|\biran\b|\bsanction"


def _get(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _clean(text: pd.Series) -> pd.Series:
    text = text.fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True)
    return text.map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()


def load_tweets(refresh: bool = False) -> pd.DataFrame:
    path = cache_dir("trump") / "tweets_2017_2021.csv"
    if refresh or not path.exists():
        path.write_text(_get(TWEETS_URL))
    df = pd.read_csv(path, skipinitialspace=True)
    ids = df["Tweet URL"].str.extract(r"status/(\d+)")[0].astype("int64").to_numpy()
    ms = np.right_shift(ids, 22) + TWITTER_EPOCH_MS
    return pd.DataFrame(
        {
            "time": pd.to_datetime(ms, unit="ms", utc=True),
            "platform": "twitter",
            "text": _clean(df["Tweet Text"]),
            "url": df["Tweet URL"],
        }
    )


def load_truths(refresh: bool = False) -> pd.DataFrame:
    path = cache_dir("trump") / "truth_social.csv"
    if refresh or not path.exists():
        errors = []
        for url in TRUTH_URLS:
            try:
                path.write_text(_get(url))
                break
            except requests.RequestException as e:
                errors.append(f"{url}: {e}")
        else:
            raise RuntimeError("no Truth Social archive reachable:\n" + "\n".join(errors))
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(df["created_at"], utc=True, format="ISO8601"),
            "platform": "truth_social",
            "text": _clean(df["content"]),
            "url": df["url"],
        }
    )


def load_posts(refresh: bool = False) -> pd.DataFrame:
    """All posts, oldest first: time (UTC), platform, text, url."""
    posts = pd.concat([load_tweets(refresh), load_truths(refresh)], ignore_index=True)
    return posts.drop_duplicates("url").sort_values("time", ignore_index=True)


def coverage(posts: pd.DataFrame) -> pd.DataFrame:
    """First and last post per platform, so backtests stay inside the data."""
    return posts.groupby("platform")["time"].agg(["min", "max", "count"])


def sessions(times: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Position in `index` of the session each post first counts for (-1 if after the last bar)."""
    et = pd.DatetimeIndex(times).tz_convert("America/New_York").tz_localize(None)
    # Shift 16:00 ET to midnight, so anything after the close rolls to the next day.
    day = (et + pd.Timedelta(hours=8)).normalize()
    pos = index.searchsorted(day, side="left")
    return pd.Series(np.where(pos < len(index), pos, -1), index=times.index)


def daily_counts(posts: pd.DataFrame, index: pd.DatetimeIndex, pattern: str = TARIFF) -> pd.Series:
    """Number of posts matching `pattern` (regex, case-insensitive) per trading session."""
    hit = posts[posts["text"].str.contains(pattern, case=False, regex=True)]
    pos = sessions(hit["time"], index)
    counts = np.bincount(pos[pos >= 0], minlength=len(index))
    return pd.Series(counts, index=index, name="posts")


def covered(posts: pd.DataFrame, index: pd.DatetimeIndex, max_gap_days: int = 30) -> pd.Series:
    """True on sessions inside a stretch of the archive with no gap longer than
    `max_gap_days`. Use it to tell "no posts" apart from "no data"."""
    t = posts["time"].sort_values().reset_index(drop=True)
    gaps = t.diff() > pd.Timedelta(days=max_gap_days)
    starts = pd.concat([t.iloc[:1], t[gaps]])
    ends = pd.concat([t[gaps.shift(-1, fill_value=False)], t.iloc[-1:]])
    out = pd.Series(False, index=index)
    for a, b in zip(sessions(starts, index), sessions(ends, index)):
        out.iloc[a : (b if b >= 0 else len(index)) + 1] = True
    return out
