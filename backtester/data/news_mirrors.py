"""Historical news headlines and Reddit posts from Hugging Face mirrors (no API key).

- Benzinga: ~1.57M stock-tagged headlines 2009-02 .. 2020-06 from the
  idleengine/financial-news-multisource mirror (subset benzinga_6000stocks).
  `analyst_ratings_processed` rows (685k) carry minute timestamps (ET);
  `raw_partner_headlines` rows are day-level only.
- Reddit: full Pushshift submission dumps, one zstd file per month
  (Avni2110/pushshift-reddit, 2005 .. 2025-12, 8-18 GB per month). `reddit_month`
  streams a month and keeps a few subreddits; it takes 7-14 minutes per month
  and is CPU-bound. From 2023-04 the JSON has a space after the colon.

Yahoo headlines (data/yahoo.py) only cover the last few weeks, so these are the
sources for any historical news test.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

import pandas as pd

from .cache import cache_dir

HF = "https://huggingface.co/datasets"
BENZINGA = [f"idleengine/financial-news-multisource/resolve/main/data/benzinga_6000stocks/benzinga_6000stocks.00{i}.parquet"
            for i in range(4)]
REDDIT_RS = "Avni2110/pushshift-reddit/resolve/main/original_dump/submissions/RS_{month}.zst"
FINANCE_SUBS = ("wallstreetbets", "stocks", "StockMarket", "investing", "options", "CryptoCurrency")


def load_benzinga(minute_only: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Columns published (UTC), title, symbols (list), kind. Cached as one parquet."""
    path = cache_dir("news") / "benzinga.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        parts = []
        for f in BENZINGA:
            local = cache_dir("news") / f.split("/")[-1]
            if not local.exists():
                urllib.request.urlretrieve(f"{HF}/{f}", local)
            parts.append(pd.read_parquet(local))
        raw = pd.concat(parts, ignore_index=True)
        extra = raw.extra_fields.map(json.loads)
        df = pd.DataFrame({
            "published": pd.to_datetime(raw.date, utc=True),
            "title": raw.text,
            "symbols": extra.map(lambda e: e.get("stocks", [])),
            "kind": extra.map(lambda e: e.get("raw_type")),
        })
        df.to_parquet(path)
    return df[df.kind == "analyst_ratings_processed"] if minute_only else df


def reddit_month(month: str, subs=FINANCE_SUBS, selftext_chars: int = 500, retries: int = 4) -> pd.DataFrame:
    """Stream one month of Reddit submissions ("2021-01") and keep `subs`.

    Needs `pip install zstandard`. Columns: created_utc, subreddit (lower case),
    title, selftext (truncated), score, num_comments, flair, author. Reddit
    scores are snapshots taken when the dump was made, so do not filter on them
    in a backtest (look-ahead).
    """
    import zstandard

    pat = re.compile(rb'"subreddit": ?"(?:' + b"|".join(s.encode() for s in subs) + rb')"')
    keep = {s.lower() for s in subs}
    for attempt in range(retries):
        try:
            rows = []
            with urllib.request.urlopen(f"{HF}/{REDDIT_RS.format(month=month)}", timeout=120) as resp:
                reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp, read_size=1 << 22)
                buf = b""
                while chunk := reader.read(1 << 26):
                    buf += chunk
                    cut = buf.rfind(b"\n")
                    block, buf = buf[:cut], buf[cut + 1:]
                    last_end = -1
                    # a regex over the raw block is ~10x faster than parsing every line
                    for m in pat.finditer(block):
                        s = block.rfind(b"\n", 0, m.start()) + 1
                        if s <= last_end:
                            continue
                        e = block.find(b"\n", m.end())
                        e = len(block) if e < 0 else e
                        last_end = e
                        try:
                            j = json.loads(block[s:e])
                        except ValueError:
                            continue
                        if (j.get("subreddit") or "").lower() in keep:
                            rows.append((int(float(j.get("created_utc") or 0)), j["subreddit"].lower(), j.get("title") or "",
                                         (j.get("selftext") or "")[:selftext_chars], j.get("score") or 0,
                                         j.get("num_comments") or 0, j.get("link_flair_text") or "", j.get("author") or ""))
            # the mirror sometimes ends a stream early without an error; retry a month that stops short
            last = pd.Timestamp(max((r[0] for r in rows), default=0), unit="s")
            if last < pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(1) - pd.Timedelta("1D") and attempt < retries - 1:
                raise OSError(f"stream for {month} stopped at {last}")
            return pd.DataFrame(rows, columns=["created_utc", "subreddit", "title", "selftext", "score",
                                               "num_comments", "flair", "author"])
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not stream Reddit dump for {month}")
