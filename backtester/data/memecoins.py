"""Free pump.fun meme coin data from public Hugging Face datasets.

- Smurfetc/solana-memecoin-calls: 10k+ timestamped Telegram "calls" with
  the market cap at the call and the peak multiple reached afterwards, with
  the losing calls kept in (CC0).
- loopholetape/pumpfun-launches: every pump.fun launch seen by an on-chain
  recorder with outcomes: graduated, rugged, peak and last SOL reserve of
  the bonding curve (CC-BY-4.0). Use the default `launches` subset (from
  2026-09-27); earlier days undercount buys.

Both are snapshots, so these are research sets, not a live feed.
"""
from __future__ import annotations

import pandas as pd
import requests

from .cache import cache_dir

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
CALLS_REPO = "Smurfetc/solana-memecoin-calls"
LAUNCHES_REPO = "loopholetape/pumpfun-launches"
LAUNCH_COLS = ["mint", "launched_at", "graduated", "rugged", "peak_rs", "terminal_rs", "mayhem", "quote_asset", "decoder_complete"]


def _fetch(repo: str, path: str, refresh: bool = False):
    local = cache_dir("hf") / repo.replace("/", "__") / path
    if not local.exists() or refresh:
        local.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(HF.format(repo=repo, path=path), timeout=120)
        r.raise_for_status()
        local.write_bytes(r.content)
    return local


def load_calls(refresh: bool = False) -> pd.DataFrame:
    """One row per call: mint, symbol, call time (UTC), market cap at the
    call in dollars (`mc`) and the peak multiple of that cap afterwards."""
    df = pd.read_json(_fetch(CALLS_REPO, "calls.jsonl", refresh), lines=True)
    df["time"] = pd.to_datetime(df["utc"]).dt.tz_localize(None)
    return df.rename(columns={"sym": "symbol"})[["time", "mint", "symbol", "mc", "peak"]].sort_values("time")


def load_launches(days, refresh: bool = False) -> pd.DataFrame:
    """Launch outcomes for the given UTC days (e.g. pd.date_range(...))."""
    frames = []
    for d in pd.DatetimeIndex(days):
        path = _fetch(LAUNCHES_REPO, f"data/launches/{d:%Y-%m-%d}.parquet", refresh)
        frames.append(pd.read_parquet(path, columns=LAUNCH_COLS))
    return pd.concat(frames, ignore_index=True)
