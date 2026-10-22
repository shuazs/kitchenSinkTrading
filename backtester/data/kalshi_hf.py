"""Kalshi trade history from the Hugging Face mirror TrevorJS/kalshi-trades (Jun 2021 - Jan 2026).

The Kalshi API is blocked from the project's cloud sandbox; this mirror is
not. `markets-000{0..3}.parquet` (1.1GB) hold every market with its result;
`trades-00{00..15}.parquet` (4.6GB, ~155M prints) are only roughly in time
order, so `load_trades` scans all 16 shards one row group at a time and keeps
the requested tickers. Shared copies for non-financial markets live in
/mnt/project-files/event-markets/ (see docs/guide/data.md).
"""
from __future__ import annotations

import os
import subprocess

import pandas as pd

from .cache import cache_dir

HF = "https://huggingface.co/datasets/TrevorJS/kalshi-trades/resolve/main/"
MARKET_COLS = ["ticker", "event_ticker", "title", "status", "result", "volume", "open_time", "close_time"]
TRADE_COLS = ["ticker", "count", "yes_price", "taker_side", "created_time"]


def _get(name: str) -> str:
    path = cache_dir("kalshi_hf") / name
    if not path.exists():
        subprocess.run(["curl", "-sL", "-o", str(path), HF + name], check=True)
    return str(path)


def load_markets(refresh: bool = False) -> pd.DataFrame:
    """Every Kalshi market: MARKET_COLS plus `series` (event ticker prefix)."""
    import pyarrow.parquet as pq

    out = cache_dir("kalshi_hf") / "markets_slim.parquet"
    if not out.exists() or refresh:
        parts = [pq.read_table(_get(f"markets-{i:04d}.parquet"), columns=MARKET_COLS).to_pandas() for i in range(4)]
        m = pd.concat(parts, ignore_index=True)
        m["series"] = m["event_ticker"].str.split("-").str[0]
        m.to_parquet(out)
    return pd.read_parquet(out)


def load_trades(tickers, name: str, keep_shards: bool = False) -> pd.DataFrame:
    """All prints for `tickers` (TRADE_COLS; yes_price in cents), cached as `name`.parquet."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out = cache_dir("kalshi_hf") / f"{name}.parquet"
    if not out.exists():
        tk = pa.array(sorted(set(tickers)))
        parts = []
        for j in range(16):
            shard = _get(f"trades-{j:04d}.parquet")
            pf = pq.ParquetFile(shard)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=TRADE_COLS)
                t = t.filter(pc.is_in(t["ticker"], value_set=tk))
                if t.num_rows:
                    parts.append(t)
            if not keep_shards:
                os.remove(shard)
        pq.write_table(pa.concat_tables(parts), out)
    return pd.read_parquet(out)
