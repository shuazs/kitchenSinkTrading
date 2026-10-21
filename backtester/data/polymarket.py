"""Polymarket prices: a free on-chain history snapshot and the live public API.

- History: SII-WANGZJ/Polymarket_data on Hugging Face, every CLOB fill from
  Nov 2022 to the snapshot date (2026-10-04 in the release used here),
  MIT licensed. `quant.parquet` (45GB) normalizes each fill to the YES price;
  `load_fills` reads only four of its columns, one row group at a time, so
  it never downloads the whole file (about 20 minutes for 2024-10 onwards).
- Live: gamma-api.polymarket.com, free and keyless. It is blocked from the
  project's cloud sandbox, so `live_markets` is untested there; run it at home.

The Iran escalation index turns hundreds of dated contracts ("US strikes
Iran by March 31?", "US x Iran ceasefire by April 7?") into one hourly
number: each market's price change, signed so that escalation is positive,
weighted by its share of the last week's dollar volume.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import requests

from .cache import cache_dir

HF_REPO = "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data/resolve/main/"
GAMMA = "https://gamma-api.polymarket.com"

ESCALATION = [
    "strikes iran", "forces enter iran", "invade iran", "closes its airspace", "close the strait",
    "military action against iran before", "next strikes iran", "no longer under iranian control",
    "charges hormuz fees", "seize another oil tanker",
]
DE_ESCALATION = [
    "ceasefire", "peace deal", "diplomatic meeting", "meeting by", "returns to normal", "blockade lifted",
    "end of military operations", "military action against iran ends", "conflict ends", "agreement",
    "end of iranian blockade", "escorts commercial ship", "obtains iranian enriched uranium",
]


def iran_sign(question: str, event_title: str = "") -> int:
    """+1 if YES means escalation with Iran, -1 if it means de-escalation, 0 to skip.

    Leadership and regime-change markets are skipped: their effect on oil
    could go either way. So are sports markets that mention Iran.
    """
    s = f"{question} {event_title}".lower()
    if not re.search(r"iran|hormuz|kharg", s) or "world cup" in s or re.search(r" vs\. |win on", s):
        return 0
    if re.search(r"khamenei|regime fall|leader|pahlavi", s):
        return 0
    if any(k in s for k in DE_ESCALATION):
        return -1
    if any(k in s for k in ESCALATION):
        return 1
    return 0


def load_markets(refresh: bool = False) -> pd.DataFrame:
    """Snapshot market metadata: id, question, event_title, volume (USD), end_date, ..."""
    path = cache_dir("polymarket") / "markets.parquet"
    if not path.exists() or refresh:
        r = requests.get(HF_REPO + "markets.parquet", timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pd.read_parquet(path)


def iran_markets(markets: pd.DataFrame, min_volume: float = 1e6) -> pd.DataFrame:
    m = markets[markets["volume"] >= min_volume].copy()
    m["sign"] = [iran_sign(q, e) for q, e in zip(m["question"], m["event_title"].fillna(""))]
    return m[m["sign"] != 0]


def load_fills(market_ids, since="2024-10-01", name="fills", refresh: bool = False) -> pd.DataFrame:
    """Fills for the given markets since a date: time (UTC), market_id, price (YES), usd_amount."""
    import fsspec
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = cache_dir("polymarket") / f"{name}.parquet"
    if not path.exists() or refresh:
        ids = pa.array(sorted({str(i) for i in market_ids}))
        t0 = int(pd.Timestamp(since, tz="UTC").timestamp())
        pf = pq.ParquetFile(fsspec.filesystem("https").open(HF_REPO + "quant.parquet", block_size=8 << 20))
        parts = []
        for i in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(i).column(0).statistics  # timestamp, file is in time order
            if stats is not None and stats.max < t0:
                continue
            t = pf.read_row_group(i, columns=["timestamp", "market_id", "price", "usd_amount"])
            parts.append(t.filter(pc.is_in(t["market_id"], value_set=ids)))
        pq.write_table(pa.concat_tables(parts), path)
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df.pop("timestamp"), unit="s")
    return df[df["time"] >= pd.Timestamp(since)].sort_values("time", ignore_index=True)


def hourly_prices(fills: pd.DataFrame, min_usd: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last YES price and dollar volume per market per hour (prices carried up to 24h)."""
    f = fills[fills["usd_amount"] >= min_usd].assign(hour=lambda d: d["time"].dt.floor("h"))
    last = f.groupby(["hour", "market_id"])["price"].last().unstack()
    vol = f.groupby(["hour", "market_id"])["usd_amount"].sum().unstack()
    idx = pd.date_range(last.index.min(), last.index.max(), freq="h")
    return last.reindex(idx).ffill(limit=24), vol.reindex(idx).fillna(0.0)


def escalation_index(prices: pd.DataFrame, volume: pd.DataFrame, signs: pd.Series, window: int = 24 * 7) -> pd.Series:
    """Volume-weighted, signed change in YES prices per period; positive = escalation.

    `signs` maps market id -> +1/-1. Weights are each market's share of the
    trailing `window` periods of dollar volume among markets with a price.
    """
    cols = [c for c in prices.columns if signs.get(c, 0) != 0]
    p, v = prices[cols], volume.reindex(columns=cols).fillna(0.0)
    s = signs.reindex(cols).astype(float)
    w = v.rolling(window, min_periods=1).sum().where(p.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (p.diff().clip(-0.5, 0.5) * s * w).sum(axis=1, min_count=1).fillna(0.0).rename("escalation")


def live_markets(pattern: str = r"iran|hormuz|kharg", min_volume: float = 1e6, timeout: float = 20) -> pd.DataFrame:
    """Open Polymarket markets whose event or question matches `pattern`, with a mid price.

    Pages through gamma-api /events. Columns: id, question, event_title,
    price (bid/ask mid, else last trade), volume_1wk, volume.
    """
    rows, offset = [], 0
    while True:
        r = requests.get(f"{GAMMA}/events", params={"closed": "false", "limit": 500, "offset": offset}, timeout=timeout)
        r.raise_for_status()
        events = r.json()
        for ev in events:
            for m in ev.get("markets") or []:
                text = f"{m.get('question', '')} {ev.get('title', '')}"
                if m.get("closed") or not re.search(pattern, text, re.I):
                    continue
                bid, ask = m.get("bestBid"), m.get("bestAsk")
                price = (float(bid) + float(ask)) / 2 if bid not in (None, "") and ask not in (None, "") else None
                if price is None:
                    try:
                        price = float(m.get("lastTradePrice") or json.loads(m.get("outcomePrices") or "[]")[0])
                    except (ValueError, IndexError):
                        continue
                rows.append({
                    "id": str(m.get("id")), "question": m.get("question", ""), "event_title": ev.get("title", ""),
                    "price": price, "volume_1wk": float(m.get("volume1wk") or 0), "volume": float(m.get("volumeNum") or m.get("volume") or 0),
                })
        if len(events) < 500:
            break
        offset += 500
    df = pd.DataFrame(rows, columns=["id", "question", "event_title", "price", "volume_1wk", "volume"])
    return df[df["volume"] >= min_volume].drop_duplicates("id").reset_index(drop=True)


def snapshot_change(old: pd.DataFrame, new: pd.DataFrame) -> float:
    """Escalation index between two `live_markets` snapshots (same weighting idea as the backtest)."""
    m = new.merge(old[["id", "price"]], on="id", suffixes=("", "_old"))
    m["sign"] = [iran_sign(q, e) for q, e in zip(m["question"], m["event_title"])]
    m = m[m["sign"] != 0]
    if m.empty or m["volume_1wk"].sum() <= 0:
        return 0.0
    w = m["volume_1wk"] / m["volume_1wk"].sum()
    return float((np.clip(m["price"] - m["price_old"], -0.5, 0.5) * m["sign"] * w).sum())
