"""Free 1-minute bars from Hugging Face mirrors (no API key).

Alpaca is the clean source for US minute bars (see alpaca.py), but these
mirrors are reachable from anywhere and cover 2021-2026:

- SPX: cash S&P 500 index, regular hours (thillsss/SPX-MES-VIX-data)
- MES: Micro E-mini S&P futures, continuous and back-adjusted, 23h session
  (same repo; the RTH and overnight files are merged). Levels are not index
  levels because of the back-adjustment, so use returns.
- NQ: E-mini Nasdaq futures, 2024-01 to 2025-07 (mdelcristo/NQ-F_1min_OHLCV_Parquet)
- BTC: Binance BTCUSDT spot (vaquum/binance_btcusdt_1m_klines)

All frames come back indexed by bar start time in UTC. Checked on 2024-2025:
1-minute returns correlate 0.97 SPX~MES and 0.91 MES~NQ at lag 0 and about 0
at +-1 minute. Do not use mnq_minute_data_CT.txt from the thillsss repo; its
timestamps are a minute off.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .cache import cache_dir

HF = "https://huggingface.co/datasets"
SOURCES = {
    "SPX": ["thillsss/SPX-MES-VIX-data/resolve/main/SPX_full_1min_CT.txt"],
    "MES": ["thillsss/SPX-MES-VIX-data/resolve/main/MES_full_1min_continuous_absolute_adjusted_CT_market_open.txt",
            "thillsss/SPX-MES-VIX-data/resolve/main/MES_full_1min_continuous_absolute_adjusted_CT_market_close.txt"],
    "NQ": [f"mdelcristo/NQ-F_1min_OHLCV_Parquet/resolve/main/NQ_1min_{y}.parquet" for y in (2024, 2025)],
    "BTC": ["vaquum/binance_btcusdt_1m_klines/resolve/main/btcusdt_1m_kline_20200101_to_20261020.parquet"],
}


def _download(path: str) -> Path:
    out = cache_dir("hf_minute") / path.split("/")[-1]
    if not out.exists():
        with requests.get(f"{HF}/{path}", stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(out)
    return out


def _central_csv(path: Path, start: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["datetime"])
    d = d[d.datetime >= start]
    d["ts"] = d.datetime.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return d.dropna(subset=["ts"]).drop(columns="datetime").set_index("ts")


def load_minute(symbol: str, start: str = "2021-06-01") -> pd.DataFrame:
    """OHLC(V) 1-minute bars for SPX, MES, NQ or BTC, UTC bar-start index."""
    files = [_download(p) for p in SOURCES[symbol]]
    if symbol in ("SPX", "MES"):
        d = pd.concat([_central_csv(f, start) for f in files])
    elif symbol == "NQ":
        d = pd.concat([pd.read_parquet(f) for f in files])
        d = d.assign(ts=d.timestamp.dt.tz_localize("UTC")).drop(columns="timestamp").set_index("ts")
    else:
        d = pd.read_parquet(files[0], columns=["datetime", "open", "high", "low", "close", "volume"])
        d = d.rename(columns={"datetime": "ts"}).set_index("ts")
    d = d[d.index >= pd.Timestamp(start, tz="UTC")].sort_index()
    d = d[~d.index.duplicated()]
    if symbol == "SPX":
        t = d.index.tz_convert("America/New_York").time
        d = d[(t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())]
    return d
