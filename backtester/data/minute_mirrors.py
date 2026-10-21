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

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .cache import cache_dir

# A folder of ready-made <SYMBOL>.parquet files (UTC bar-start index), used
# before downloading. The project's shared folder also holds QQQ, ETH and NDX.
LOCAL_DIR = os.environ.get("KS_MINUTE_DIR", "/mnt/project-files/minute-data")

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
    """OHLC(V) 1-minute bars for SPX, MES, NQ or BTC (also QQQ, ETH when
    LOCAL_DIR has them), UTC bar-start index."""
    local = Path(LOCAL_DIR) / f"{symbol}.parquet"
    if local.exists():
        d = pd.read_parquet(local)
        d.index = pd.DatetimeIndex(d.index).as_unit("ns")
        return d[d.index >= pd.Timestamp(start, tz="UTC")]
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


def _new_york(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    return d


def _rth(d: pd.DataFrame) -> pd.DataFrame:
    m = d.index.hour * 60 + d.index.minute
    return d[(m >= 570) & (m < 960)]


def rth_bars(symbol: str, start: str = "2021-06-01") -> pd.DataFrame:
    """Regular-session bars (09:30-15:59 bar starts, New York, tz-naive), the
    format `intraday.minute_grid` and `alpaca.load_minute_bars` use.

    - "SPY": a SPY stand-in from the cash S&P 500 index. The index's 09:30
      print is a stale copy of the previous close, so the first bar's open is
      replaced by the previous close times the MES overnight move, and MES
      volume stands in for volume (for VWAP).
    - "QQQ": NQ futures in regular hours to 2025-03-27, then QQQ itself. The
      level jump on the day the source switches is not a return:
      `intraday_lab.Grid` drops overnight moves over 20%.
    - anything else: that symbol's bars cut to regular hours.
    """
    if symbol == "SPY":
        spx = _rth(_new_york(load_minute("SPX", start)))
        mes = _new_york(load_minute("MES", start))
        spx["volume"] = _rth(mes)["volume"].reindex(spx.index).fillna(0.0)
        day = spx.index.normalize()
        first = ~day.duplicated()
        mes_close = _rth(mes)["close"].groupby(_rth(mes).index.normalize()).last()
        mes_open = _rth(mes)["open"].groupby(_rth(mes).index.normalize()).first()
        overnight = (mes_open / mes_close.shift(1)).reindex(day[first])
        spx_close = spx["close"].groupby(day).last().shift(1).reindex(day[first])
        implied = (spx_close * overnight).to_numpy()
        opens = spx["open"].to_numpy().copy()
        ok = np.isfinite(implied)
        idx = np.flatnonzero(first)
        opens[idx[ok]] = implied[ok]
        spx["open"] = opens
        spx["high"] = np.maximum(spx["high"], spx["open"])
        spx["low"] = np.minimum(spx["low"], spx["open"])
        return spx
    if symbol == "QQQ":
        nq = _rth(_new_york(load_minute("NQ", start)))
        qqq = _rth(_new_york(load_minute("QQQ", start)))
        switch = qqq.index.normalize().min()
        nq = nq[nq.index < switch]
        out = pd.concat([nq, qqq])
        return out[~out.index.duplicated()]
    return _rth(_new_york(load_minute(symbol, start)))
