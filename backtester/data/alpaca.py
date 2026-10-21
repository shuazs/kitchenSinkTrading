"""Minute bars for US stocks and ETFs from Alpaca's market data API.

yfinance keeps only about 60 days of 5-minute bars, too little to test an
intraday strategy. Alpaca's free plan serves minute bars back to 2016:

1. Sign up at https://alpaca.markets (a paper-trading account is enough, no
   deposit needed).
2. In the dashboard, generate an API key for paper trading.
3. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment.

The free plan can query the consolidated SIP feed for anything older than
15 minutes, which covers backtests. Bars are cached per symbol and month.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from .cache import cache_dir

URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
SESSION = ("09:30", "15:59")


def _headers() -> dict[str, str]:
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (free account at https://alpaca.markets); "
            "see backtester/data/alpaca.py"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str, feed: str) -> pd.DataFrame:
    params = {
        "timeframe": timeframe,
        "start": start.tz_localize("America/New_York").isoformat(),
        "end": end.tz_localize("America/New_York").isoformat(),
        "adjustment": "all",
        "feed": feed,
        "limit": 10_000,
    }
    rows = []
    while True:
        r = requests.get(URL.format(symbol=symbol), headers=_headers(), params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        rows += body.get("bars") or []
        if not body.get("next_page_token"):
            break
        params["page_token"] = body["next_page_token"]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.to_datetime(df.pop("time"), utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def load_minute_bars(symbol: str, start="2016-01-01", end=None, timeframe: str = "1Min", feed: str = "sip",
                     refresh: bool = False) -> pd.DataFrame:
    """Regular-session bars (09:30-16:00 New York time), indexed by bar START
    time, split- and dividend-adjusted. Cached one csv per month."""
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    out = []
    for month in pd.period_range(pd.Timestamp(start), end, freq="M"):
        path = cache_dir(f"alpaca/{symbol}_{timeframe}") / f"{month}.csv"
        current = month == pd.Timestamp.today().to_period("M")
        if path.exists() and not refresh and not current:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = _fetch(symbol, month.start_time, month.end_time.floor("s"), timeframe, feed)
            if not current:
                df.to_csv(path)
        out.append(df)
    df = pd.concat(out).sort_index()
    df = df[~df.index.duplicated()].between_time(*SESSION)
    return df.loc[pd.Timestamp(start):end + pd.Timedelta(days=1)]


def load_yahoo_intraday(symbol: str, interval: str = "5m") -> pd.DataFrame:
    """The last ~60 days of regular-session bars from yfinance, for a quick
    smoke test without an Alpaca key. Indexed by bar start, New York time."""
    import yfinance as yf

    df = yf.download(symbol, period="60d", interval=interval, auto_adjust=True, progress=False, prepost=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].between_time(*SESSION)
