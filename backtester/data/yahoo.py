"""Market data and headlines from Yahoo Finance via yfinance.

Limits worth knowing: yfinance only covers listings that exist today
(survivorship bias), intraday history is short (1m bars for ~7 days, 1h for
~730 days), and it serves current option chains only, not history.
"""
from __future__ import annotations

import pandas as pd

from ..core import MarketData
from .cache import cache_dir


def load_bars(symbol: str, start="2000-01-01", end=None, interval="1d", refresh=False) -> pd.DataFrame:
    """Split- and dividend-adjusted OHLCV with lower-case columns, cached as parquet or csv."""
    import yfinance as yf

    path = cache_dir("yahoo") / f"{symbol.replace('/', '_')}_{interval}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(symbol, start="1990-01-01", interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(path)
    return df.loc[start:end, ["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def load_hourly(symbol: str, refresh=False) -> pd.DataFrame:
    """The last ~730 days of 1h bars, indexed by the bar's start in UTC (tz-naive).

    Futures (CL=F, ES=F) trade nearly 24 hours; ETF bars cover market hours only.
    """
    import yfinance as yf

    path = cache_dir("yahoo") / f"{symbol.replace('/', '_')}_1h_utc.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = yf.download(symbol, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_convert("UTC").tz_localize(None).floor("h")
    df = df[~df.index.duplicated(keep="last")][["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.to_csv(path)
    return df


def load_market(symbols, start="2000-01-01", end=None, interval="1d", **extra) -> MarketData:
    return MarketData({s: load_bars(s, start, end, interval) for s in symbols}, extra)


def option_chain(symbol: str, expiry: str | None = None) -> dict[str, pd.DataFrame]:
    """Today's option chain (calls and puts). There is no free option history."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    exp = expiry or t.options[0]
    chain = t.option_chain(exp)
    return {"expiry": exp, "calls": chain.calls, "puts": chain.puts}


def headlines(symbol: str) -> pd.DataFrame:
    """Recent headlines for a ticker. Yahoo returns only the latest few dozen,
    so collect these on a schedule to build a history for backtesting."""
    import yfinance as yf

    # Ticker.news has come back empty in recent yfinance releases; Search still works.
    items = yf.Ticker(symbol).news or yf.Search(symbol, news_count=50).news or []
    rows = []
    for item in items:
        c = item.get("content", item)
        rows.append(
            {
                "symbol": symbol,
                "published": pd.to_datetime(c.get("pubDate") or c.get("providerPublishTime"), utc=True, unit=None if c.get("pubDate") else "s"),
                "title": c.get("title"),
                "summary": c.get("summary", ""),
                "publisher": (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher"),
            }
        )
    return pd.DataFrame(rows)


def append_headlines(symbols, path=None) -> pd.DataFrame:
    """Fetch headlines for symbols and append new ones to a local archive."""
    path = path or cache_dir("news") / "headlines.csv"
    new = pd.concat([headlines(s) for s in symbols], ignore_index=True) if symbols else pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path, parse_dates=["published"])
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["symbol", "title"]).sort_values("published")
    new.to_csv(path, index=False)
    return new
