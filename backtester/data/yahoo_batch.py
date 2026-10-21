"""Download daily bars for many tickers at once into the `load_bars` cache.

`yahoo.load_bars` fetches one symbol per call, which is slow for the hundreds
of small caps an event strategy touches. `prefetch` downloads in chunks with
yfinance's threaded multi-ticker mode and writes each symbol to the same
cache file `load_bars` reads, so later `load_bars`/`load_market` calls hit
the cache. Symbols Yahoo has no data for (usually delisted, which is the
survivorship bias of this source) are remembered and not retried.
"""
from __future__ import annotations

import pandas as pd

from .cache import cache_dir


def _path(symbol: str, interval: str):
    return cache_dir("yahoo") / f"{symbol.replace('/', '_')}_{interval}.csv"


def prefetch(symbols, interval: str = "1d", chunk: int = 200, verbose: bool = False) -> list[str]:
    """Make sure every symbol is cached. Returns the symbols that have data."""
    import yfinance as yf

    missing_path = cache_dir("yahoo") / f"_missing_{interval}.txt"
    known_missing = set(missing_path.read_text().split()) if missing_path.exists() else set()
    symbols = list(dict.fromkeys(symbols))
    todo = [s for s in symbols if not _path(s, interval).exists() and s not in known_missing]
    for k in range(0, len(todo), chunk):
        batch = todo[k : k + chunk]
        if verbose:
            print(f"yahoo: downloading {k + len(batch)}/{len(todo)}")
        raw = yf.download(batch, start="1990-01-01", interval=interval, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        for s in batch:
            try:
                df = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            except KeyError:
                df = pd.DataFrame()
            df = df.dropna(how="all")
            if df.empty or "Close" not in df or df["Close"].notna().sum() < 5:
                known_missing.add(s)
                continue
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            df.to_csv(_path(s, interval))
        missing_path.write_text("\n".join(sorted(known_missing)))
    return [s for s in symbols if _path(s, interval).exists()]
