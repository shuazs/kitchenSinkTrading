"""Macro series from FRED's public CSV endpoint (no API key needed)."""
from __future__ import annotations

import io

import pandas as pd
import requests

from .cache import cache_dir

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# Handy series for regime filters and cash yield.
SERIES = {
    "DTB3": "3-month T-bill rate, %",
    "DGS10": "10-year Treasury yield, %",
    "T10Y2Y": "10y minus 2y Treasury spread, %",
    "T10Y3M": "10y minus 3m Treasury spread, %",
    "BAMLH0A0HYM2": "High-yield OAS, %",
    "UNRATE": "Unemployment rate, %",
    "VIXCLS": "VIX close",
}


def load_series(series_id: str, refresh: bool = False) -> pd.Series:
    path = cache_dir("fred") / f"{series_id}.csv"
    if path.exists() and not refresh:
        text = path.read_text()
    else:
        r = requests.get(URL.format(series_id), timeout=30)
        r.raise_for_status()
        text = r.text
        if text.lstrip().startswith("<"):
            # FRED serves an HTML maintenance page with status 200 at times.
            raise RuntimeError(f"FRED returned a web page instead of {series_id} data")
        path.write_text(text)
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")[series_id], errors="coerce").dropna()


def tbill_rate() -> pd.Series:
    """Annual 3-month T-bill rate as a decimal, for Backtester(cash_rate=...).
    Falls back to Yahoo's 13-week T-bill yield (^IRX) when FRED is down."""
    try:
        return load_series("DTB3") / 100
    except (RuntimeError, requests.RequestException):
        from .yahoo import load_bars

        return load_bars("^IRX")["close"].dropna() / 100
