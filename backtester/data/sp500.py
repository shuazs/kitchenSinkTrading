"""S&P 500 membership history and the dated changes it implies.

Source: https://github.com/fja05680/sp500, a community-maintained daily
membership list from 1996 (Andreas Clenow's data, extended from Wikipedia
and press reports). Rows are dated on the day a change took effect, and
tickers are as they were at the time, so renamed or delisted stocks will not
match today's Yahoo tickers. Announcement dates are not in the data; S&P
usually announces 2 to 5 trading days before the effective date.
"""
from __future__ import annotations

import pandas as pd
import requests

from .cache import cache_dir

URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"


def membership(refresh: bool = False) -> pd.Series:
    """Date -> frozenset of member tickers on and after that date."""
    path = cache_dir("sp500") / "components.csv"
    if not path.exists() or refresh:
        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["tickers"].map(lambda s: frozenset(s.split(",")))


def changes(since: str = "2000-01-01") -> pd.DataFrame:
    """One row per ticker added to or removed from the index: date, symbol,
    change ("add" or "remove"). Yahoo spells class shares with a dash
    (BRK-B), so dots are converted."""
    m = membership()
    rows = []
    for prev, (d, cur) in zip(m.iloc[:-1], m.iloc[1:].items()):
        rows += [(d, s, "add") for s in cur - prev]
        rows += [(d, s, "remove") for s in prev - cur]
    df = pd.DataFrame(rows, columns=["date", "symbol", "change"])
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    return df[df["date"] >= since].sort_values(["date", "symbol"]).reset_index(drop=True)
