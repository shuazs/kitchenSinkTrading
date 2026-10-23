"""FOMC meeting calendar from Hugging Face `ZipLime/fomc-events` (2007+).

Scheduled meetings are published a year ahead, so using their dates is not
look-ahead.
"""
import pandas as pd

from .cache import cache_dir

URL = "https://huggingface.co/datasets/ZipLime/fomc-events/resolve/main/data/meetings/part-00000.parquet"


def meetings(refresh: bool = False) -> pd.DataFrame:
    path = cache_dir("fomc") / "meetings.parquet"
    if refresh or not path.exists():
        import requests

        r = requests.get(URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pd.read_parquet(path)


def statement_dates(refresh: bool = False) -> pd.DatetimeIndex:
    """Statement days of scheduled meetings."""
    m = meetings(refresh)
    m = m[m["scheduled"] & m["has_statement"]]
    return pd.DatetimeIndex(pd.to_datetime(m["meeting_end_date"])).sort_values()
