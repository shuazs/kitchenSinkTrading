"""Shared helpers for the paper loggers. Importing this module loads
paper_loggers/keys.env (copy keys.example.env and fill it in) into the
environment; variables already set in the shell win.

Logs go to ~/.cache/kitchensink/paper/<logger>/ (KS_CACHE_DIR moves the root).
Nothing here places orders.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys.env"


def load_keys(path: Path = KEYS) -> dict[str, str]:
    """KEY=value lines (# comments, optional quotes); blank values are skipped."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            out[k.strip()] = v
            os.environ.setdefault(k.strip(), v)
    return out


load_keys()

from backtester.data.cache import cache_dir  # noqa: E402  (after keys.env may set KS_CACHE_DIR)


def need(name: str) -> str:
    """An env var a logger cannot run without; exits with a pointer to keys.env."""
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set: copy paper_loggers/keys.example.env to paper_loggers/keys.env and fill it in")
    return v


def log_dir(name: str) -> Path:
    return cache_dir(f"paper/{name}")


def read_log(path: Path, fields: list[str], dates: tuple[str, ...] = ()) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype=object).reindex(columns=fields)
    else:
        df = pd.DataFrame(columns=fields, dtype=object)
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def write_log(df: pd.DataFrame, path: Path) -> None:
    """CSV with dates as YYYY-MM-DD (also in object columns holding Timestamps)."""
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda v: f"{v:%Y-%m-%d}" if isinstance(v, pd.Timestamp) and not pd.isna(v) and v == v.normalize()
                            else v)
    out.to_csv(path, index=False)


def tstat(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")


class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("July4th", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_DAY = CustomBusinessDay(calendar=NYSECalendar())


def trading_days(start, end) -> pd.DatetimeIndex:
    """NYSE full trading days (early closes count as days; a Saturday New Year is not moved back)."""
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq=NYSE_DAY)


def today() -> pd.Timestamp:
    """Today's date in New York."""
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def closes(tickers, start, field: str = "Close") -> pd.DataFrame:
    """Daily adjusted closes (or another field) from Yahoo, one column per ticker."""
    import yfinance as yf

    d = yf.download(list(tickers), start=str(pd.Timestamp(start).date()), auto_adjust=True, progress=False)
    out = d[field]
    if isinstance(out, pd.Series):
        out = out.to_frame(list(tickers)[0])
    return out.dropna(how="all")


def sec_get(url: str, params: dict | None = None, timeout: float = 60.0):
    """GET an SEC URL with the User-Agent EDGAR requires, at <= ~8 requests/s."""
    import time

    import requests

    r = requests.get(url, params=params, headers={"User-Agent": need("SEC_USER_AGENT")}, timeout=timeout)
    time.sleep(0.12)
    r.raise_for_status()
    return r
