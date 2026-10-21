"""Currency total-return series for carry strategies.

Holding a currency future earns the spot move plus the interest-rate gap
between the two currencies. `load_fx_market` rebuilds that from free data:
Yahoo spot rates and the OECD 3-month interbank rates FRED republishes
(monthly averages). Each series is the USD value of a deposit in that
currency: spot return plus the foreign rate. Bought with cash in the
engine, it forgoes the USD cash rate, so its net return is the futures
return; shorted, the sale proceeds earn the USD rate, which is the short
side of a forward.

Yahoo's spot history has one-day bad ticks; those are removed. Euro-area and UK rates on FRED stopped updating in early 2026,
so their last value is carried forward.
"""
from __future__ import annotations

import pandas as pd

from ..core import MarketData
from .fred import load_series
from .yahoo import load_bars

# Currency -> (Yahoo symbol, True if quoted as USD per unit, FRED rate series)
CURRENCIES = {
    "AUD": ("AUDUSD=X", True, "IR3TIB01AUM156N"),
    "NZD": ("NZDUSD=X", True, "IR3TIB01NZM156N"),
    "GBP": ("GBPUSD=X", True, "IR3TIB01GBM156N"),
    "EUR": ("EURUSD=X", True, "IR3TIB01EZM156N"),
    "JPY": ("JPY=X", False, "IR3TIB01JPM156N"),
    "CAD": ("CAD=X", False, "IR3TIB01CAM156N"),
    "CHF": ("CHF=X", False, "IR3TIB01CHM156N"),
    "MXN": ("MXN=X", False, "IR3TIB01MXM156N"),
}
USD_RATE = "IR3TIB01USM156N"

# CME micro FX contract sizes, in units of the foreign currency. MXN has no
# micro; its smallest is the 500,000-peso standard contract.
CONTRACT_UNITS = {
    "AUD": 10_000, "NZD": 10_000, "GBP": 6_250, "EUR": 12_500,
    "JPY": 1_250_000, "CAD": 10_000, "CHF": 12_500, "MXN": 500_000,
}


def remove_spikes(close: pd.Series, jump: float = 0.04) -> pd.Series:
    """Drop one-day prints more than `jump` away from both neighbours while the
    neighbours themselves agree, the shape of a bad tick rather than a real move."""
    prev, nxt = close.ffill().shift(1), close.bfill().shift(-1)
    bad = ((close / prev - 1).abs() > jump) & ((close / nxt - 1).abs() > jump) & ((nxt / prev - 1).abs() < jump)
    return close.mask(bad).ffill()


def usd_per_unit(ccy: str, start="2000-01-01", end=None) -> pd.Series:
    symbol, direct, _ = CURRENCIES[ccy]
    close = remove_spikes(load_bars(symbol, start, end)["close"])
    return close if direct else 1.0 / close


def rate(series_id: str) -> pd.Series:
    """Monthly rate as a decimal, stamped on the first day of the month it averages."""
    return load_series(series_id) / 100


def daily_rate(series_id: str, index: pd.DatetimeIndex) -> pd.Series:
    r = rate(series_id)
    return r.reindex(index.union(r.index)).ffill().reindex(index)


def total_return_index(spot: pd.Series, foreign_rate: pd.Series) -> pd.Series:
    """Spot move plus interest accrued at `foreign_rate` (annual, decimal) per calendar day."""
    days = spot.index.to_series().diff().dt.days.fillna(0)
    growth = spot.pct_change(fill_method=None).fillna(0) + 1
    accrual = (1 + foreign_rate.fillna(0)) ** (days / 365.0)
    return (growth * accrual).cumprod() * 100


def known_rate_gap(index: pd.DatetimeIndex, ccys) -> pd.DataFrame:
    """Foreign minus USD rate, as it was known on each day.

    A monthly average is only complete at the end of its month, so a month's
    value is first used on the first day of the following month.
    """
    usd = rate(USD_RATE)
    out = {}
    for c in ccys:
        gap = (rate(CURRENCIES[c][2]) - usd).dropna()
        gap.index = gap.index + pd.offsets.MonthBegin(1)
        out[c] = gap.reindex(index.union(gap.index)).ffill().reindex(index)
    return pd.DataFrame(out)


def load_fx_market(ccys=None, start="2004-01-01", end=None) -> MarketData:
    """Total-return bars per currency plus `extra['rate_gap']` (foreign - USD rate, as known)."""
    ccys = list(ccys or CURRENCIES)
    bars, spots = {}, {}
    for c in ccys:
        spot = usd_per_unit(c, start, end)
        tr = total_return_index(spot, daily_rate(CURRENCIES[c][2], spot.index))
        bars[c] = pd.DataFrame({"open": tr, "high": tr, "low": tr, "close": tr, "volume": 0.0})
        spots[c] = spot
    data = MarketData(bars)
    data.extra["rate_gap"] = known_rate_gap(data.index, ccys)
    data.extra["spot"] = pd.DataFrame(spots).reindex(data.index).ffill()
    return data
