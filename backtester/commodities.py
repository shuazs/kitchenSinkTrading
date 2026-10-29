"""Commodity futures research helpers (examples/run_commodity_futures.py).

No expired-contract history is reachable from the cloud (Yahoo serves live
contracts only, CME/EIA/USDA/Stooq/Nasdaq are blocked), so tradeable returns
come from commodity ETFs and Yahoo's front-month `=F` series (unadjusted: roll
gaps on expiry days) is used only for signals and event-day returns away from
rolls. Carry is proxied by the roll yield an ETF realised over the lookback.

The one lead, `storage_report_days` / `storage_day_returns`: natural gas falls
on the day of the EIA weekly storage report (Thursday 10:30 ET since May
2002; AGA reported on Wednesdays before that).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# single-commodity ETF -> Yahoo front-month root
ETF_ROOT = {"USO": "CL", "BNO": "BZ", "UNG": "NG", "UGA": "RB", "CORN": "ZC", "SOYB": "ZS", "WEAT": "ZW",
            "CANE": "SB", "GLD": "GC", "SLV": "SI", "CPER": "HG", "PPLT": "PL", "PALL": "PA"}
EIA_START = "2002-05-01"  # first EIA weekly storage report (Thursdays); AGA (Wednesdays) before


def realized_carry(etf_close: pd.DataFrame, front_close: pd.DataFrame, tbill: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """Roll yield realised over `lookback` days: ETF log return minus the
    front-month price change minus T-bill income (annual rate, decimal)."""
    tb = tbill.reindex(etf_close.index).ffill().rolling(lookback).sum() / 252
    return np.log(etf_close / etf_close.shift(lookback)) - np.log(front_close / front_close.shift(lookback)) - tb.values[:, None]


def expiry_mask(index: pd.DatetimeIndex, months, first_day: int = 12, last_day: int = 17) -> pd.Series:
    """True on days in a contract month's expiry window (grains roll ~14th),
    whose front-month returns carry roll gaps."""
    return pd.Series(index.month.isin(list(months)) & (index.day >= first_day) & (index.day <= last_day), index=index)


def storage_report_days(index: pd.DatetimeIndex, start: str = EIA_START) -> pd.DatetimeIndex:
    """EIA storage report days among `index`: each week's Thursday, or its
    Wednesday when Thursday is a holiday (Thanksgiving, July 4, Christmas)."""
    idx = pd.DatetimeIndex(index)
    out = []
    for _, days in pd.Series(idx, index=idx).groupby(idx.to_period("W-SUN")):
        dow = days.index.dayofweek
        pick = days.index[dow == 3] if (dow == 3).any() else days.index[dow == 2]
        if len(pick):
            out.append(pick[0])
    out = pd.DatetimeIndex(out)
    return out[out >= pd.Timestamp(start)]


def storage_day_returns(close: pd.Series, days, side: int = -1, cost: float = 6e-4) -> pd.Series:
    """Net log return per report day of holding `side` (-1 short) from the
    previous close to the report-day close (both legs market-on-close, dates
    known in advance); `cost` is the round trip."""
    r = np.log(close).diff()
    return (side * r.reindex(pd.DatetimeIndex(days)) - cost).dropna()


def season_returns(close: pd.Series, start_md: str, end_md: str, mask: pd.Series | None = None) -> pd.Series:
    """Sum of daily log returns per year between month-day strings ("07-01",
    "09-30"); days where `mask` is True (roll gaps) count as zero."""
    r = np.log(close).diff()
    if mask is not None:
        r = r.where(~mask.reindex(r.index).fillna(False))
    out = {}
    for y in sorted(set(r.index.year)):
        seg = r.loc[f"{y}-{start_md}":f"{y}-{end_md}"]
        if seg.count() > 20:
            out[y] = seg.sum()
    return pd.Series(out, dtype=float)


def ratio_positions(log_ratio: pd.Series, lookback: int = 60, z_in: float = 2.0) -> pd.Series:
    """+1 / -1 / 0 position in the ratio: fade |z| > z_in, exit when z crosses 0."""
    z = (log_ratio - log_ratio.rolling(lookback).mean()) / log_ratio.rolling(lookback).std()
    pos, cur = [], 0
    for v in z.values:
        if np.isnan(v):
            cur = 0
        elif cur == 0 and abs(v) > z_in:
            cur = -int(np.sign(v))
        elif cur * v > 0:  # crossed zero
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=z.index, dtype=float)


def wasde_revisions(wasde: pd.DataFrame, commodities=("Corn", "Oilseed, Soybean", "Wheat")) -> pd.DataFrame:
    """Month-on-month revision of the US ending-stocks projection for the
    latest marketing year in each WASDE (`data/wasde.py` table)."""
    w = wasde[(wasde.Region == "United States") & (wasde.Attribute.str.lower() == "ending stocks")
              & wasde.ProjEstFlag.astype(str).str.startswith("Proj")
              & wasde.ReportTitle.str.contains("Supply and Use", case=False)
              & ~wasde.ReportTitle.str.contains("World|Reliab", case=False)]
    rows = []
    for c in commodities:
        piv = (w[w.Commodity == c].groupby(["release", "MarketYear"]).Value.first()
               .unstack().sort_index())
        for i in range(1, len(piv)):
            my = piv.iloc[i].dropna().index.max()
            prev = piv.iloc[i - 1].get(my, np.nan)
            rows.append({"date": piv.index[i], "commodity": c, "market_year": my, "ending_stocks": piv.iloc[i][my],
                         "revision": piv.iloc[i][my] / prev - 1 if prev == prev and prev else np.nan})
    return pd.DataFrame(rows)
