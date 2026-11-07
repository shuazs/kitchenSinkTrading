"""Annual advertising expense for every SEC filer (XBRL `AdvertisingExpense`)
and the ad-intensity / ad-growth signals built from it.

Frames carry no filing date, so a value counts as public `LAG` days after the
fiscal year ends (10-K deadlines are 60-90 days; a restated number would be a
mild look-ahead, as in `fundamentals.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import ANNUAL, frame
from .edgar import ticker_to_cik

LAG = 100


def panel(start_year: int = 2009, end_year: int = 2025) -> pd.DataFrame:
    """cik, end (fiscal year end), ad, rev, available."""
    ad, rev = [], []
    for y in range(start_year, end_year + 1):
        f = frame("AdvertisingExpense", f"CY{y}")
        if not f.empty:
            ad.append(f[["cik", "end", "val"]].rename(columns={"val": "ad"}))
        for rank, tag in enumerate(ANNUAL["revenue"]):
            r = frame(tag, f"CY{y}")
            if not r.empty:
                rev.append(r.assign(rank=rank)[["cik", "end", "val", "rank"]])
    ad = pd.concat(ad)
    rev = pd.concat(rev).sort_values("rank").drop_duplicates(["cik", "end"]).drop(columns="rank").rename(columns={"val": "rev"})
    out = ad.merge(rev, on=["cik", "end"], how="left")
    out["end"] = pd.to_datetime(out["end"])
    out["available"] = out["end"] + pd.Timedelta(days=LAG)
    return out.sort_values(["cik", "end"]).drop_duplicates(["cik", "end"], keep="last").reset_index(drop=True)


def signals(p: pd.DataFrame, min_ad: float = 2.5e7) -> pd.DataFrame:
    """Year-over-year signals per (cik, fiscal year): ad growth, ad intensity
    (ad/revenue), change in intensity and growth in ad minus growth in sales."""
    p = p.sort_values(["cik", "end"]).copy()
    g = p.groupby("cik")
    prev_end = g["end"].shift(1)
    ok = (p["end"] - prev_end).dt.days.between(330, 400)  # consecutive fiscal years
    for c in ("ad", "rev"):
        p[c + "_prev"] = g[c].shift(1).where(ok)
    p["ad_g"] = np.log(p["ad"] / p["ad_prev"])
    p["rev_g"] = np.log(p["rev"] / p["rev_prev"])
    p["intensity"] = p["ad"] / p["rev"]
    p["d_intensity"] = p["intensity"] - p["ad_prev"] / p["rev_prev"]
    p["excess_g"] = p["ad_g"] - p["rev_g"]
    p = p[(p["ad"] >= min_ad) & p["ad_g"].notna() & (p["ad_prev"] > 0)]
    return p.replace([np.inf, -np.inf], np.nan)


def cik_tickers() -> dict[int, str]:
    """Current ticker for each CIK (first listed)."""
    out: dict[int, str] = {}
    for t, c in ticker_to_cik().items():
        out.setdefault(int(c), t)
    return out
