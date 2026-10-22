"""CFTC Commitments of Traders (COT) positions, timed to publication.

Source: Hugging Face `ZipLime/commitments-of-traders` (`data/pit`), one row
per futures market per weekly report, 2010 onward. Positions are taken on
Tuesday and published Friday 3:30 pm ET (later after holidays and shutdowns,
e.g. Oct-Dec 2025). The mirror records the real publication instant in
`knowledge_date`, so `weekly_panel` dates every report by when it could be
known, never by the Tuesday it describes. A shared copy is in
/mnt/project-files/cot-data/.

Net positions are long minus short. Legacy categories exist for every
market: `net_noncommercial` (speculators: funds, CTAs), `net_commercial`
(hedgers: producers, merchants, swap dealers), `net_nonreportable` (small
traders).
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd

from .cache import cache_dir

HF = "https://huggingface.co/datasets/ZipLime/commitments-of-traders/resolve/main/data/pit/part-00000.parquet"
SHARED = "/mnt/project-files/cot-data/pit.parquet"

# CFTC contract code -> ETF that tracks the same future (or the same currency
# vs USD). All have COT data from 2010 and an ETF from 2009-2011.
ETF_MAP = {
    "088691": "GLD", "084691": "SLV", "076651": "PPLT", "085692": "CPER",
    "067651": "USO", "023651": "UNG",
    "002602": "CORN", "001602": "WEAT", "005602": "SOYB", "080732": "CANE",
    "099741": "FXE", "097741": "FXY", "096742": "FXB", "232741": "FXA", "090741": "FXC", "092741": "FXF",
    "020601": "TLT", "043602": "IEF",
    "13874A": "SPY", "209742": "QQQ",
}
COMMODITY_ETFS = ["GLD", "SLV", "PPLT", "CPER", "USO", "UNG", "CORN", "WEAT", "SOYB", "CANE"]
FINANCIAL_ETFS = [s for s in ETF_MAP.values() if s not in COMMODITY_ETFS]
FIELDS = ["open_interest", "net_noncommercial", "net_commercial", "net_nonreportable"]


def load_pit(refresh: bool = False) -> pd.DataFrame:
    """The raw point-in-time table (all ~760 markets)."""
    if os.path.exists(SHARED) and not refresh:
        return pd.read_parquet(SHARED)
    path = cache_dir("cot") / "pit.parquet"
    if not path.exists() or refresh:
        subprocess.run(["curl", "-sL", "-o", str(path), HF], check=True)
    return pd.read_parquet(path)


def weekly_panel(pit: pd.DataFrame, symbols: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """{field: report x symbol frame} for FIELDS, plus `known`: the ET date
    each report was published (a Series indexed like the frames).

    Index is the report's Tuesday. Use `known` (not the index) to place a
    signal on the price calendar.
    """
    m = symbols or ETF_MAP
    p = pit[pit["entity_id"].isin(m)].copy()
    p["sym"] = p["entity_id"].map(m)
    p["event_date"] = pd.to_datetime(p["event_date"])
    out = {f: p.pivot_table(index="event_date", columns="sym", values=f) for f in FIELDS}
    known = p["knowledge_date"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    out["known"] = known.groupby(p["event_date"]).max()
    return out


def positioning_signals(panel: dict[str, pd.DataFrame], vol_window: int = 52, rank_window: int = 156) -> dict[str, pd.DataFrame]:
    """Report x symbol signals; higher = more bullish for the asset.

    - `flow{k}` (k = 1, 4 weeks): minus the change in speculators' net
      position as a share of open interest, scaled by its own trailing std.
      Kang, Rouwenhorst and Tang (2020): assets speculators just bought
      underperform the ones they sold (speculators pay for liquidity).
    - `hedging_pressure`: minus the percentile of hedgers' net/OI within its
      own trailing `rank_window` reports. Basu and Miffre (2013): long where
      hedgers are unusually short, short where they are unusually long.
    - `commercial_index`: the opposite sign, the practitioner "COT index"
      (follow commercials at extremes, i.e. fade crowded speculators).
    - `combo`: average cross-sectional percentile of `commercial_index` and
      `flow4`.
    """
    oi = panel["open_interest"]
    spec = panel["net_noncommercial"] / oi
    hedge = panel["net_commercial"] / oi
    out = {}
    for k in (1, 4):
        d = spec.diff(k)
        out[f"flow{k}"] = -d / d.rolling(vol_window, min_periods=vol_window // 2).std()
    rank = hedge.rolling(rank_window, min_periods=rank_window // 3).rank(pct=True)
    out["hedging_pressure"] = -rank
    out["commercial_index"] = rank
    out["combo"] = (rank.rank(axis=1, pct=True) + out["flow4"].rank(axis=1, pct=True)) / 2
    return out


def to_trading_days(signal: pd.DataFrame, known: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Re-index a report-dated signal onto the price calendar.

    A report published on day D (3:30 pm ET, before the 4 pm close) lands on
    D, or the next trading day if D is not one; the engine then fills at the
    following open (Monday for a normal Friday release).
    """
    pos = np.searchsorted(days, known.reindex(signal.index).values, side="left")
    ok = pos < len(days)
    s = signal[ok].copy()
    report = pd.Series(s.index, index=days[pos[ok]])
    s.index = report.index
    order = np.lexsort((report.values, report.index.values))
    s, report = s.iloc[order], report.iloc[order]
    # a backlogged report published after a newer one adds nothing new
    newest = report.values >= report.cummax().values
    s = s[newest]
    return s[~s.index.duplicated(keep="last")]
