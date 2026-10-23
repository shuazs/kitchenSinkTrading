"""US Congress stock trades disclosed under the STOCK Act.

Source: Hugging Face `ZipLime/congress-trading` (`data/trades`, House and
Senate splits), shared copy in /mnt/project-files/congress-data/. Periodic
Transaction Reports (PTRs) only: annual Schedule B rows are public a year
late. A trade is public on its `filing_date` (median 28 days after the trade,
90th percentile 54); never key on `transaction_date` or `notification_date`.
The House is complete from 2014; the Senate is thin before 2023.

Amounts are disclosed as ranges ($1,001-15,000, $15,001-50,000, ...);
`amount_low` is the bottom of the range.
"""
from __future__ import annotations

import os
import subprocess

import pandas as pd

from .cache import cache_dir

HF = "https://huggingface.co/datasets/ZipLime/congress-trading/resolve/main/data/trades/"
SHARED = "/mnt/project-files/congress-data/"
FILES = ["trades-house.parquet", "trades-senate.parquet"]

# Party leaders and whips (Speaker, floor leaders, whips) in 2014-2026 who
# filed PTRs. Nancy Pelosi (Speaker 2019-23) is the only one with many trades.
LEADERS = {"P000197": "Nancy Pelosi", "B000589": "John Boehner", "J000294": "Hakeem Jeffries",
           "T000250": "John Thune", "C001056": "John Cornyn", "B001261": "John Barrasso",
           "E000294": "Tom Emmer", "R000570": "Paul Ryan"}


def _path(name: str) -> str:
    if os.path.exists(SHARED + name):
        return SHARED + name
    path = cache_dir("congress") / name
    if not path.exists():
        subprocess.run(["curl", "-sL", "-o", str(path), HF + name], check=True)
    return str(path)


def load_trades(stocks_only: bool = True) -> pd.DataFrame:
    """One row per PTR transaction, latest amendment only.

    Columns: member, bioguide_id, chamber, party, owner, ticker (Yahoo
    format, BRK.B -> BRK-B), side (+1 purchase, -1 sale), amount_low,
    trade_date, filing_date, lag_days. `stocks_only` drops options, rows
    without a ticker and non-stock asset categories.
    """
    raw = pd.concat([pd.read_parquet(_path(f)) for f in FILES], ignore_index=True)
    d = raw[raw["superseded_by"].isna() & (raw["source_form"] == "ptr")]
    d = d[d["transaction_type"].isin(["purchase", "sale_full", "sale_partial"])]
    if stocks_only:
        d = d[d["ticker"].notna() & ~d["is_option"].astype(bool) & d["asset_category"].isin(["stock", "other"])]
    out = pd.DataFrame({
        "member": d["member_name"], "bioguide_id": d["bioguide_id"], "chamber": d["chamber"],
        "party": d["party"], "owner": d["owner"],
        "ticker": d["ticker"].str.replace(".", "-", regex=False),
        "side": d["transaction_type"].map({"purchase": 1, "sale_full": -1, "sale_partial": -1}),
        "amount_low": d["min_amount_usd"],
        "trade_date": pd.to_datetime(d["transaction_date"], errors="coerce"),
        "filing_date": pd.to_datetime(d["filing_date"], errors="coerce"),
    })
    out["lag_days"] = (out["filing_date"] - out["trade_date"]).dt.days
    # misread years and impossible dates
    out = out[out["filing_date"].notna() & out["lag_days"].between(0, 400)]
    return out.sort_values("filing_date").reset_index(drop=True)


def buy_events(trades: pd.DataFrame, delay_days: int = 1) -> pd.DataFrame:
    """Purchases as events for `EventHold`: one row per (ticker, public day).

    `date` is the filing date plus `delay_days` calendar days (filings are
    posted during the day and sometimes overnight, so acting the day after
    is the safe reading). Adds `n_members` (distinct buyers that day),
    `amount_low` (summed) and `members_60d` (distinct members who disclosed a
    purchase of the ticker in the trailing 60 days, this filing included).
    """
    b = trades[trades["side"] == 1].copy()
    b["date"] = b["filing_date"] + pd.Timedelta(days=delay_days)
    ev = b.groupby(["ticker", "date"]).agg(n_members=("bioguide_id", "nunique"), amount_low=("amount_low", "sum"),
                                           party=("party", "first"), lag_days=("lag_days", "median"),
                                           leader=("bioguide_id", lambda s: s.isin(LEADERS).any())).reset_index()
    ev["members_60d"] = 0
    for t, g in b.groupby("ticker"):
        dates = g["date"].to_numpy()
        ids = g["bioguide_id"].to_numpy()
        mask = ev["ticker"] == t
        ev.loc[mask, "members_60d"] = [len(set(ids[(dates <= d) & (dates > d - pd.Timedelta(days=60))]))
                                      for d in ev.loc[mask, "date"].to_numpy()]
    return ev.rename(columns={"ticker": "symbol"}).sort_values("date").reset_index(drop=True)
