"""US Treasury auction dates from Hugging Face `Farmaanaa/us_treasury_auctions`
(TreasuryDirect results, 2008+; treasurydirect.gov and fiscaldata are blocked
from the cloud). Auction dates are announced ahead of time, so using them is
not look-ahead.
"""
import json

import pandas as pd

from .cache import cache_dir

URL = "https://huggingface.co/datasets/Farmaanaa/us_treasury_auctions/resolve/main/observations.parquet"
LONG_TERMS = r"(10-Year|9-Year|30-Year|29-Year)"  # new issues and reopenings


def load_auctions(refresh: bool = False) -> pd.DataFrame:
    """One row per auction: date, type (Bill/Note/Bond/...), term, bid_to_cover."""
    path = cache_dir("treasury") / "auctions.parquet"
    if refresh or not path.exists():
        import requests

        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        raw = cache_dir("treasury") / "observations.parquet"
        raw.write_bytes(r.content)
        o = pd.read_parquet(raw)
        b = o["breakdown"].map(json.loads)
        out = pd.DataFrame({
            "date": pd.to_datetime(o["attributes"].map(lambda s: json.loads(s)["gregorian_date"])),
            "type": b.map(lambda d: list(d.values())[0]),
            "term": b.map(lambda d: list(d.values())[1] if len(d) > 1 else ""),
            "bid_to_cover": o["value"],
        })
        out.to_parquet(path)
    return pd.read_parquet(path)


def long_bond_auction_dates(refresh: bool = False) -> pd.DatetimeIndex:
    """10-year and 30-year auction days (including reopenings)."""
    a = load_auctions(refresh)
    a = a[a["type"].isin(["Note", "Bond"]) & a["term"].str.match(LONG_TERMS)]
    return pd.DatetimeIndex(sorted(a["date"].unique()))
