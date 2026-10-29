"""Daily on-chain metrics from the Coin Metrics community CSVs (GitHub raw).

Day `t` rows are complete only at the end of day `t` (UTC), so callers should
act no earlier than the next daily close. The repo snapshot ends 2026-05-23.
"""
from __future__ import annotations

import pandas as pd
import requests

from .cache import cache_dir

URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{}.csv"
COINS = ["btc", "eth", "usdt", "usdc", "dai", "ltc", "bch", "doge", "xrp", "ada", "bnb", "trx", "link"]


def load_coin(coin: str, refresh: bool = False) -> pd.DataFrame:
    """Daily frame indexed by date; all-empty columns dropped."""
    local = cache_dir("coinmetrics") / f"{coin}.csv"
    if not local.exists() or refresh:
        r = requests.get(URL.format(coin), timeout=120)
        r.raise_for_status()
        local.write_bytes(r.content)
    df = pd.read_csv(local, index_col=0, parse_dates=True)
    return df.dropna(axis=1, how="all")


def stablecoin_supply(coins=("usdt", "usdc", "dai")) -> pd.Series:
    """Summed circulating supply (USD) of the big stablecoins."""
    parts = [load_coin(c)["SplyCur"] for c in coins]
    return pd.concat(parts, axis=1).ffill().sum(axis=1, min_count=1)
