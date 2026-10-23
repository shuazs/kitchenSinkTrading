"""Passive (maker) quotes on Kalshi, replayed from taker prints.

There is no order-book history, so quotes are rebuilt from prints. After
every taker order a bot re-centres a bid at `last - k` and an ask at
`last + k` cents, where `last` is the final price of the most recent order
at least `latency_s` seconds earlier (orders are prints sharing one
timestamp). The bid fills when a later taker sells YES below it (strictly,
`through=True`: the level was emptied, so we were filled even if last in
the queue) or at it (`through=False`: an upper bound, as if first in the
queue). Fills are capped by the printed size, held to settlement and charged
the maker fee 0.0175 * p * (1 - p) per contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KALSHI_MAKER = 0.0175


def maker_fee(price_cents) -> np.ndarray:
    """Maker fee in cents per contract (no per-order rounding)."""
    p = np.asarray(price_cents, float) / 100
    return KALSHI_MAKER * p * (1 - p) * 100


def orders(prints: pd.DataFrame) -> pd.DataFrame:
    """Collapse prints into taker orders: one row per (ticker, created_time),
    with the order's final price (highest for YES buyers, lowest for sellers)."""
    g = prints.groupby(["ticker", "created_time"], sort=False)
    o = g.agg(taker_side=("taker_side", "first"), hi=("yes_price", "max"), lo=("yes_price", "min")).reset_index()
    o["last"] = np.where(o["taker_side"] == "yes", o["hi"], o["lo"])
    return o[["ticker", "created_time", "last"]].sort_values("created_time")


def quote_reference(prints: pd.DataFrame, latency_s: float = 2.0) -> pd.DataFrame:
    """Each print with `ref`: the last price of the latest order at least
    `latency_s` seconds older (NaN before the market's first such order)."""
    o = orders(prints).rename(columns={"created_time": "ref_time", "last": "ref"})
    o["ref_time"] = o["ref_time"] + pd.Timedelta(seconds=latency_s)
    p = prints.sort_values("created_time")
    return pd.merge_asof(p, o, left_on="created_time", right_on="ref_time", by="ticker",
                         allow_exact_matches=True).drop(columns="ref_time")


def maker_fills(prints: pd.DataFrame, k: int = 1, latency_s: float = 2.0, through: bool = True) -> pd.DataFrame:
    """Simulated maker fills: ticker, created_time, side ('bid' buys YES,
    'ask' sells YES), price (cents) and contracts."""
    q = quote_reference(prints, latency_s).dropna(subset=["ref"])
    bid, ask = q["ref"] - k, q["ref"] + k
    px = q["yes_price"]
    if through:
        hit_bid, hit_ask = px < bid, px > ask
    else:
        hit_bid, hit_ask = px <= bid, px >= ask
    hit_bid &= (q["taker_side"] == "no") & (bid >= 1)
    hit_ask &= (q["taker_side"] == "yes") & (ask <= 99)
    cols = ["ticker", "created_time", "count"]
    b = q.loc[hit_bid, cols].assign(side="bid", price=bid[hit_bid])
    a = q.loc[hit_ask, cols].assign(side="ask", price=ask[hit_ask])
    return pd.concat([b, a]).rename(columns={"count": "contracts"}).sort_values("created_time").reset_index(drop=True)


def settle(fills: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Join each fill to its market's result and add P&L in cents per
    contract held to settlement, gross and after the maker fee."""
    m = markets.loc[markets["result"].isin(["yes", "no"]), ["ticker", "result"] + [c for c in ("cat", "volume", "close_time") if c in markets]]
    f = fills.merge(m, on="ticker")
    payout = np.where(f["result"] == "yes", 100.0, 0.0)
    f["gross"] = np.where(f["side"] == "bid", payout - f["price"], f["price"] - payout)
    f["net"] = f["gross"] - maker_fee(f["price"])
    return f


def summarize(settled: pd.DataFrame, by=None) -> pd.DataFrame:
    """Contracts, markets, gross and net cents per contract (size-weighted)."""
    s = settled.assign(g=settled["gross"] * settled["contracts"], n=settled["net"] * settled["contracts"])
    keys = by or (lambda _: "all")
    out = s.groupby(keys).agg(contracts=("contracts", "sum"), markets=("ticker", "nunique"), g=("g", "sum"), n=("n", "sum"))
    out["gross_c"] = out.pop("g") / out["contracts"]
    out["net_c"] = out.pop("n") / out["contracts"]
    return out
