"""Kalshi passive maker quotes in non-financial markets (alpha hunt #1).

Replays 131M Kalshi prints (shared folder event-markets/raw/) with
backtester.kalshi_maker: quote last print +/- k cents, 2 s latency, fill on
trade-through (conservative) or trade-at (front of queue), hold to
settlement, maker fee. Kill rule: under 0.5c/contract net or under 200
fills/month in the Jul 2024+ holdout.

    python examples/run_kalshi_maker.py [--shards 0-15] [--k 1 2 3 5]
"""
import argparse
import glob

import numpy as np
import pandas as pd

from backtester.kalshi_maker import maker_fills, settle

E = "/mnt/project-files/event-markets/"


def vol_bucket(v):
    return pd.cut(v, [-1, 1e3, 1e4, 1e5, 1e6, np.inf], labels=["<1k", "1-10k", "10-100k", "100k-1M", ">1M"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0-15")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5])
    a = ap.parse_args()
    lo, hi = map(int, a.shards.split("-"))
    mk = pd.read_parquet(E + "nf_markets.parquet", columns=["ticker", "cat", "result", "volume", "close_time"])
    mk = mk[mk.result.isin(["yes", "no"])]
    parts = []
    for f in sorted(glob.glob(E + "raw/nf_trades_*.parquet"))[lo : hi + 1]:
        p = pd.read_parquet(f)
        p = p[p.ticker.isin(mk.ticker) & p.yes_price.between(1, 99)]
        for k in a.k:
            for through in (True, False):
                s = settle(maker_fills(p, k=k, through=through), mk)
                s = s.assign(k=k, fill="through" if through else "at", holdout=s.created_time >= "2024-07-01",
                             vb=vol_bucket(s.volume).astype(str), month=s.created_time.dt.strftime("%Y-%m"))
                s["g"], s["n"] = s.gross * s.contracts, s.net * s.contracts
                parts.append(s.groupby(["k", "fill", "holdout", "cat", "vb", "side", "month"], observed=True)
                             [["contracts", "g", "n"]].sum().reset_index())
        print(f, flush=True)
    agg = pd.concat(parts).groupby(["k", "fill", "holdout", "cat", "vb", "side", "month"]).sum().reset_index()
    agg.to_parquet("kalshi_maker_agg.parquet")
    for keys in (["k", "fill", "holdout"], ["fill", "holdout", "cat"], ["fill", "holdout", "vb"]):
        sub = agg if "k" in keys else agg[agg.k == 1]
        t = sub.groupby(keys)[["contracts", "g", "n"]].sum()
        t["gross_c"], t["net_c"] = t.g / t.contracts, t.n / t.contracts
        print(t[["contracts", "gross_c", "net_c"]].round(2).unstack("holdout").to_string(), "\n")
    ho = agg[(agg.k == 1) & agg.holdout]
    print("holdout fills/month (k=1, through):", ho[ho.fill == "through"].groupby("month").contracts.sum().median())


if __name__ == "__main__":
    main()
