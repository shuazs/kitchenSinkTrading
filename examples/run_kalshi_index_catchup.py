"""Kalshi S&P 500 markets of 2022-24 (INX, INXD, INXU, INXW) vs the live index.

The FX test's model and rule (examples/run_kalshi_fx_catchup.py) on the old,
busier S&P series, pre-registered in
/mnt/project-files/kalshi-index-old/HYPOTHESES.md: lognormal fair value from
the SPX 1-minute close known one minute before each taker print and its
trailing 60-minute vol; buy the printed side when the edge after fee is > 3c
(5-95c, first per market), hold to the 16:00 ET close. In-sample Aug 2022 -
Dec 2023, holdout 2024. Placebo: SPX 30 minutes stale.

    python examples/run_kalshi_index_catchup.py
"""
import re

import numpy as np
import pandas as pd

from run_kalshi_fx_catchup import log_loss, p_model, stats, trades

D = "/mnt/project-files/kalshi-index-old/"
HOLD = "2024-01-01"


def parse(title):
    t = title.lower().split(" be ", 1)[-1].split(" at the end", 1)[0]
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", t)]
    if t.startswith("between") and len(nums) >= 2:
        return "between", nums[0], nums[1]
    if t.startswith("above") and nums:
        return "above", nums[0], np.nan
    if t.startswith("below") and nums:
        return "below", nums[0], np.nan
    return None, np.nan, np.nan


def spx():
    b = pd.read_parquet("/mnt/project-files/minute-data/SPX.parquet")
    b = b.set_index("ts") if "ts" in b.columns else b
    b.index = pd.DatetimeIndex(b.index).as_unit("us")
    b["sig"] = np.log(b["close"]).diff().where(b.index.to_series().diff() == pd.Timedelta(minutes=1)).rolling(60, min_periods=40).std()
    return b[["close", "sig"]]


def build(lag_min=1):
    m = pd.read_parquet(D + "markets.parquet")
    m = m[m["series"].str.startswith("INX") & m["result"].isin(["yes", "no"])].copy()
    m[["kind", "lo", "hi"]] = pd.DataFrame(m["title"].map(parse).tolist(), index=m.index)
    m = m.dropna(subset=["kind"])
    m["y"] = (m["result"] == "yes").astype(int)
    p = pd.read_parquet(D + "prints.parquet").merge(
        m[["ticker", "series", "kind", "lo", "hi", "y", "close_time"]], on="ticker")
    p["mins"] = (p["close_time"] - p["created_time"]).dt.total_seconds() / 60
    b = spx()
    at = p["created_time"].dt.floor("min") - pd.Timedelta(minutes=lag_min + 1)  # bar start; close known lag min before
    fx = b.reindex(pd.DatetimeIndex(at))
    p["S"], p["sig"] = fx["close"].values, fx["sig"].values
    p["tau"] = (p["close_time"] - at - pd.Timedelta(minutes=1)).dt.total_seconds().values / 60
    p["S_settle"] = b["close"].reindex(pd.DatetimeIndex(p["close_time"] - pd.Timedelta(minutes=1))).values
    p = p.dropna(subset=["S", "sig"])
    p = p[p["sig"] > 0]
    p["q"] = p_model(p["kind"].values, p["lo"].values, p["hi"].values, p["S"].values, p["sig"].values, p["tau"].values)
    p["pair"] = "spx"
    return m, p


def show(label, t):
    print(f"{label:<32} IS {stats(t[t['close_time'] < HOLD])}  |  HO {stats(t[t['close_time'] >= HOLD])}")


def main():
    m, p = build()
    g = p.drop_duplicates("ticker").dropna(subset=["S_settle"])
    pred = np.where(g["kind"] == "above", g["S_settle"] > g["lo"], np.where(
        g["kind"] == "below", g["S_settle"] < g["lo"], (g["S_settle"] >= g["lo"]) & (g["S_settle"] <= g["hi"])))
    print(f"{len(m)} settled markets; SPX 16:00 close matches the result on {(pred.astype(int) == g['y']).mean():.1%} of {len(g)}")
    late = p[p["mins"] <= 60]
    print(f"last-hour prints {len(late)} on {late['ticker'].nunique()} markets; log loss model "
          f"{log_loss(late['y'], late['q']):.4f} vs price {log_loss(late['y'], late['yes_price'] / 100):.4f}")
    for thr in (0.03, 0.05, 0.10):
        show(f"H1 last hour, edge > {thr * 100:.0f}c", trades(p, thr))
    t = trades(p)
    for s in sorted(t["series"].unique()):
        show(f"  {s}", t[t["series"] == s])
    show("H3 whole session (390 min)", trades(p, window=390))
    _, pl = build(lag_min=30)
    show("H2 placebo (SPX 30 min stale)", trades(pl))


if __name__ == "__main__":
    main()
