"""Prediction markets vs the asset they track, traded as pairs.

    python examples/run_prediction_market_pairs.py [project-files dir]

Needs the project's prediction-market files (Kalshi and Polymarket history
come from Hugging Face mirrors, not an API; see docs/README.md):
  kalshi-pairs/data/levels_<SPX|NDX|BTC|ETH>.parquet   5-min Kalshi implied level vs spot
  kalshi-catchup/data/stale_<series>.parquet           Kalshi prints with model fair value F0
  prediction-markets/poly/themes.parquet, prices/      Polymarket indices, hourly futures
"""
import sys

import numpy as np
import pandas as pd

from backtester.pairs import asset_leg_trades, fade_contract_spread, half_life, rolling_hedge, zscore

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project-files"
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)


def summarize(tr, col):
    x = tr[col].dropna()
    return {"n": len(x), "mean": x.mean(), "t": x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan}


# 1. Level pairs: Kalshi ladder implied level vs the index/coin, 5-min bars.
COST_BP = {"SPX": 1.5, "NDX": 1.5, "BTC": 6.0, "ETH": 6.0}
rows = []
for a, cost in COST_BP.items():
    try:
        x = pd.read_parquet(f"{ROOT}/kalshi-pairs/data/levels_{a}.parquet")
    except FileNotFoundError:
        continue
    x = x[((x.close_time - x.index).dt.total_seconds() >= 600)]
    g = np.log(x.level / x.spot) * 1e4
    x = x[g.abs() < 200]
    g = g[g.abs() < 200]
    same = x.event.eq(x.event.shift()) & (x.index.to_series().diff() == pd.Timedelta("5min"))
    # Asset leg only, on a 5-min grid, entering one bar (5 min) after the signal. With a 1-minute delay
    # (scripts in kalshi-pairs/scripts) ETH grosses about +4.5 bp per 30 min, still under realistic crypto costs.
    grid = pd.date_range(x.index.min(), x.index.max(), freq="5min")
    spot = np.log(x.spot.reindex(grid).ffill(limit=1))
    z = zscore(g, 24).reindex(grid)
    for h in [1, 3, 6]:
        tr = asset_leg_trades(spot, -z, entry=2.0, hold=h, cost_bp=cost)  # Kalshi above spot -> buy the asset
        rows.append({"asset": a, "half_life_min": 5 * half_life(g, same), "hold_min": 5 * h,
                     **{f"net_{k}": v for k, v in summarize(tr, "net_bp").items()}})
print("Level pairs: Kalshi implied level vs asset, trade the asset leg at |z| > 2 (bp per trade)")
print(pd.DataFrame(rows).to_string(index=False))

# 2. Contract pairs: Kalshi strike price vs digital fair value from the asset, fills on real prints.
rows = []
for ser in ["KXINXU", "KXNASDAQ100U", "KXETHD"]:
    try:
        p = pd.read_parquet(f"{ROOT}/kalshi-catchup/data/stale_{ser}.parquet",
                            columns=["ticker", "created_time", "close_time", "result", "p", "F0", "yes", "tau"])
    except FileNotFoundError:
        continue
    p = p[(p.tau.between(5, 360)) & (p.F0.between(0.1, 0.9))].assign(e=lambda d: d.p - d.F0)
    for thr, hold in [(0.05, 5), (0.08, 30), (0.12, None)]:
        tr = fade_contract_spread(p, thr, hold)
        rows.append({"series": ser, "threshold_c": thr * 100, "hold": hold or "settle",
                     "gross_c": tr.gross.mean() * 100, **{f"net_{k}": v for k, v in summarize(tr, "net").items()}})
out = pd.DataFrame(rows)
out["net_mean"] *= 100
print("\nContract pairs: fade Kalshi price vs fair value (cents per contract, taker fills and fees)")
print(out.rename(columns={"net_mean": "net_c"}).to_string(index=False))

# 3. Slow pairs: Polymarket indices vs futures, hourly, 20-day rolling hedge, asset leg only.
PM = f"{ROOT}/prediction-markets"


def hourly(t):
    c = pd.read_parquet(f"{PM}/prices/{t}.parquet")["Close"].dropna()
    c.index = c.index.tz_convert("UTC").floor("h")
    return np.log(c[~c.index.duplicated(keep="last")])


th = pd.read_parquet(f"{PM}/poly/themes.parquet")
rows = []
for name, pm, fut, cost in [("Iran escalation vs WTI", th.iran.fillna(0).cumsum(), "CL=F", 4.0),
                            ("Fed expected bps vs 2y note", th.fed_level, "ZT=F", 1.0),
                            ("Fed expected bps vs 10y note", th.fed_level, "ZN=F", 1.5)]:
    y = hourly(fut)
    df = pd.concat([y.rename("y"), pm.rename("x")], axis=1).ffill(limit=6).dropna()
    df = df[df.index.isin(y.index)]
    sp = rolling_hedge(df.y, df.x, 24 * 20).spread
    z = zscore(sp, 24 * 20)
    for h in [1, 3, 6, 24]:
        tr = asset_leg_trades(df.y, z, entry=2.0, hold=h, cost_bp=cost)
        rows.append({"pair": name, "half_life_h": half_life(sp), "hold_h": h,
                     **{f"net_{k}": v for k, v in summarize(tr, "net_bp").items()}})
print("\nSlow pairs: Polymarket index vs futures, |z| > 2, asset leg (bp per trade)")
print(pd.DataFrame(rows).to_string(index=False))
