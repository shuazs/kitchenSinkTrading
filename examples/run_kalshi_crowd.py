"""Kalshi crowd bias after big moves: calibration, taker lean and fade/follow backtests.

Needs the print-level Kalshi files and minute bars from the project folder
(KALSHI_CROWD_DATA, default /mnt/project-files): kalshi-catchup/data/stale_<SERIES>.parquet
(KXBTCD built the same way into kalshi-crowd/data/prints_KXBTCD.parquet) and minute-data/.

    python examples/run_kalshi_crowd.py
"""
import itertools
import os

import numpy as np
import pandas as pd

from backtester.kalshi_crowd import (BANDS, FEE_MULT, WINDOWS, Ladder, calibration, cap_capital, label_prints, move_triggers,
                                     move_z, simulate, spike_triggers, summarize)

ROOT = os.environ.get("KALSHI_CROWD_DATA", "/mnt/project-files")
UNDERLYING = {"KXINXU": "SPX", "KXNASDAQ100U": "NDX", "KXBTCD": "BTC", "KXETHD": "ETH"}
COLS = ["ticker", "count", "yes_price", "taker_side", "created_time", "event_ticker", "K", "close_time", "result", "tau"]


def prints(series):
    f = f"{ROOT}/kalshi-crowd/data/prints_KXBTCD.parquet" if series == "KXBTCD" else f"{ROOT}/kalshi-catchup/data/stale_{series}.parquet"
    x = pd.read_parquet(f, columns=COLS)
    return x[(x.tau >= 2) & (x.tau <= 1440) & x.result.isin(["yes", "no"])].sort_values("created_time").reset_index(drop=True)


def minute_price(name):
    """Price indexed by the time it is known (bar start + 1 minute)."""
    if name == "NDX":
        s = pd.read_parquet(f"{ROOT}/kalshi-catchup/data/NDX_synth_known.parquet")["close"]
    else:
        s = pd.read_parquet(f"{ROOT}/minute-data/{name}.parquet")["close"]
        s.index = s.index + pd.Timedelta("1min")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = s.index.astype("datetime64[ns, UTC]")
    return s


pd.set_option("display.width", 220)
calib, results = [], []
for series, und in UNDERLYING.items():
    x, z = prints(series), move_z(minute_price(und))
    lo, hi = x.created_time.min(), x.created_time.max()
    trig = {w: move_triggers(z, w) for w in WINDOWS}
    trig["spike"] = spike_triggers(x, z)
    trig = {k: v[(v.index > lo) & (v.index < hi)] for k, v in trig.items()}
    for w in WINDOWS:
        c = calibration(label_prints(x, trig[w], z))
        calib.append(c.assign(series=series, trigger=w))
        print(series, w, len(trig[w]), "triggers\n", c[c.q == "ALL"].round(3).to_string(index=False), flush=True)
    book = Ladder(x)
    for (w, tr), side, band, exe in itertools.product(trig.items(), ["follow", "fade"], BANDS, ["taker", "maker"]):
        out = [simulate(book, t, d, side, band, exe, FEE_MULT[series]) for t, d in tr.items()]
        f = pd.DataFrame([o for o in out if o and o["qty"] > 0])
        f = cap_capital(f) if len(f) else f
        results.append(dict(series=series, trigger=w, side=side, band=band, execution=exe, triggers=len(tr), **summarize(f)))
R = pd.DataFrame(results)
print(f"\n{len(R)} variants; taker variants with t>2: {((R.execution == 'taker') & (R.t > 2)).sum()}, t<-2: {((R.execution == 'taker') & (R.t < -2)).sum()}")
print(R.sort_values("t", ascending=False).round(2).head(15).to_string(index=False))
pd.concat(calib).to_csv("kalshi_crowd_calibration.csv", index=False)
R.to_csv("kalshi_crowd_backtests.csv", index=False)
