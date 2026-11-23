"""Kalshi order flow around CPI / jobs / GDP releases (pre-registered H1-H4).

Data: /mnt/project-files/kalshi-macro/ (markets + 443k prints, Jul 2021 - Jan
2026, HF TrevorJS/kalshi-trades). In-sample: releases before 2025-07-01;
holdout: from then on. 10 variants in all (H1 4, H2 4, H4 2) plus the H3 table.

    python examples/run_kalshi_macro_flow.py
"""
import numpy as np
import pandas as pd

from backtester import kalshi_macro as km

D = "/mnt/project-files/kalshi-macro/"
HOLD = "2025-07-01"


def split(t):
    return {"in-sample": km.stats(t[t["release"] < HOLD]), "holdout": km.stats(t[t["release"] >= HOLD])}


def show(label, t):
    r = split(t)
    print(f"{label:<34} IS {r['in-sample']}  |  HO {r['holdout']}")


def main():
    m = km.release_events(pd.read_parquet(D + "markets.parquet"))
    p = pd.read_parquet(D + "prints.parquet")
    print(f"{len(m)} settled release markets, {m['event_ticker'].nunique()} events, "
          f"{m['release'].min():%Y-%m} to {m['release'].max():%Y-%m}")
    for wh in (24, 6):
        s = km.flow_signals(m, p, window_h=wh)
        for k in (0.2, 0.4):
            hit = s["imb_share"].abs() >= k
            follow = pd.Series(np.where(~hit, None, np.where(s["imb"] > 0, "yes", "no")), index=s.index)
            fade = follow.map({"yes": "no", "no": "yes"})
            show(f"H1 follow {wh}h k={k}", km.take(s, follow))
            show(f"H2 fade   {wh}h k={k}", km.take(s, fade))
    s = km.flow_signals(m, p, window_h=24)
    move = s["p_last"] - s["p_first"]
    big = move.abs() >= 0.05
    mom = pd.Series(np.where(~big, None, np.where(move > 0, "yes", "no")), index=s.index)
    show("H4 momentum 24h (|move| >= 5c)", km.take(s, mom))
    show("H4 reversal 24h", km.take(s, mom.map({"yes": "no", "no": "yes"})))
    print("\nH3 calibration in the last hour: taker net by side and price bucket")
    rows = []
    for side in ("yes", "no"):
        t = km.take(s, pd.Series(side, index=s.index), lo=0.0, hi=1.0)
        t["bucket"] = pd.cut(t["px"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        for b, g in t.groupby("bucket", observed=True):
            r = split(g)
            rows.append({"side": side, "bucket": str(b), **{f"IS_{k}": v for k, v in r["in-sample"].items()},
                         **{f"HO_{k}": v for k, v in r["holdout"].items()}})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
