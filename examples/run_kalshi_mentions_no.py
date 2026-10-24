"""Kalshi mention markets: buy No at 20-80c (alpha hunt batch 4, NEAR).

Reads the taker calibration aggregate built by the non-financial prediction
markets work (/mnt/project-files/event-markets/ev_agg.parquet: Kalshi prints
from HF TrevorJS/kalshi-trades, Aug 2024 - Jan 2026, net per contract after
the taker fee). Prints, for "will X say word Y" markets, the No-taker and
Yes-taker net by price bucket and year, the same No-side rule in other
categories as a control, and the split by series.

Result: No-takers at 20-80c earn +3.6c/contract in 2025-26 (2025 +2.4c, event
t 5.3; Jan 2026 +6.0c, t 2.6),
Yes-takers at 20-50c lose 10-13c; other categories' No-takers lose. There is
no untouched holdout, so it is a forward-test candidate, not a pass.
"""
import numpy as np
import pandas as pd

AGG = "/mnt/project-files/event-markets/ev_agg.parquet"
MID = ["[0.2, 0.35)", "[0.35, 0.5)", "[0.5, 0.65)", "[0.65, 0.8)"]


def table(x: pd.DataFrame, by: str) -> pd.DataFrame:
    g = x.groupby(by)[["cw", "nw"]].sum()
    ev = x.groupby([by, "event_ticker"])[["cw", "nw"]].sum()
    ev["net"] = ev["nw"] / ev["cw"]
    k = ev.groupby(level=0)["net"].agg(["count", "mean", "std"])
    return pd.DataFrame({"contracts": g["cw"], "events": k["count"], "net_c": g["nw"] / g["cw"] * 100,
                         "event_t": k["mean"] / k["std"] * np.sqrt(k["count"])}).round(2)


def main():
    e = pd.read_parquet(AGG)
    men = e[e["cat"] == "mentions"]
    print("Mentions, No-takers by price bucket (2025-26)")
    print(table(men[(men["taker_side"] == "no") & (men["year"] >= 2025)], "bucket").to_string())
    rule = men[(men["taker_side"] == "no") & men["bucket"].isin(MID)]
    print("\nRule (No at 20-80c) by year")
    print(table(rule, "year").to_string())
    print("\nComplement: Yes-takers at 20-50c by year")
    print(table(men[(men["taker_side"] == "yes") & men["bucket"].isin(MID[:2])], "year").to_string())
    print("\nControl: No-takers at 20-80c in other categories (2025-26)")
    print(table(e[(e["taker_side"] == "no") & e["bucket"].isin(MID) & (e["year"] >= 2025)], "cat").to_string())
    s = table(rule[rule["year"] >= 2025], "series")
    s = s[s["events"] >= 10].sort_values("contracts", ascending=False)
    print(f"\nBy series (>= 10 events): {(s['net_c'] > 0).mean():.0%} of {len(s)} positive")
    print(s.head(15).to_string())


if __name__ == "__main__":
    main()
