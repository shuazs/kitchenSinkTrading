"""FINRA short interest cross-section, Nov 2025 - Sep 2026 (pre-registered,
/mnt/project-files/short-interest/HYPOTHESES.md).

Each semi-monthly settlement is used from the open 10 business days later
(FINRA publishes ~7 days after settlement) to the open of the next formation
date. Universe: NYSE/Nasdaq common stocks, ADV >= 300k shares, price >= $5.
H1: top-minus-bottom decile of days to cover; H2: of the change in short
interest. Equal weight, 20 bp per leg per period; t over periods. Data: HF
FinzData/finra-short-interest + yfinance (prices for the names Yahoo served).

    python examples/run_short_interest.py
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

D = "/mnt/project-files/short-interest/"
HOLD, COST = "2026-05-01", 0.002
BAD = r"ETF|Fund|Trust|Notes|Warrant|Units|Rights|ETN|Shares"


def periods(si, px):
    o, c = px["open"], px["close"]
    days = o.index
    dates = sorted(si["settlement_date"].unique())
    form = [days[days.searchsorted(pd.Timestamp(d) + BDay(10))] for d in dates if
            days.searchsorted(pd.Timestamp(d) + BDay(10)) < len(days)]
    rows = []
    for d, f0, f1 in zip(dates, form[:-1], form[1:]):
        s = si[si["settlement_date"] == d].set_index("tk")
        s = s[s.index.isin(o.columns)]
        p_prev = c.loc[:f0].iloc[-2]  # close before formation day
        s = s[(p_prev.reindex(s.index) >= 5).values]
        r = (o.loc[f1, s.index] / o.loc[f0, s.index] - 1).dropna()
        s = s.loc[r.index].assign(r=r)
        rows.append((f0, s))
    return rows


def main():
    si = pd.read_parquet(D + "finz.parquet")
    si = si[~si["issuer"].str.contains(BAD, case=False, na=False) & (si["avg_daily_volume"] >= 300000)
            & si["market"].isin(["NYSE", "NNM", "SC"])].copy()
    si["tk"] = si["ticker"].str.replace(".", "-", regex=False)
    px = pd.read_parquet(D + "prices.parquet")
    out = []
    for f0, s in periods(si, px):
        row = {"formed": f0, "n": len(s), "all": s["r"].mean()}
        for col, lab in (("days_to_cover", "H1"), ("change_pct", "H2")):
            dec = pd.qcut(s[col].rank(method="first"), 10, labels=False)
            top, bot = s.loc[dec == 9, "r"].mean(), s.loc[dec == 0, "r"].mean()
            row[f"{lab} top-bottom"] = top - bot - 2 * COST
            row[f"{lab} top-all"] = top - row["all"] - 2 * COST
        out.append(row)
    d = pd.DataFrame(out).set_index("formed")
    print(f"{len(d)} periods, median {d['n'].median():.0f} stocks with prices")
    print((d.drop(columns="n") * 100).round(2).to_string())
    for lab, g in (("in-sample", d[d.index < HOLD]), ("holdout", d[d.index >= HOLD])):
        print(lab, "  ".join(f"{c} {g[c].mean() * 100:+.2f}% t {g[c].mean() / g[c].std() * np.sqrt(len(g)):+.2f}"
                             for c in d.columns if c.startswith("H")))


if __name__ == "__main__":
    main()
