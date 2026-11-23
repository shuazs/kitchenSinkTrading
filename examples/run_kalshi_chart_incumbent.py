"""Kalshi "top of the chart" markets: does yesterday's #1 keep the top spot? (pre-registered)

Series: Spotify daily/weekly song, artist and album charts (US, global,
UK, Canada, Australia, Brazil, Costa Rica, viral), Netflix #1 movie/show,
App Store #1 free app, Billboard Hot 100 / 200 #1. Incumbent = the item
(`yes_sub_title`) that won the series' most recent settled event before t_s.
t_s = event listing + 1 h; entry at the first taker print of the side bought
in the next 12 h (before close), held to settlement, Kalshi taker fee.
H13: Yes on the incumbent at 20-90c. H14: No on non-incumbents at No 50-90c.
Holdout from Jul 2025 (/mnt/project-files/kalshi-charts/HYPOTHESES.md).

    python examples/run_kalshi_chart_incumbent.py
"""
import numpy as np
import pandas as pd

from backtester.event_markets import kalshi_fee

D = "/mnt/project-files/kalshi-charts/"
HOLD = pd.Timestamp("2025-07-01", tz="UTC")


def norm(s):
    return s.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def entries(m: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    m = m[m["result"].isin(["yes", "no"])].copy()
    m["item"] = norm(m["yes_sub_title"])
    ev = m.groupby("event_ticker").agg(series=("series", "first"), open=("open_time", "min"),
                                       close=("close_time", "max"))
    win = m[m["result"] == "yes"].groupby("event_ticker")["item"].first()
    ev = ev.join(win.rename("winner"))
    ev["t_s"] = ev["open"] + pd.Timedelta(hours=1)
    inc = {}
    for s, g in ev.dropna(subset=["winner"]).groupby("series"):
        done = g.sort_values("close")
        for e, r in ev[ev["series"] == s].iterrows():
            prev = done[done["close"] < r["t_s"]]
            inc[e] = prev["winner"].iloc[-1] if len(prev) else None
    ev["incumbent"] = pd.Series(inc)
    m = m.join(ev[["t_s", "incumbent", "close"]].rename(columns={"close": "ev_close"}), on="event_ticker")
    m["is_inc"] = m["item"] == m["incumbent"]
    q = p.merge(m[["ticker", "t_s", "ev_close"]], on="ticker")
    q = q[(q["created_time"] >= q["t_s"]) & (q["created_time"] < np.minimum(q["t_s"] + pd.Timedelta(hours=12),
                                                                            q["ev_close"]))]
    first = q.sort_values("created_time").groupby(["ticker", "taker_side"])["yes_price"].first().unstack()
    m = m.join(first.rename(columns={"yes": "px_yes", "no": "px_no"}), on="ticker")
    m["yes_ask"] = m["px_yes"] / 100
    m["no_ask"] = 1 - m["px_no"] / 100
    m["y"] = (m["result"] == "yes").astype(int)
    return m[m["incumbent"].notna()]


def stats(t):
    if len(t) < 3:
        return {"n": len(t)}
    e = t.groupby("event_ticker")["net"].mean()
    return {"n": len(t), "events": len(e), "net_c": round(t["net"].mean() * 100, 2),
            "t": round(e.mean() / e.std() * np.sqrt(len(e)), 2), "win": round(t["win"].mean(), 3),
            "avg_px": round(t["px"].mean(), 3)}


def main():
    m = entries(pd.read_parquet(D + "markets.parquet"), pd.read_parquet(D + "prints.parquet"))
    inc = m[m["is_inc"]]
    print(f"{m['event_ticker'].nunique()} events; incumbent wins {inc['y'].mean():.1%} of {len(inc)} events")
    h13 = inc[inc["yes_ask"].between(0.2, 0.9)].assign(px=lambda x: x["yes_ask"], win=lambda x: x["y"])
    h14 = m[~m["is_inc"] & m["no_ask"].between(0.5, 0.9)].assign(px=lambda x: x["no_ask"], win=lambda x: 1 - x["y"])
    for lab, t in [("H13 Yes incumbent 20-90c", h13), ("H14 No challengers 50-90c", h14)]:
        t = t.assign(net=t["win"] - t["px"] - kalshi_fee(t["px"]))
        print(f"{lab:<27} IS {stats(t[t['t_s'] < HOLD])}\n{'':<27} HO {stats(t[t['t_s'] >= HOLD])}")
        fam = t["series"].str.replace(r"(D|W|2D)$", "", regex=True)
        print("   by family (all):", t.groupby(fam).apply(lambda g: round(g["net"].mean() * 100, 1)).to_dict())


if __name__ == "__main__":
    main()
