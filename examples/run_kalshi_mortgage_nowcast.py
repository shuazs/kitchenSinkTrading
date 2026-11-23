"""Kalshi 30-year mortgage-rate markets vs a Treasury nowcast (pre-registered).

FRM markets settle on Freddie Mac's weekly PMMS rate (FRED MORTGAGE30US,
Thursday noon ET; survey Monday-Wednesday). Nowcast: last PMMS + beta x the
change in the Mon-Wed mean 10-year yield (FRED DGS10), beta and residual
sigma fit on in-sample weeks. P(above k) = Phi((nowcast - k) / sigma). At
close - 12 h, take the side whose edge after fee is > 5c (10c sensitivity)
at its first taker print before close; hold to settlement.
In-sample: before Jul 2023; holdout to Feb 2024 (/mnt/project-files/kalshi-frm/).

    python examples/run_kalshi_mortgage_nowcast.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtester.event_markets import kalshi_fee
from backtester.kalshi_macro import parse_strike

D = "/mnt/project-files/kalshi-frm/"
HOLD = pd.Timestamp("2023-07-01", tz="UTC")


def nowcast_table():
    pm = pd.read_csv(D + "pmms.csv", parse_dates=["observation_date"]).set_index("observation_date")["MORTGAGE30US"]
    y = pd.read_csv(D + "dgs10.csv", parse_dates=["observation_date"], na_values=".").set_index("observation_date")["DGS10"]
    rows = []
    for d, v in pm["2019":].items():  # d = Thursday release date
        wk = y[d - pd.Timedelta(days=3):d - pd.Timedelta(days=1)].mean()
        prev = y[d - pd.Timedelta(days=10):d - pd.Timedelta(days=8)].mean()
        rows.append({"date": d, "pmms": v, "dy": wk - prev})
    t = pd.DataFrame(rows).set_index("date")
    t["prev"] = t["pmms"].shift(1)
    return t.dropna()


def main():
    nc = nowcast_table()
    fit = nc[nc.index < HOLD.tz_localize(None)]
    beta = np.polyfit(fit["dy"], fit["pmms"] - fit["prev"], 1)
    resid = (fit["pmms"] - fit["prev"]) - np.polyval(beta, fit["dy"])
    sigma = resid.std()
    nc["pred"] = nc["prev"] + np.polyval(beta, nc["dy"])
    print(f"beta {beta[0]:.2f}, intercept {beta[1]:.3f}, sigma {sigma:.3f} pp; random-walk sigma "
          f"{(fit['pmms'] - fit['prev']).std():.3f}")
    m = pd.read_parquet(D + "markets.parquet")
    m = m[m["result"].isin(["yes", "no"])].copy()
    m["k"] = m["title"].map(lambda s: parse_strike(str(s).split("above")[-1]))
    m["date"] = m["close_time"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    m = m.join(nc[["pred", "pmms"]], on="date").dropna(subset=["pred", "k"])
    m["q"] = norm.cdf((m["pred"] - m["k"]) / sigma)
    m["y"] = (m["result"] == "yes").astype(int)
    p = pd.read_parquet(D + "prints.parquet").merge(m[["ticker", "close_time"]], on="ticker")
    p = p[(p["created_time"] >= p["close_time"] - pd.Timedelta(hours=12)) & (p["created_time"] < p["close_time"])]
    first = p.sort_values("created_time").groupby(["ticker", "taker_side"])["yes_price"].first().unstack()
    m = m.join(first.rename(columns={"yes": "py", "no": "pn"}), on="ticker")
    m["yes_ask"], m["no_ask"] = m["py"] / 100, 1 - m["pn"] / 100
    print(f"{len(m)} markets, {m['event_ticker'].nunique()} weeks; with any entry print: "
          f"{(m['yes_ask'].notna() | m['no_ask'].notna()).sum()}")
    for thr in (0.05, 0.10):
        ey = m["q"] - m["yes_ask"] - kalshi_fee(m["yes_ask"])
        en = (1 - m["q"]) - m["no_ask"] - kalshi_fee(m["no_ask"])
        ty = m[ey > thr].assign(px=lambda x: x["yes_ask"], win=lambda x: x["y"])
        tn = m[en > thr].assign(px=lambda x: x["no_ask"], win=lambda x: 1 - x["y"])
        t = pd.concat([ty, tn])
        t["net"] = t["win"] - t["px"] - kalshi_fee(t["px"])
        for lab, g in [("IS", t[t["close_time"] < HOLD]), ("HO", t[t["close_time"] >= HOLD])]:
            e = g.groupby("event_ticker")["net"].mean()
            tt = e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 else np.nan
            print(f"edge > {thr:.0%} {lab}: {len(g)} trades, {len(e)} weeks, {g['net'].mean() * 100:+.2f}c, t {tt:.2f}")


if __name__ == "__main__":
    main()
