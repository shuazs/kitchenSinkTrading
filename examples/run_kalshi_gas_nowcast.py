"""Kalshi weekly gas-price markets vs an RBOB-futures nowcast (pre-registered).

GAS markets close Monday ~17:00 ET; settlement is taken to be the EIA weekly
US regular retail price (FRED GASREGW, dated Monday). Nowcast: last week's
EIA price + b1 x this week's RBOB change (Friday to Friday, yfinance RB=F)
+ b2 x last week's RBOB change, fit with sigma on in-sample weeks.
P(above k) = Phi((nowcast - k) / sigma). From Saturday 00:00 UTC to the
close, take the side with > 5c edge after fee (10c sensitivity) at its first
taker print; hold to settlement. In-sample 2021-2022, holdout 2023
(/mnt/project-files/kalshi-gas/).

    python examples/run_kalshi_gas_nowcast.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtester.event_markets import kalshi_fee
from backtester.kalshi_macro import parse_strike

D = "/mnt/project-files/kalshi-gas/"
HOLD = pd.Timestamp("2023-01-01", tz="UTC")


def nowcast_table():
    g = pd.read_csv(D + "gasregw.csv", parse_dates=["observation_date"]).set_index("observation_date")["GASREGW"]
    rb = pd.read_csv(D + "rbob.csv", parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    rbw = rb.resample("W-FRI").last()
    t = pd.DataFrame({"gas": g["2020-03":]})
    t["prev"] = t["gas"].shift(1)
    fri = t.index - pd.Timedelta(days=3)
    t["d1"] = rbw.reindex(fri).values - rbw.reindex(fri - pd.Timedelta(days=7)).values
    t["d2"] = rbw.reindex(fri - pd.Timedelta(days=7)).values - rbw.reindex(fri - pd.Timedelta(days=14)).values
    return t.dropna()


def main():
    nc = nowcast_table()
    fit = nc[nc.index < HOLD.tz_localize(None)]
    X = np.column_stack([fit["d1"], fit["d2"], np.ones(len(fit))])
    b, *_ = np.linalg.lstsq(X, fit["gas"] - fit["prev"], rcond=None)
    sigma = (fit["gas"] - fit["prev"] - X @ b).std()
    nc["pred"] = nc["prev"] + np.column_stack([nc["d1"], nc["d2"], np.ones(len(nc))]) @ b
    print(f"b1 {b[0]:.2f} b2 {b[1]:.2f} c {b[2]:.3f}; sigma {sigma:.3f} vs random walk {(fit['gas'] - fit['prev']).std():.3f}")
    m = pd.read_parquet(D + "markets.parquet")
    m = m[m["result"].isin(["yes", "no"])].copy()
    m["k"] = m["title"].map(lambda s: parse_strike(str(s).replace("**", "").split("$")[-1]))
    m["date"] = m["close_time"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    m["date"] = m["date"] - pd.to_timedelta(m["date"].dt.dayofweek, unit="D")  # the Monday of that week
    m = m.join(nc[["pred", "gas"]], on="date").dropna(subset=["pred", "k"])
    agree = ((m["gas"] > m["k"]) == (m["result"] == "yes")).mean()
    print(f"{len(m)} markets, {m['event_ticker'].nunique()} weeks; EIA value agrees with the settlement on {agree:.0%}")
    m["q"] = norm.cdf((m["pred"] - m["k"]) / sigma)
    m["y"] = (m["result"] == "yes").astype(int)
    m["t_s"] = m["close_time"].dt.normalize() - pd.Timedelta(days=2)  # Saturday 00:00 UTC
    p = pd.read_parquet(D + "prints.parquet").merge(m[["ticker", "close_time", "t_s"]], on="ticker")
    p = p[(p["created_time"] >= p["t_s"]) & (p["created_time"] < p["close_time"])]
    first = p.sort_values("created_time").groupby(["ticker", "taker_side"])["yes_price"].first().unstack()
    m = m.join(first.rename(columns={"yes": "py", "no": "pn"}), on="ticker")
    m["yes_ask"], m["no_ask"] = m["py"] / 100, 1 - m["pn"] / 100
    for thr in (0.05, 0.10):
        ey = m["q"] - m["yes_ask"] - kalshi_fee(m["yes_ask"])
        en = (1 - m["q"]) - m["no_ask"] - kalshi_fee(m["no_ask"])
        t = pd.concat([m[ey > thr].assign(px=lambda x: x["yes_ask"], win=lambda x: x["y"]),
                       m[en > thr].assign(px=lambda x: x["no_ask"], win=lambda x: 1 - x["y"])])
        t["net"] = t["win"] - t["px"] - kalshi_fee(t["px"])
        for lab, g in [("IS", t[t["close_time"] < HOLD]), ("HO", t[t["close_time"] >= HOLD])]:
            e = g.groupby("event_ticker")["net"].mean()
            tt = e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 else np.nan
            print(f"edge > {thr:.0%} {lab}: {len(g)} trades, {len(e)} weeks, {g['net'].mean() * 100:+.2f}c, t {tt:.2f}")


if __name__ == "__main__":
    main()
