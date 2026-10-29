"""Kalshi daily 10-year Treasury yield markets vs a live futures nowcast (pre-registered).

KXTNOTED markets close at 15:25 ET and settle on the Treasury 10-year par
yield (FRED DGS10). At 15:00 ET the day's move is mostly known from 10-year
note futures, so: nowcast = DGS10(t-1) + beta x ZN=F % change from 15:00 ET
on t-1 to 15:00 ET on t (yfinance 1-hour bars; the bar starting 14:00 closes
at 15:00), beta and residual sigma fit in-sample; P(bucket) from
Normal(nowcast, sigma). Take the side with > 5c edge after fee (10c
sensitivity) at its first taker print in [15:00, 15:25) ET; hold to settlement.
A 12:00 ET version (entries 12:00-15:25, sigma refit) was added afterwards
because the last 25 minutes have almost no prints.
Placebo: nowcast frozen at DGS10(t-1). In-sample Jun 2024 - Jun 2025,
holdout Jul 2025 - Jan 2026 (/mnt/project-files/kalshi-tnote/HYPOTHESES.md).

    python examples/run_kalshi_tnote_nowcast.py
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtester.event_markets import kalshi_fee

D = "/mnt/project-files/kalshi-tnote/"
HOLD = pd.Timestamp("2025-07-01")


def bucket(title: str):
    """'between 4.08 and 4.10' -> (4.08, 4.10); 'below 4.06' -> (-inf, 4.06); 'above 4.44' -> (4.44, inf)."""
    t = str(title)
    nums = [float(x) for x in re.findall(r"\d+\.\d+", t)]
    if "between" in t and len(nums) >= 2:
        return nums[0], nums[1]
    if "below" in t and nums:
        return -np.inf, nums[0]
    if "above" in t and nums:
        return nums[0], np.inf
    return np.nan, np.nan


def nowcasts(hour: int = 15):
    y = pd.read_csv(D + "dgs10.csv", parse_dates=["observation_date"], na_values=".").set_index("observation_date")["DGS10"].dropna()
    zn = pd.read_parquet(D + "zn_1h.parquet")["Close"]
    zn.index = pd.DatetimeIndex(zn.index).tz_convert("America/New_York")
    at15 = zn[zn.index.hour == hour - 1]  # bar (hour-1):00-hour:00, its close is the price at `hour`
    at15.index = at15.index.tz_localize(None).normalize()
    t = pd.DataFrame({"y": y, "zn": at15}).dropna()
    t["y_prev"] = t["y"].shift(1)
    t["dzn"] = t["zn"].pct_change() * 100
    return t.dropna()


def main(hour: int = 15):
    nc = nowcasts(hour)
    fit = nc[nc.index < HOLD]
    beta = np.polyfit(fit["dzn"], fit["y"] - fit["y_prev"], 1)
    resid = fit["y"] - fit["y_prev"] - np.polyval(beta, fit["dzn"])
    sigma = resid.std()
    nc["pred"] = nc["y_prev"] + np.polyval(beta, nc["dzn"])
    print(f"beta {beta[0]:.3f} pp per 1% ZN, sigma {sigma * 100:.1f} bp vs {(fit['y'] - fit['y_prev']).std() * 100:.1f} bp unconditional")
    m = pd.read_parquet(D + "markets.parquet")
    m = m[m["result"].isin(["yes", "no"]) & m["series"].isin(["KXTNOTED", "TNOTED"])].copy()
    m[["lo", "hi"]] = pd.DataFrame(m["title"].map(bucket).tolist(), index=m.index)
    m["date"] = m["close_time"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    m = m.join(nc[["pred", "y_prev", "y"]], on="date").dropna(subset=["pred", "lo"])
    inside = ((m["y"] > m["lo"]) & (m["y"] <= m["hi"])) == (m["result"] == "yes")
    print(f"{len(m)} markets, {m['event_ticker'].nunique()} days; DGS10 agrees with the settlement on {inside.mean():.1%}")
    m["y01"] = (m["result"] == "yes").astype(int)
    p = pd.read_parquet(D + "prints.parquet").merge(m[["ticker", "close_time"]], on="ticker")
    p = p[(p["created_time"] >= p["close_time"] - pd.Timedelta(minutes=25 + 60 * (15 - hour))) & (p["created_time"] < p["close_time"])]
    first = p.sort_values("created_time").groupby(["ticker", "taker_side"])["yes_price"].first().unstack()
    m = m.join(first.rename(columns={"yes": "py", "no": "pn"}), on="ticker")
    m["yes_ask"], m["no_ask"] = m["py"] / 100, 1 - m["pn"] / 100
    for lab, center in [("nowcast", m["pred"]), ("placebo (yesterday)", m["y_prev"])]:
        q = norm.cdf((m["hi"] - center) / sigma) - norm.cdf((m["lo"] - center) / sigma)
        for thr in (0.05, 0.10):
            ey = q - m["yes_ask"] - kalshi_fee(m["yes_ask"])
            en = (1 - q) - m["no_ask"] - kalshi_fee(m["no_ask"])
            t = pd.concat([m[ey > thr].assign(px=lambda x: x["yes_ask"], win=lambda x: x["y01"]),
                           m[en > thr].assign(px=lambda x: x["no_ask"], win=lambda x: 1 - x["y01"])])
            t["net"] = t["win"] - t["px"] - kalshi_fee(t["px"])
            for per, g in [("IS", t[t["date"] < HOLD]), ("HO", t[t["date"] >= HOLD])]:
                e = g.groupby("event_ticker")["net"].mean()
                tt = e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 else np.nan
                print(f"{lab:<20} edge>{thr:.0%} {per}: {len(g)} trades, {len(e)} days, "
                      f"{g['net'].mean() * 100:+.2f}c, t {tt:.2f}")


if __name__ == "__main__":
    print("== decision at 15:00 ET (pre-registered) ==")
    main(15)
    print("\n== decision at 12:00 ET (added after seeing that 15:00-15:25 has almost no prints) ==")
    main(12)
