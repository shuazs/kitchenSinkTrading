"""Kalshi markets on scheduled US data releases (CPI, payrolls, unemployment, GDP, PCE, claims).

Trading stops at the release (markets close at 08:25-08:30 ET) and each
contract settles on the published number, so every print is pre-release.
Pre-registered tests (/mnt/project-files/kalshi-macro/HYPOTHESES.md):

- H1/H2: signed taker flow on a strike in a window before the release
  (taker-Yes minus taker-No contracts over window volume); follow it (H1) or
  fade it (H2) at the first taker print of that side in the last hour.
- H3: taker calibration by price bucket in the last hour.
- H4: price drift over the window, momentum or reversal into the release.

Entries are real taker prints of the side bought, price 10-90c, held to
settlement, taker fee 0.07 p (1-p); t is clustered by release event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .event_markets import kalshi_fee

RELEASE_HOURS = ("08:25", "08:29", "08:30")


def release_events(markets: pd.DataFrame) -> pd.DataFrame:
    """Settled markets whose event closes at a morning release time;
    adds `release` (the event's first close) and `y` (1 = Yes)."""
    m = markets[markets["result"].isin(["yes", "no"])].copy()
    rel = m.groupby("event_ticker")["close_time"].min().rename("release")
    m = m.join(rel, on="event_ticker")
    hhmm = m["release"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    m = m[hhmm.isin(RELEASE_HOURS)]
    m["y"] = (m["result"] == "yes").astype(int)
    return m


def flow_signals(m: pd.DataFrame, prints: pd.DataFrame, window_h: float, gap_h: float = 1.0,
                 lo: float = 0.10, hi: float = 0.90) -> pd.DataFrame:
    """Per market: window [release - window_h, release - gap_h) taker imbalance
    and price change, then the first taker-Yes / taker-No prints in
    [release - gap_h, release - 1 min) as entry prices (Yes ask / No ask)."""
    p = prints.merge(m[["ticker", "event_ticker", "release", "y"]], on="ticker")
    hrs = (p["release"] - p["created_time"]).dt.total_seconds() / 3600
    w = p[(hrs < window_h) & (hrs >= gap_h)].sort_values("created_time")
    signed = np.where(w["taker_side"] == "yes", w["count"], -w["count"])
    g = w.assign(signed=signed).groupby("ticker")
    sig = pd.DataFrame({"vol": g["count"].sum(), "imb": g["signed"].sum(),
                        "p_first": g["yes_price"].first() / 100, "p_last": g["yes_price"].last() / 100})
    sig["imb_share"] = sig["imb"] / sig["vol"]
    e = p[(hrs < gap_h) & (hrs >= 1 / 60)].sort_values("created_time")
    first = e.groupby(["ticker", "taker_side"])["yes_price"].first().unstack()
    ent = pd.DataFrame(index=first.index)
    ent["yes_ask"] = first.get("yes") / 100
    ent["no_ask"] = 1 - first.get("no") / 100
    out = m[["ticker", "event_ticker", "series", "release", "y"]].set_index("ticker").join(sig, how="inner").join(ent)
    return out.reset_index()


def take(df: pd.DataFrame, side: pd.Series, lo: float = 0.10, hi: float = 0.90) -> pd.DataFrame:
    """Buy `side` ('yes'/'no'/None per row) at its entry print; net after fee."""
    d = df.assign(side=side.values)
    px = np.where(d["side"] == "yes", d["yes_ask"], np.where(d["side"] == "no", d["no_ask"], np.nan))
    d = d.assign(px=px)
    d = d[d["px"].between(lo, hi)]
    win = np.where(d["side"] == "yes", d["y"], 1 - d["y"])
    return d.assign(net=win - d["px"] - kalshi_fee(d["px"]))


def stats(t: pd.DataFrame) -> dict:
    if len(t) < 3:
        return {"trades": len(t), "events": t["event_ticker"].nunique() if len(t) else 0, "net_c": np.nan, "t": np.nan}
    e = t.groupby("event_ticker")["net"].mean()
    tt = e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 and e.std() > 0 else np.nan
    return {"trades": len(t), "events": len(e), "net_c": round(t["net"].mean() * 100, 2), "t": round(float(tt), 2)}


SURPRISE_SERIES = {"CPI": 1, "KXCPI": 1, "CPICORE": 1, "KXCPICORE": 1, "PAYROLLS": 1, "KXPAYROLLS": 1,
                   "U3": -1, "KXU3": -1}  # sign: +1 = a higher number is hawkish


def parse_strike(text: str) -> float:
    """'Above 0.4%', '-100,000 or above', 'Above 500,000' -> number."""
    import re

    mt = re.search(r"-?[\d,]*\.?\d+", str(text))
    return float(mt.group().replace(",", "")) if mt else np.nan


def ladder_surprises(m: pd.DataFrame, prints: pd.DataFrame, window_h: float = 24.0) -> pd.DataFrame:
    """Per release event of an "above X" ladder: Kalshi implied median (where
    the last pre-release P(above) crosses 0.5), the settled value (midpoint of
    the highest Yes strike and the lowest No strike) and the surprise in
    strike steps, signed so that > 0 is hawkish (SURPRISE_SERIES)."""
    m = m[m["series"].isin(list(SURPRISE_SERIES))].copy()
    m["k"] = m["yes_sub_title"].map(parse_strike)
    p = prints.merge(m[["ticker", "release"]], on="ticker")
    hrs = (p["release"] - p["created_time"]).dt.total_seconds() / 3600
    last = p[(hrs > 0) & (hrs < window_h)].sort_values("created_time").groupby("ticker")["yes_price"].last() / 100
    m = m.join(last.rename("p_last"), on="ticker")
    rows = []
    for ev, g in m.dropna(subset=["k"]).groupby("event_ticker"):
        g = g.sort_values("k")
        step = np.median(np.diff(g["k"].unique())) if g["k"].nunique() > 1 else np.nan
        yes, no = g.loc[g["y"] == 1, "k"], g.loc[g["y"] == 0, "k"]
        if yes.empty or no.empty or not step:
            continue
        actual = (yes.max() + no.min()) / 2
        q = g.dropna(subset=["p_last"])
        if len(q) < 2:
            continue
        # P(above k) falls with k; interpolate the 0.5 crossing (clipped to the ladder)
        implied = float(np.interp(0.5, q["p_last"].values[::-1], q["k"].values[::-1]))
        rows.append({"event_ticker": ev, "series": g["series"].iloc[0], "release": g["release"].iloc[0],
                     "implied": implied, "actual": actual, "step": step,
                     "surprise": SURPRISE_SERIES[g["series"].iloc[0]] * (actual - implied) / step})
    return pd.DataFrame(rows)
