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
