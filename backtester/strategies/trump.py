"""Strategies keyed to Donald Trump's posts and his administration's deals.

All three use only public information, on daily bars. Trading his posts in
real time is left out on purpose: institutions get them in milliseconds and
prices often move before the post (see docs/README.md for the research).

PostRiskOverlay: cut a base strategy's exposure after a burst of posts on a
    topic (tariffs by default). Daily post counts go in
    `MarketData.extra[counts_key]` (see `backtester.data.trump_posts.daily_counts`).
EventDip: buy an index ETF at the open after a sharp down day, optionally only
    when he posted about tariffs that session (the "TACO" trade, from the FT's
    "Trump Always Chickens Out"), and hold a few days.
GovStakeEvents: buy companies after the first report that the government is
    taking an equity stake, and hold a fixed number of days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .insider import InsiderCluster


class PostRiskOverlay(Strategy):
    """Scale down `base` for `hold` sessions after at least `min_posts` posts
    matching the topic in the trailing `window` sessions.

    While triggered, every weight except `risk_off` is multiplied by `scale`
    and the freed weight goes to `risk_off` (or cash). Days with no post data
    (outside the archive) never trigger, so compare against the base over the
    same dates.
    """

    name = "post_risk_overlay"
    params = {
        "base": None,
        "counts_key": "trump_counts",
        "window": 5,
        "min_posts": 3,
        "hold": 5,
        "scale": 0.5,
        "risk_off": None,
    }

    def triggered(self, data: MarketData) -> pd.Series:
        counts = data.extra[self.p["counts_key"]].reindex(data.index).fillna(0)
        burst = counts.rolling(self.p["window"], min_periods=1).sum() >= self.p["min_posts"]
        # Stay cut for `hold` sessions after the last burst day.
        return burst.astype(float).rolling(self.p["hold"], min_periods=1).max().astype(bool)

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        w = self.p["base"].generate_targets(data).reindex(columns=data.symbols).reindex(data.index).ffill().fillna(0.0)
        on = self.triggered(data)
        risky = [s for s in w.columns if s != self.p["risk_off"]]
        before = w[risky].sum(axis=1)
        w.loc[on, risky] = w.loc[on, risky] * self.p["scale"]
        if self.p["risk_off"]:
            w[self.p["risk_off"]] += before - w[risky].sum(axis=1)
        return w


class EventDip(Strategy):
    """Buy `symbols` at the next open after a session that fell at least `dip`, hold `hold` sessions.

    event_key: name of a boolean date Series in `MarketData.extra` (e.g. "he
    posted about tariffs this session"). mode "require" buys only dips on
    event days, "exclude" only dips on other days, None ignores events (the
    control: any dip). Each symbol is traded on its own drop; open positions
    share equity equally.
    """

    name = "event_dip"
    params = {"symbols": ["SPY"], "dip": -0.015, "hold": 5, "event_key": None, "mode": None, "trend_ma": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        syms = self.p["symbols"]
        close = data.field("close")[syms].ffill()
        ret = close.pct_change(fill_method=None)
        enter = ret <= self.p["dip"]
        if self.p["trend_ma"]:
            enter &= close > close.rolling(self.p["trend_ma"]).mean()
        if self.p["mode"]:
            ev = data.extra[self.p["event_key"]].reindex(data.index).fillna(False).astype(bool)
            ev = ev if self.p["mode"] == "require" else ~ev
            enter = enter.mul(ev, axis=0).astype(bool)
        # A fresh entry restarts the clock, so back-to-back drops extend the hold.
        held = enter.astype(float).rolling(self.p["hold"], min_periods=1).max().astype(bool)
        n = held.sum(axis=1).clip(lower=1)
        w = held.astype(float).div(n, axis=0)
        return w.reindex(columns=data.symbols, fill_value=0.0)


class GovStakeEvents(InsiderCluster):
    """Buy after the first report of a government equity stake.

    Events (from `backtester.data.gov_stakes.stake_events(index)`) go in
    `MarketData.extra[events_key]`. `delay` waits that many more sessions
    after the first possible entry, to test buying after the opening pop.
    `outcomes` keeps only events whose story ended that way ("deal", "talks",
    "denied"); note that this uses hindsight unless it includes all three.
    """

    name = "gov_stake_events"
    params = {
        "hold_days": 20,
        "max_positions": 4,
        "delay": 0,
        "outcomes": ("deal", "talks", "denied"),
        "min_price": 1.0,
        "min_dollar_volume": 1_000_000,
        "events_key": "gov_stakes",
    }

    def events(self, data: MarketData) -> pd.DataFrame:
        ev = data.extra[self.p["events_key"]]
        ev = ev[ev["outcome"].isin(self.p["outcomes"]) & ev["symbol"].isin(data.symbols)].dropna(subset=["filed"])
        idx = data.index
        pos = idx.searchsorted(ev["filed"].to_numpy(), side="left") + self.p["delay"]
        ev = ev.assign(filed=[idx[i] if i < len(idx) else pd.NaT for i in pos]).dropna(subset=["filed"])
        # InsiderCluster ranks same-day events by these; all stakes rank equal.
        return ev.assign(n_insiders=1, value=0.0)
