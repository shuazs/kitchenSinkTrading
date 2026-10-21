"""Hold stocks for a fixed time after a dated public event.

Shared by the event-driven strategies (insider clusters, activist 13D
filings, S&P 500 index changes). Each event becomes a signal on the first
trading day on or after its public date; the engine buys at the next open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


def liquid(data: MarketData, min_price: float, min_dollar_volume: float) -> np.ndarray:
    """Point-in-time liquidity mask: price and 20-day median dollar volume."""
    close = data.field("close")
    dollar_vol = (close * data.field("volume")).rolling(20, min_periods=10).median()
    return ((close >= min_price) & (dollar_vol >= min_dollar_volume)).to_numpy()


def hold_events(data: MarketData, events: pd.DataFrame, hold: int, k: int, ok: np.ndarray, sort_by=(), extra_cols=()):
    """Weights for holding up to `k` event names for `hold` trading days each.

    `events` has columns symbol and date (when it became public), plus any
    `sort_by` columns used to pick among events on a crowded day (descending).
    Slots are 1/k of equity (or the cash left). Held positions drift with
    prices: weights come from a frictionless shadow book and only change on
    entry and exit days, so the engine does not churn them back to equal size.
    Returns (weights, taken) where taken lists the events actually bought.
    """
    idx = data.index
    syms = data.symbols
    col = {s: j for j, s in enumerate(syms)}
    close_a = data.field("close").ffill().to_numpy(float)

    ev = events[events["symbol"].isin(syms)].copy()
    ev["day"] = idx.searchsorted(pd.to_datetime(ev["date"]).to_numpy(), side="left")
    ev = ev[ev["day"] < len(idx)].sort_values(["day", *sort_by], ascending=[True] + [False] * len(sort_by))
    by_day = {d: g for d, g in ev.groupby("day")}

    weights = np.zeros((len(idx), len(syms)))
    held: dict[int, int] = {}  # column -> exit day
    value = np.zeros(len(syms))  # shadow book, starts at 1.0 in cash
    cash = 1.0
    taken = []
    for i in range(len(idx)):
        changed = False
        if i > 0 and held:
            cols = list(held)
            r = close_a[i, cols] / close_a[i - 1, cols]
            value[cols] *= np.where(np.isfinite(r), r, 1.0)
        for j in [j for j, out in held.items() if i >= out]:
            cash += value[j]
            value[j] = 0.0
            del held[j]
            changed = True
        if i in by_day:
            for e in by_day[i].itertuples():
                if len(held) >= k:
                    break
                j = col[e.symbol]
                if j in held or not ok[i, j]:
                    continue
                alloc = min((cash + value.sum()) / k, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                value[j] = alloc
                held[j] = i + hold
                changed = True
                taken.append((idx[i], e.symbol, *(getattr(e, c) for c in extra_cols)))
        if changed or i == 0:
            weights[i] = value / (cash + value.sum())
        else:
            weights[i] = weights[i - 1]
    taken = pd.DataFrame(taken, columns=["signal_day", "symbol", *extra_cols])
    return pd.DataFrame(weights, index=idx, columns=syms), taken


class EventHold(Strategy):
    """Buy each event's stock the day after it is public and hold `hold_days`.

    Put a DataFrame with columns symbol and date (plus optional `priority`,
    higher first when more events fire than there are free slots) in
    `MarketData.extra[events_key]`. Liquidity filters use bars up to the
    signal day only.
    """

    name = "event_hold"
    params = {
        "events_key": "events",
        "hold_days": 63,
        "max_positions": 5,
        "min_price": 3.0,
        "min_dollar_volume": 1_000_000,
    }

    def events(self, data: MarketData) -> pd.DataFrame:
        return data.extra[self.p["events_key"]]

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        ev = self.events(data)
        sort_by = ("priority",) if "priority" in ev else ()
        ok = liquid(data, self.p["min_price"], self.p["min_dollar_volume"])
        w, self.taken = hold_events(data, ev, self.p["hold_days"], self.p["max_positions"], ok, sort_by, sort_by)
        return w
