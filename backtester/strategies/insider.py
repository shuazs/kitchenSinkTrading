"""Insider cluster buying.

Research basis:
- Lakonishok and Lee (2001), "Are Insider Trades Informative?": insider
  purchases (not sales) predict returns, mostly in small firms.
- Cohen, Malloy and Pomorski (2012), "Decoding Inside Information":
  "opportunistic" (non-routine) insider trades earn the abnormal returns.
- Alldredge and Blank (2019), "Do Insiders Cluster Trades with Colleagues?":
  purchases made by several insiders close together are more informative
  than solo purchases.

Rule: a cluster fires on a filing date when, within the trailing `window`
calendar days of filing dates, at least `min_insiders` distinct insiders
(officers or directors, optionally requiring at least one officer) filed
open-market purchases (Form 4 code P) worth at least `min_value` dollars in
total. Because the window runs on FILING dates and the signal is formed at the
close of the filing day (or the next trading day if it was a holiday), the
engine buys at the next open: nothing is used before it was public.

Positions are held `hold_days` trading days in up to `max_positions` slots of
1/max_positions of equity each (or the cash left, if less). Held positions are
left to drift: targets only change on entry/exit days, and then carry each
held name's drifted weight from a frictionless shadow book, so with the
engine's default rebalance="on_change" (and a min_trade_value of a few
percent of a slot) existing positions are not churned back to equal weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


def cluster_signals(
    trades: pd.DataFrame,
    min_insiders: int = 2,
    window: int = 30,
    min_value: float = 100_000,
    require_officer: bool = False,
    insiders_only: bool = True,
    exclude_plan: bool = True,
    min_mcap: float | None = None,
    max_mcap: float | None = None,
) -> pd.DataFrame:
    """Cluster purchase events from open-market trades.

    `trades` needs columns symbol, filed, owner_cik (or owner), code, value,
    and optionally officer, director, plan_10b5_1. Returns one row per
    (symbol, filing date) on which the cluster condition holds and a new
    qualifying filing arrived: symbol, filed, n_insiders, n_officers, value,
    mcap. If `trades` has an `mcap` column (see insider_bulk.add_market_cap),
    min_mcap/max_mcap keep only events whose filing-day size is in range.
    """
    df = trades[trades["code"] == "P"].dropna(subset=["symbol"])
    df = df[df["value"] > 0]
    owner = "owner_cik" if "owner_cik" in df else "owner"
    if insiders_only and {"officer", "director"} <= set(df.columns):
        # Pure 10% holders are usually funds, not people with inside information.
        df = df[df["officer"].astype(bool) | df["director"].astype(bool)]
    if exclude_plan and "plan_10b5_1" in df:
        df = df[~df["plan_10b5_1"].astype(bool)]
    if "officer" not in df:
        df = df.assign(officer=False)
    daily = (
        df.groupby(["symbol", "filed", owner], as_index=False)
        .agg(value=("value", "sum"), officer=("officer", "any"))
        .sort_values(["symbol", "filed"])
    )
    win = np.timedelta64(window - 1, "D")
    has_mcap = "mcap" in df
    if has_mcap:
        last_mcap = df.groupby(["symbol", "filed"])["mcap"].last()
    rows = []
    for sym, g in daily.groupby("symbol", sort=False):
        dates = g["filed"].to_numpy()
        owners = g[owner].to_numpy()
        vals = g["value"].to_numpy(float)
        offs = g["officer"].to_numpy(bool)
        for d in np.unique(dates):
            lo = np.searchsorted(dates, d - win, side="left")
            hi = np.searchsorted(dates, d, side="right")
            o = owners[lo:hi]
            n = len(set(o))
            officers = len(set(o[offs[lo:hi]]))
            total = vals[lo:hi].sum()
            if n >= min_insiders and total >= min_value and (officers > 0 or not require_officer):
                cap = last_mcap.get((sym, d), np.nan) if has_mcap else np.nan
                rows.append((sym, pd.Timestamp(d), n, officers, total, cap))
    ev = pd.DataFrame(rows, columns=["symbol", "filed", "n_insiders", "n_officers", "value", "mcap"])
    if min_mcap is not None or max_mcap is not None:
        # Unknown size (no XBRL share count) is excluded when a band is set.
        ev = ev[ev["mcap"].between(min_mcap or 0, max_mcap or np.inf)]
    return ev.reset_index(drop=True)


class InsiderCluster(Strategy):
    """Buy small caps after cluster insider buying, hold a fixed number of days.

    Put the open-market trades table (see `backtester.data.insider_bulk.
    open_market_trades`) in `MarketData.extra[trades_key]`, or precomputed
    events from `cluster_signals` in `extra[events_key]`.

    min_price / min_dollar_volume are point-in-time liquidity filters using
    bars up to the signal day: skip sub-$min_price stocks and those whose
    20-day median dollar volume is below min_dollar_volume.
    """

    name = "insider_cluster"
    params = {
        "min_insiders": 2,
        "window": 30,
        "min_value": 100_000,
        "require_officer": False,
        "insiders_only": True,
        "exclude_plan": True,
        "min_mcap": None,
        "max_mcap": None,
        "hold_days": 60,
        "max_positions": 10,
        "min_price": 2.0,
        "min_dollar_volume": 250_000,
        "trades_key": "insider_trades",
        "events_key": "insider_events",
    }

    def events(self, data: MarketData) -> pd.DataFrame:
        if self.p["events_key"] in data.extra:
            # Precomputed events (from a looser or equal rule) are tightened
            # by this instance's thresholds; window/size bands are fixed upstream.
            ev = data.extra[self.p["events_key"]]
            keep = (ev["n_insiders"] >= self.p["min_insiders"]) & (ev["value"] >= self.p["min_value"])
            if self.p["require_officer"]:
                keep &= ev["n_officers"] > 0
            ev = ev[keep]
        else:
            ev = cluster_signals(
                data.extra[self.p["trades_key"]],
                min_insiders=self.p["min_insiders"],
                window=self.p["window"],
                min_value=self.p["min_value"],
                require_officer=self.p["require_officer"],
                insiders_only=self.p["insiders_only"],
                exclude_plan=self.p["exclude_plan"],
                min_mcap=self.p["min_mcap"],
                max_mcap=self.p["max_mcap"],
            )
        return ev[ev["symbol"].isin(data.symbols)]

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        syms = data.symbols
        col = {s: j for j, s in enumerate(syms)}
        close = data.field("close")
        dollar_vol = (close * data.field("volume")).rolling(20, min_periods=10).median()
        close_a = close.ffill().to_numpy(float)
        ok = ((close >= self.p["min_price"]) & (dollar_vol >= self.p["min_dollar_volume"])).to_numpy()

        # Signal day = first trading day on or after the filing date.
        ev = self.events(data).copy()
        ev["day"] = idx.searchsorted(ev["filed"].to_numpy(), side="left")
        # When more clusters fire than there are free slots, prefer more
        # insiders, then more dollars (both stronger signals in the papers).
        ev = ev[ev["day"] < len(idx)].sort_values(
            ["day", "n_insiders", "value"], ascending=[True, False, False]
        )
        by_day = {d: g for d, g in ev.groupby("day")}

        k, hold = self.p["max_positions"], self.p["hold_days"]
        weights = np.zeros((len(idx), len(syms)))
        held: dict[int, int] = {}  # column -> exit day
        value = np.zeros(len(syms))  # shadow book, starts at 1.0 in cash
        cash = 1.0
        self.taken = []
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
                    self.taken.append((idx[i], e.symbol, e.n_insiders, e.value))
            if changed or i == 0:
                weights[i] = value / (cash + value.sum())
            else:
                weights[i] = weights[i - 1]
        self.taken = pd.DataFrame(self.taken, columns=["signal_day", "symbol", "n_insiders", "value"])
        return pd.DataFrame(weights, index=idx, columns=syms)
