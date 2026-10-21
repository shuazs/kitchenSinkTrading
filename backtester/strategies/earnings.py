"""Earnings announcement premium.

Research basis:
- Frazzini and Lamont (2007), "The Earnings Announcement Premium and Trading
  Volume": stocks earn about 7%/yr more (annualized) in months they are
  expected to announce earnings.
- Savor and Wilson (2016), "Earnings Announcements and Systematic Risk":
  the premium is compensation for announcement-day risk.
- Barber, De George, Lehavy and Trueman (2013), "The Earnings Announcement
  Premium Around the Globe": present in 46 countries.

Rule: predict each stock's next announcement as the date it reported a
year earlier plus 52 weeks (same weekday), which is known in advance. Buy
`pre_days` trading days before the predicted date and sell `post_days`
trading days after it, so the trade does not need the surprise, only the
calendar. Up to `max_positions` names are held at 1/max_positions each;
the rest of the book sits in `idle` (e.g. SPY) or cash, which makes it a
long-only tilt toward announcers.

Announcement dates come from `backtester.data.edgar.earnings_dates`
(8-K Item 2.02 filings), passed as `extra[dates_key]`: a dict of
symbol -> sequence of announcement dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


def announcement_windows(
    dates: dict[str, pd.Series | pd.DatetimeIndex | list],
    index: pd.DatetimeIndex,
    pre_days: int = 5,
    post_days: int = 1,
    predict_days: int = 364,
) -> pd.DataFrame:
    """Trading-day windows around predicted announcements.

    Returns rows of symbol, predicted (date), entry and exit (positions in
    `index`): the target is set at the close of `entry` and cleared at the
    close of `exit`, so the engine buys `pre_days` sessions before the
    predicted date and sells `post_days` sessions after it, at the open.
    """
    rows = []
    for sym, ds in dates.items():
        for d in pd.DatetimeIndex(ds):
            pred = d + pd.Timedelta(days=predict_days)
            k = index.searchsorted(pred)
            entry, exit_ = k - pre_days - 1, k + post_days - 1
            if entry >= 0 and exit_ < len(index):
                rows.append((sym, pred, entry, exit_))
    return pd.DataFrame(rows, columns=["symbol", "predicted", "entry", "exit"])


class EarningsPremium(Strategy):
    """Hold stocks around their predicted earnings dates, else `idle`."""

    name = "earnings_premium"
    params = {
        "pre_days": 5,
        "post_days": 1,
        "predict_days": 364,
        "max_positions": 5,
        "idle": None,
        "dates_key": "earnings_dates",
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx, syms = data.index, data.symbols
        col = {s: j for j, s in enumerate(syms)}
        dates = {s: d for s, d in data.extra[self.p["dates_key"]].items() if s in col}
        win = announcement_windows(dates, idx, self.p["pre_days"], self.p["post_days"], self.p["predict_days"])
        starts = {i: g for i, g in win.sort_values("predicted").groupby("entry")}

        k = self.p["max_positions"]
        slot = 1.0 / k
        weights = np.zeros((len(idx), len(syms)))
        held: dict[int, int] = {}  # column -> exit position
        self.taken = []
        for i in range(len(idx)):
            for j in [j for j, out in held.items() if i >= out]:
                del held[j]
            # Earliest predicted dates first when more names start than fit.
            for e in starts.get(i, pd.DataFrame()).itertuples():
                j = col[e.symbol]
                if len(held) >= k or j in held:
                    continue
                held[j] = e.exit
                self.taken.append((idx[i], e.symbol, e.predicted))
            for j in held:
                weights[i, j] = slot
        out = pd.DataFrame(weights, index=idx, columns=syms)
        if self.p["idle"]:
            out[self.p["idle"]] = 1.0 - out.drop(columns=self.p["idle"]).sum(axis=1)
        self.taken = pd.DataFrame(self.taken, columns=["entry_day", "symbol", "predicted"])
        return out
