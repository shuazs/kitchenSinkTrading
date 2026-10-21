"""Same-calendar-month return seasonality.

Heston and Sadka (2008), "Seasonality in the Cross-Section of Stock
Returns", and Keloharju, Linnainmaa and Nyberg (2016), "Return
Seasonalities": assets that did well in a given calendar month in past
years tend to do well in that month again. KLN find it in stocks (about
13%/yr long-short), anomalies, commodities and country stock indices, so it
can be traded with sector and country ETFs.
"""
from __future__ import annotations

import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends

COUNTRY_ETFS = ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWN", "EWP", "EWQ", "EWS", "EWU",
                "EWW", "EWY", "EWZ", "EWT"]


class SameMonthSeasonality(Strategy):
    """At each month end, hold the `top_n` assets with the highest average
    return in the coming calendar month over the past `years` years.

    An asset needs `min_years` past observations of that month. With
    `demean`, the score is the same-month average minus the asset's average
    over all months in the same years, so a strong asset is not picked every
    month just for being strong. Only positive scores are held; empty slots
    go to `safe` (or cash).
    """

    name = "seasonality"
    params = {"universe": None, "top_n": 3, "years": 20, "min_years": 5, "demean": True, "safe": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        uni = self.p["universe"] or [s for s in data.symbols if s != self.p["safe"]]
        close = data.field("close").ffill()[uni]
        ends = month_ends(data.index)
        monthly = close.loc[ends].pct_change(fill_method=None)
        monthly.index = ends.to_period("M")

        w = pd.DataFrame(0.0, index=ends, columns=data.symbols)
        slot = 1.0 / self.p["top_n"]
        for d in ends:
            nxt = d.to_period("M") + 1
            past = monthly.loc[: d.to_period("M")].iloc[-12 * self.p["years"]:]
            same = past[past.index.month == nxt.month]
            score = same.mean().where(same.count() >= self.p["min_years"])
            if self.p["demean"]:
                score = score - past.mean()
            picks = score.dropna().nlargest(self.p["top_n"])
            picks = picks[picks > 0]
            w.loc[d, picks.index] = slot
            if self.p["safe"]:
                w.loc[d, self.p["safe"]] = 1.0 - slot * len(picks)
        return w
