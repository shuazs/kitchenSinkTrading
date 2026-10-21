import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .turn_of_month import month_position


class PensionRebalance(Strategy):
    """Trade ahead of pension funds' month-end stock/bond rebalancing.

    Pensions and balanced funds hold fixed stock and bond weights. When
    stocks have beaten bonds during a month, they sell stocks in the last
    days of the month to get back to target, and buy when bonds have led.
    Harvey, Mazzoleni and Melone (2025), "The Unintended Consequences of
    Rebalancing", find this flow predicts lower equity returns over the
    last days of the month after large stock outperformance, and higher
    ones after large bond outperformance.

    Rule: at the close `days` trading days before month end, compute stock
    minus bond return month to date. If the gap is above the `quantile`
    upper cutoff of past months' gaps (only months already finished, so no
    lookahead), hold `when_stocks_lead` until the month ends; if it is
    below the lower cutoff, hold `when_bonds_lead`. Other times hold `base`
    (or cash). With base="SPY" this is an overlay on a buy-and-hold book.
    """

    name = "pension_rebalance"
    params = {
        "stock": "SPY",
        "bond": "TLT",
        "days": 4,
        "quantile": 1 / 3,
        "min_months": 36,
        "base": "SPY",
        "when_stocks_lead": None,
        "when_bonds_lead": "SSO",
    }

    def gaps(self, data: MarketData) -> pd.DataFrame:
        """Month-to-date stock minus bond return on each month's signal day."""
        idx = data.index
        close = data.field("close").ffill()
        _, from_end = month_position(idx)
        # Signal at the close of the day before the window, so the position
        # is on from the open `days` trading days before the month ends.
        sig_days = idx[from_end == self.p["days"] + 1]
        month = idx.to_period("M")
        last = close.groupby(month).last()
        base = last.shift().reindex(month).set_axis(idx)  # prior month's last close
        mtd = close / base - 1
        gap = (mtd[self.p["stock"]] - mtd[self.p["bond"]]).loc[sig_days]
        return pd.DataFrame({"gap": gap})

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        g = self.gaps(data)["gap"].dropna()
        q = self.p["quantile"]
        # Cutoffs from months strictly before each signal day.
        hi = g.expanding(self.p["min_months"]).quantile(1 - q).shift()
        lo = g.expanding(self.p["min_months"]).quantile(q).shift()
        _, from_end = month_position(idx)
        window = pd.Series(from_end <= self.p["days"] + 1, index=idx)
        # Mark state on signal days, carry it through the month's last day.
        state = pd.Series(np.nan, index=idx)
        state.loc[g.index] = np.select([g > hi, g < lo], [1.0, -1.0], 0.0)
        state = state.where(window).groupby(idx.to_period("M")).ffill().where(window).fillna(0.0)
        # The last day of the month's close is the signal to go back to base.
        state = state.where(from_end > 1, 0.0)

        w = pd.DataFrame(0.0, index=idx, columns=data.symbols)
        if self.p["base"]:
            w[self.p["base"]] = 1.0
        for side, sym in ((1.0, self.p["when_stocks_lead"]), (-1.0, self.p["when_bonds_lead"])):
            on = state == side
            if self.p["base"]:
                w.loc[on, self.p["base"]] = 0.0
            if sym:
                w.loc[on, sym] += 1.0
        self.state = state
        return w
