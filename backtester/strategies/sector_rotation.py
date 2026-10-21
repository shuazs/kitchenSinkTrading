import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends

SECTOR_SPDRS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


class SectorRotation(Strategy):
    """Monthly: hold the `top_n` sectors by momentum, equally weighted.

    Industry momentum (Moskowitz and Grinblatt 1999). Momentum is the average
    of the `lookbacks` returns, which is less sensitive to one window than a
    single lookback. Only sectors with positive momentum are held; their
    unused slots go to `safe` (or cash). With `market` and `ma` set, the whole
    book goes to `safe` while `market` is below its `ma`-day average.
    Sectors without enough history (XLRE from 2015, XLC from 2018) are
    skipped until they have it.
    """

    name = "sector_rotation"
    params = {"universe": None, "safe": None, "top_n": 3, "lookbacks": (63, 126, 252), "market": None, "ma": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        uni = self.p["universe"] or [s for s in SECTOR_SPDRS if s in data.bars]
        close = data.field("close").ffill()
        mom = sum(close[uni].pct_change(n, fill_method=None) for n in self.p["lookbacks"]) / len(self.p["lookbacks"])
        risk_on = pd.Series(True, index=data.index)
        if self.p["market"] and self.p["ma"]:
            mkt = close[self.p["market"]]
            risk_on = mkt > mkt.rolling(self.p["ma"]).mean()

        rebal = month_ends(data.index)
        w = pd.DataFrame(0.0, index=rebal, columns=data.symbols)
        slot = 1.0 / self.p["top_n"]
        for d in rebal:
            m = mom.loc[d].dropna()
            if len(m) < self.p["top_n"]:
                continue
            picks = m.nlargest(self.p["top_n"])
            picks = picks[picks > 0] if risk_on.loc[d] else picks.iloc[:0]
            for s in picks.index:
                w.loc[d, s] = slot
            if self.p["safe"]:
                w.loc[d, self.p["safe"]] = 1.0 - slot * len(picks)
        return w
