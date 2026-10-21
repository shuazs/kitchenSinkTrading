import pandas as pd

from ..core import MarketData, Strategy


class VixTermStructure(Strategy):
    """Hold a short-volatility ETP while VIX futures are in contango, else
    `risk_off` (or cash).

    Short VIX futures earn the roll yield when the curve slopes up (Simon and
    Campasano 2014, "The VIX Futures Basis"). Spot VIX over 3-month VIX is
    the free proxy for the curve: below `threshold` means contango. The risk
    is a one-day gap, as on 5 Feb 2018 when the -1x products lost over 80%.
    Use SVXY's real price history, which is -1x before 2018-02-28 and -0.5x
    after, rather than a modeled series.

    smooth: days to average the ratio over before comparing, to cut whipsaw.
    """

    name = "vix_term_structure"
    params = {"symbol": "SVXY", "vix": "^VIX", "vix3m": "^VIX3M", "threshold": 1.0, "smooth": 1, "risk_off": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        vix = data.bars[self.p["vix"]]["close"].reindex(idx).ffill()
        vix3m = data.bars[self.p["vix3m"]]["close"].reindex(idx).ffill()
        ratio = (vix / vix3m).rolling(self.p["smooth"]).mean()
        on = (ratio < self.p["threshold"]).fillna(False)

        w = pd.DataFrame(0.0, index=idx, columns=data.symbols)
        w[self.p["symbol"]] = on.astype(float)
        if self.p["risk_off"]:
            w[self.p["risk_off"]] = (~on).astype(float)
        return w
