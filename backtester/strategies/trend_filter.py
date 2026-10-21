import pandas as pd

from ..core import MarketData, Strategy


class TrendFilter(Strategy):
    """Hold `risk_on` while its signal asset closes above its moving average,
    else hold `risk_off` (or cash if None).

    Gayed and Bilello (2016), "Leverage for the Long Run": e.g. risk_on="SSO"
    or "TQQQ", signal="SPY" or "QQQ", risk_off="SHY". Using the unlevered
    index as the signal avoids whipsaw from the levered ETF's decay.
    `band` adds hysteresis: exit below ma*(1-band), re-enter above ma*(1+band).
    """

    name = "trend_filter"
    params = {"risk_on": "SSO", "signal": None, "risk_off": None, "ma": 200, "band": 0.0}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        sig_sym = self.p["signal"] or self.p["risk_on"]
        close = data.bars[sig_sym]["close"].reindex(data.index).ffill()
        ma = close.rolling(self.p["ma"]).mean()
        up = close > ma * (1 + self.p["band"])
        down = close < ma * (1 - self.p["band"])
        state = pd.Series(pd.NA, index=data.index, dtype="boolean")
        state[up] = True
        state[down] = False
        state = state.ffill().fillna(False).astype(bool)
        state[ma.isna()] = False

        w = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        w[self.p["risk_on"]] = state.astype(float)
        if self.p["risk_off"]:
            w[self.p["risk_off"]] = (~state).astype(float)
        return w
