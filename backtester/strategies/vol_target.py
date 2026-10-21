import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


class VolTarget(Strategy):
    """Scale exposure to `symbol` so its expected volatility is `target_vol`.

    Moreira and Muir (2017), "Volatility-Managed Portfolios": cut exposure
    when recent volatility is high, since it predicts volatility but not
    returns. Cederburg et al. (2020) find the gain often fails out of sample,
    so treat results with care.

    Volatility is measured on `signal` (e.g. SPY for UPRO) and multiplied by
    `leverage` (the traded ETF's multiple), which avoids the levered ETF's
    shorter history. `band` skips rebalances smaller than that change in
    weight. Whatever is not in `symbol` sits in `risk_off` (or cash).
    """

    name = "vol_target"
    params = {
        "symbol": "UPRO",
        "signal": None,
        "leverage": 1.0,
        "target_vol": 0.20,
        "lookback": 20,
        "max_weight": 1.0,
        "band": 0.10,
        "risk_off": None,
    }

    def target_weight(self, data: MarketData) -> pd.Series:
        close = data.bars[self.p["signal"] or self.p["symbol"]]["close"].reindex(data.index).ffill()
        vol = close.pct_change(fill_method=None).rolling(self.p["lookback"]).std() * np.sqrt(252)
        raw = self.p["target_vol"] / (vol * self.p["leverage"])
        return raw.clip(upper=self.p["max_weight"])

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        raw = self.target_weight(data).to_numpy(float)
        held = np.full(len(raw), np.nan)
        cur = np.nan
        for i, x in enumerate(raw):
            if np.isnan(x):
                continue
            if np.isnan(cur) or abs(x - cur) >= self.p["band"]:
                cur = x
            held[i] = cur

        w = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        weight = pd.Series(held, index=data.index).fillna(0.0)
        w[self.p["symbol"]] = weight
        if self.p["risk_off"]:
            w[self.p["risk_off"]] = 1.0 - weight
        return w
