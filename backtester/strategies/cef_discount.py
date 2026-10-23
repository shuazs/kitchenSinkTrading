import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


class CEFDiscount(Strategy):
    """Closed-end fund discount reversion.

    Discount = price / NAV - 1, with NAV in `data.extra[nav_key]` (date x
    fund frame, e.g. Yahoo's XADEX for ADX). Buy a fund when its discount z
    score versus its own `lookback`-day history drops below `-entry_z`, sell
    when z recovers to `exit_z` or after `max_hold` days. Up to `max_positions`
    equal slots of 1/max_positions each; the widest discounts fill first.
    Discount jumps above `max_jump` in a day are treated as bad NAV prints.
    """

    name = "cef_discount"
    params = {
        "nav_key": "nav",
        "lookback": 252,
        "entry_z": 1.5,
        "exit_z": 0.5,
        "max_hold": 126,
        "max_positions": 10,
        "max_jump": 0.15,
        "min_dollar_volume": 50_000,
    }

    def zscores(self, data: MarketData) -> pd.DataFrame:
        close = data.field("close")
        nav = data.extra[self.p["nav_key"]].reindex(index=close.index, columns=close.columns)
        disc = close / nav - 1
        disc = disc.where((disc - disc.shift()).abs() < self.p["max_jump"])
        lb = self.p["lookback"]
        mean = disc.rolling(lb, min_periods=lb // 2).mean()
        std = disc.rolling(lb, min_periods=lb // 2).std()
        return (disc - mean) / std

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        z = self.zscores(data).to_numpy(float)
        dollar_vol = (data.field("close") * data.field("volume")).rolling(20).median()
        liquid = (dollar_vol > self.p["min_dollar_volume"]).to_numpy()
        n, held = self.p["max_positions"], {}
        out = np.zeros_like(z)
        for i in range(len(z)):
            for j in list(held):
                held[j] += 1
                if np.isnan(z[i, j]):
                    if held[j] > 5:
                        del held[j]
                elif z[i, j] >= self.p["exit_z"] or held[j] >= self.p["max_hold"]:
                    del held[j]
            if len(held) < n:
                cand = [(z[i, j], j) for j in np.where((z[i] < -self.p["entry_z"]) & liquid[i])[0] if j not in held]
                for _, j in sorted(cand)[: n - len(held)]:
                    held[j] = 0
            for j in held:
                out[i, j] = 1.0 / n
        return pd.DataFrame(out, index=data.index, columns=data.symbols)
