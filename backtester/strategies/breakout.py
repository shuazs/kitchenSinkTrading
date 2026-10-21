import pandas as pd

from ..core import MarketData, Strategy


class Breakout(Strategy):
    """Buy when the close makes a new `entry`-day high; exit on a new
    `exit`-day low (a Donchian channel).

    Capital is split equally among open positions. With `max_positions`,
    each position gets 1/max_positions and, when more symbols qualify, the
    ones furthest above their breakout level are kept.

    Matches Quantpedia's Bitcoin MAX rule (buy new 10-50 day highs).
    """

    name = "breakout"
    params = {"entry": 20, "exit": 10, "max_positions": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        close = data.field("close").ffill()
        hi = close.rolling(self.p["entry"]).max().shift(1)
        lo = close.rolling(self.p["exit"]).min().shift(1)
        state = pd.DataFrame(pd.NA, index=close.index, columns=close.columns, dtype="boolean")
        state[close > hi] = True
        state[close < lo] = False
        held = state.ffill().fillna(False).astype(bool)

        cap = self.p["max_positions"]
        if cap is None:
            return held.astype(float).div(held.sum(axis=1).clip(lower=1), axis=0)
        strength = (close / lo).where(held)
        keep = held & (strength.rank(axis=1, ascending=False, method="first") <= cap)
        return keep.astype(float) / cap
