import pandas as pd

from ..core import MarketData, Strategy
from .indicators import rsi


class RSIMeanReversion(Strategy):
    """Buy short-term pullbacks inside a long-term uptrend.

    Enter when RSI(`rsi_n`) < `entry` and close > `trend_ma`-day average.
    Exit when RSI > `exit` or after `max_hold` days. Equal weight across
    open positions. Popularised by Larry Connors; tested on ETFs by
    Micaletti (2023) among other short-term reversal indicators.

    event_key / event_mode optionally gate entries on a boolean date Series
    in `MarketData.extra`: "require" enters only on event days, "exclude"
    only on other days (e.g. days he posted about tariffs).
    """

    name = "rsi_mean_reversion"
    params = {"rsi_n": 2, "entry": 10, "exit": 70, "trend_ma": 200, "max_hold": 10, "event_key": None, "event_mode": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        close = data.field("close").ffill()
        r = rsi(close, self.p["rsi_n"])
        trend = close > close.rolling(self.p["trend_ma"]).mean()
        enter = (r < self.p["entry"]) & trend
        if self.p["event_mode"]:
            ev = data.extra[self.p["event_key"]].reindex(close.index).fillna(False).astype(bool)
            ev = ev if self.p["event_mode"] == "require" else ~ev
            enter = enter.mul(ev, axis=0).astype(bool)
        leave = r > self.p["exit"]

        held = pd.DataFrame(False, index=close.index, columns=close.columns)
        for s in close.columns:
            on, age = False, 0
            col = []
            for e, x in zip(enter[s].values, leave[s].values):
                if on:
                    age += 1
                    if x or age >= self.p["max_hold"]:
                        on, age = False, 0
                elif e:
                    on, age = True, 0
                col.append(on)
            held[s] = col
        n = held.sum(axis=1).clip(lower=1)
        return held.astype(float).div(n, axis=0)
