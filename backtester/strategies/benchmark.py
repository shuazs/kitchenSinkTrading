import pandas as pd

from ..core import MarketData, Strategy


class BuyAndHold(Strategy):
    """Equal-weight buy and hold of all symbols (or the listed ones)."""

    name = "buy_and_hold"
    params = {"symbols": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        syms = self.p["symbols"] or data.symbols
        w = pd.DataFrame(0.0, index=data.index[:1], columns=data.symbols)
        w[syms] = 1.0 / len(syms)
        return w
