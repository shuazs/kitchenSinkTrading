import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends


class DualMomentum(Strategy):
    """Monthly: hold the top `top_n` of `universe` by `lookback`-day return,
    but only those whose return beats `safe` (absolute momentum); the rest
    of the capital goes to `safe` (e.g. SHY or BIL).

    Antonacci's Global Equities Momentum is universe=["SPY","EFA"],
    safe="AGG", top_n=1, lookback=252.
    """

    name = "dual_momentum"
    params = {"universe": None, "safe": "SHY", "top_n": 1, "lookback": 252}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        uni = self.p["universe"] or [s for s in data.symbols if s != self.p["safe"]]
        close = data.field("close").ffill()
        mom = close.pct_change(self.p["lookback"], fill_method=None)
        rebal = month_ends(data.index)
        w = pd.DataFrame(0.0, index=rebal, columns=data.symbols)
        for d in rebal:
            m = mom.loc[d]
            if m[uni].isna().all():
                continue
            hurdle = m.get(self.p["safe"], 0.0)
            hurdle = 0.0 if pd.isna(hurdle) else hurdle
            picks = m[uni].dropna().nlargest(self.p["top_n"])
            picks = picks[picks > hurdle]
            slot = 1.0 / self.p["top_n"]
            for s in picks.index:
                w.loc[d, s] = slot
            w.loc[d, self.p["safe"]] = 1.0 - slot * len(picks)
        return w
