import pandas as pd

from ..core import MarketData, Strategy


class CotPositioning(Strategy):
    """Weekly: rank assets on a Commitments of Traders signal, hold the best.

    `data.extra[signal_key]` is a date x symbol frame already placed on the
    trading day each report became public (`data.cot.to_trading_days`), with
    higher = more bullish (`data.cot.positioning_signals`). On each report
    day the strategy holds the `top_n` highest-ranked symbols equally
    weighted; with `short=True` it also shorts the `top_n` lowest, each side
    at half of `gross`. Symbols without a price that day are skipped.
    """

    name = "cot_positioning"
    params = {"signal_key": "cot_signal", "universe": None, "top_n": 3, "short": False, "gross": 1.0}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        sig = data.extra[self.p["signal_key"]]
        uni = [s for s in (self.p["universe"] or sig.columns) if s in data.bars]
        close = data.field("close")[uni]
        sig = sig.reindex(columns=uni)
        sig = sig[sig.index.isin(close.index)]
        n = self.p["top_n"]
        side = self.p["gross"] / 2 if self.p["short"] else self.p["gross"]
        w = pd.DataFrame(0.0, index=sig.index, columns=data.symbols)
        for d, row in sig.iterrows():
            x = row[close.loc[d].notna()].dropna()
            if len(x) < (2 * n if self.p["short"] else n):
                w.loc[d] = float("nan")
                continue
            w.loc[d, x.nlargest(n).index] = side / n
            if self.p["short"]:
                w.loc[d, x.nsmallest(n).index] = -side / n
        return w.dropna(how="all")
