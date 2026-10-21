import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends


class FXCarry(Strategy):
    """Monthly: long the highest-yielding currencies, short the lowest, vs USD.

    The G10 carry trade (Lustig, Roussanov and Verdelhan 2011; Koijen et al.
    2018, "Carry"). Ranks currencies by their interest-rate gap to USD as
    known at the rebalance (`extra['rate_gap']` from `data.fx.load_fx_market`)
    and holds `n_long` / `n_short` legs, each side weighted by inverse
    volatility. With `trend` set, a leg is only held while its own
    `trend`-day total return agrees with it (carry plus trend, which
    sidesteps some carry crashes). The book is then scaled so its ex-ante
    volatility is `target_vol`, capped at `max_gross` times equity.

    Run it with `Backtester(allow_short=True, max_leverage=max_gross,
    cash_rate=<USD rate>)` so shorts and leverage are financed at the USD rate.
    """

    name = "fx_carry"
    params = {
        "n_long": 2,
        "n_short": 2,
        "min_gap": 0.0,
        "trend": None,
        "vol_lookback": 63,
        "target_vol": 0.10,
        "max_gross": 3.0,
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        gap = data.extra["rate_gap"].reindex(data.index)
        close = data.field("close").ffill()
        rets = close.pct_change(fill_method=None)
        lb = self.p["vol_lookback"]
        trend = close.pct_change(self.p["trend"], fill_method=None) if self.p["trend"] else None

        w = pd.DataFrame(0.0, index=month_ends(data.index), columns=data.symbols)
        for d in w.index:
            g = gap.loc[d].dropna()
            window = rets.loc[:d].iloc[-lb:]
            vol = window.std() * np.sqrt(252)
            g = g[vol.reindex(g.index).gt(0) & window.count().reindex(g.index).ge(lb // 2)]
            if len(g) < self.p["n_long"] + self.p["n_short"]:
                continue
            ranked = g.sort_values()
            longs = ranked.index[-self.p["n_long"]:] if self.p["n_long"] else []
            shorts = ranked.index[: self.p["n_short"]] if self.p["n_short"] else []
            # Only trade a real rate gap: a long must out-yield the shorts (or USD).
            short_rate = ranked[shorts].mean() if len(shorts) else 0.0
            longs = [c for c in longs if ranked[c] - short_rate > self.p["min_gap"]]
            shorts = [c for c in shorts if ranked[longs].mean() - ranked[c] > self.p["min_gap"]] if longs else []
            if trend is not None:
                t = trend.loc[d]
                longs = [c for c in longs if t[c] > 0]
                shorts = [c for c in shorts if t[c] < 0]
            row = pd.Series(0.0, index=data.symbols)
            for legs, sign in ((longs, 1.0), (shorts, -1.0)):
                if len(legs):
                    inv = 1 / vol[legs]
                    row[legs] = sign * inv / inv.sum()
            if not row.any():
                w.loc[d] = row
                continue
            cov = window[data.symbols].fillna(0).cov() * 252
            book_vol = float(np.sqrt(row @ cov @ row))
            scale = self.p["target_vol"] / book_vol if book_vol > 0 else 0.0
            scale = min(scale, self.p["max_gross"] / row.abs().sum())
            w.loc[d] = row * scale
        return w
