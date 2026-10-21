"""Crypto strategies: cross-sectional momentum, a volatility-targeted
breakout, and a blend that caps a risky sleeve (e.g. meme coins).

Crypto trades every day, so volatility is annualized with 365 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .breakout import Breakout

CRYPTO_DAYS = 365


class CrossSectionalMomentum(Strategy):
    """Every `rebalance_days`, hold the `top_n` coins with the highest
    `lookback`-day return, 1/top_n each; the rest stays in cash.

    Short-horizon (1-4 week) momentum is the strongest cross-sectional
    effect in crypto, with reversal at longer horizons (Liu, Tsyvinski and
    Wu 2022, "Common Risk Factors in Cryptocurrency"). `abs_momentum` only
    holds coins whose own return is positive. With `trend_symbol`, the whole
    book goes to cash while that coin closes below its `trend_ma`-day
    average. A coin needs `lookback` days of history before it can be picked.
    """

    name = "xs_momentum"
    params = {
        "universe": None,
        "top_n": 3,
        "lookback": 21,
        "rebalance_days": 7,
        "abs_momentum": True,
        "trend_symbol": "BTC-USD",
        "trend_ma": 140,
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        uni = self.p["universe"] or data.symbols
        close = data.field("close").ffill()
        mom = close[uni].pct_change(self.p["lookback"], fill_method=None)
        risk_on = pd.Series(True, index=close.index)
        if self.p["trend_symbol"]:
            sig = close[self.p["trend_symbol"]]
            risk_on = sig > sig.rolling(self.p["trend_ma"]).mean()

        rebal = close.index[:: self.p["rebalance_days"]]
        w = pd.DataFrame(0.0, index=rebal, columns=data.symbols)
        slot = 1.0 / self.p["top_n"]
        for d in rebal:
            if not risk_on.loc[d]:
                continue
            m = mom.loc[d].dropna()
            if self.p["abs_momentum"]:
                m = m[m > 0]
            w.loc[d, m.nlargest(self.p["top_n"]).index] = slot
        return w


class VolTargetBreakout(Breakout):
    """The Donchian breakout, but each open position is sized to
    `target_vol` annualized volatility from the last `vol_lookback` days,
    capped at its equal-weight slot times `max_weight`.

    Volatility scaling mostly cuts drawdowns rather than raising returns
    (Moreira and Muir 2017; Cederburg et al. 2020 on out-of-sample decay).
    A size only changes when it moves by more than `band` (as a fraction of
    the current size), so the engine does not pay fees to chase small
    volatility wiggles.
    """

    name = "vol_target_breakout"
    params = {**Breakout.params, "target_vol": 0.5, "vol_lookback": 30, "max_weight": 1.0, "band": 0.25}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        base = super().generate_targets(data)
        close = data.field("close").ffill()
        vol = close.pct_change(fill_method=None).rolling(self.p["vol_lookback"]).std() * np.sqrt(CRYPTO_DAYS)
        scale = (self.p["target_vol"] / vol).clip(upper=self.p["max_weight"]).fillna(0.0)
        raw = (base * scale).to_numpy(float)

        out = np.zeros_like(raw)
        cur = np.zeros(raw.shape[1])
        for i, row in enumerate(raw):
            for j, x in enumerate(row):
                opened_or_closed = (x == 0) != (cur[j] == 0)
                if opened_or_closed or (cur[j] and abs(x / cur[j] - 1) > self.p["band"]):
                    cur[j] = x
            out[i] = cur
        return pd.DataFrame(out, index=base.index, columns=base.columns)


class Sleeves(Strategy):
    """Run several strategies side by side, each on its own symbols with a
    fixed share of capital: `sleeves` is a list of (strategy, symbols, weight).

    Used for a core-plus-lottery book, e.g. 90% in a BTC+ETH breakout and
    10% in a meme coin breakout. Weights are fixed at each sleeve's
    rebalance, so a winning sleeve can drift above its share between trades.
    """

    name = "sleeves"
    params = {"sleeves": ()}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        parts = []
        for strat, symbols, weight in self.p["sleeves"]:
            sub = MarketData({s: data.bars[s] for s in symbols}, data.extra)
            t = strat.generate_targets(sub).reindex(data.index).ffill().fillna(0.0)
            parts.append(t * weight)
        return pd.concat(parts, axis=1).T.groupby(level=0).sum().T.reindex(columns=data.symbols, fill_value=0.0)
