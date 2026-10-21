"""Leveraged risk parity, the idea behind Bridgewater's All Weather and AQR's
risk parity funds.

Research basis:
- Qian (2005), "Risk Parity Portfolios", and Asness, Frazzini and Pedersen
  (2012), "Leverage Aversion and Risk Parity": weighting assets so each adds
  the same risk, then levering the whole book, has beaten 60/40 because
  investors who cannot borrow overpay for risky assets.
- 2022 is the known failure: stocks and bonds fell together while rates rose,
  and levered bonds lost the most.

Retail version: monthly, weight each asset by 1 / its volatility (naive risk
parity, robust to bad correlation estimates), then scale the book so its
forecast volatility equals `target_vol`, capped at `max_gross`. Leverage
comes from 2x/3x ETFs (UPRO, TMF, UGL), so a cash account can hold it; set
`leverage` per symbol so volatility is measured on the unlevered `signal`
ETF, which has a longer history. An optional trend filter moves an asset
to cash while its signal ETF is below its `trend_ma`-day average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends

# Levered ETF -> (unlevered signal ETF, leverage).
LEVERED = {"UPRO": ("SPY", 3), "TMF": ("TLT", 3), "UGL": ("GLD", 2), "DBC": ("DBC", 1)}


class RiskParity(Strategy):
    """Inverse-volatility weights, scaled to a volatility target, monthly.

    assets: {traded symbol: (signal symbol, leverage)}; default LEVERED.
    vol_days: lookback for each asset's volatility, on daily signal returns.
    cov_days: lookback for the book-level covariance.
    target_vol: book volatility target (None = weights sum to max_gross).
    max_gross: cap on the sum of weights (1.0 = no borrowing).
    trend_ma: optional; assets whose signal is below this moving average
        are moved to `safe` (or cash).
    """

    name = "risk_parity"
    params = {
        "assets": LEVERED,
        "vol_days": 63,
        "cov_days": 252,
        "target_vol": 0.20,
        "max_gross": 1.0,
        "trend_ma": None,
        "safe": None,
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        assets = {s: v for s, v in self.p["assets"].items() if s in data.bars}
        close = data.field("close").ffill()
        sig = pd.DataFrame({s: close[sg] for s, (sg, _) in assets.items()})
        lev = pd.Series({s: L for s, (_, L) in assets.items()}, dtype=float)
        # Returns of the levered position, approximated from the signal ETF.
        ret = sig.pct_change(fill_method=None) * lev
        traded = close[list(assets)]
        vol = ret.rolling(self.p["vol_days"]).std() * np.sqrt(252)
        trend = sig > sig.rolling(self.p["trend_ma"]).mean() if self.p["trend_ma"] else None

        w = pd.DataFrame(0.0, index=month_ends(data.index), columns=data.symbols)
        for d in w.index:
            live = [s for s in assets if pd.notna(vol.at[d, s]) and pd.notna(traded.at[d, s])]
            if not live:
                continue
            inv = 1.0 / vol.loc[d, live]
            raw = inv / inv.sum()
            if self.p["target_vol"]:
                hist = ret.loc[:d, live].iloc[-self.p["cov_days"] :].dropna()
                cov = hist.cov().to_numpy() * 252
                book = float(np.sqrt(raw.to_numpy() @ cov @ raw.to_numpy()))
                raw = raw * (self.p["target_vol"] / book if book > 0 else 0.0)
            if raw.sum() > self.p["max_gross"]:
                raw = raw * self.p["max_gross"] / raw.sum()
            if trend is not None:
                raw = raw.where(trend.loc[d, live], 0.0)
            w.loc[d, live] = raw
            if self.p["safe"]:
                w.loc[d, self.p["safe"]] = max(0.0, 1.0 - raw.sum())
        return w
