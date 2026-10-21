import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .multi_asset_trend import scale_to_vol, weekly_closes


class SignalBlend(Strategy):
    """Trade weekly positions from a signal-combination model (backtester.ml).

    positions: date x symbol frame of model scores (unit-scaled; clipped to
        [-1, 1] here for risk_parity), one row per week end, made out of
        sample (see ml.contest.out_of_sample_positions).
    mode: "risk_parity" sizes each asset to asset_vol / its weekly vol over
        vol_weeks, divided by the number of assets with a position, then
        scales the book to target_vol from a cov_weeks covariance, capped at
        max_gross (the MultiAssetTrend recipe). "top_n" holds the top_n
        symbols with a positive position in equal weight (stock picking).
    long_only: clip positions below zero to cash.
    """

    name = "signal_blend"
    params = {
        "positions": None,
        "mode": "risk_parity",
        "long_only": True,
        "asset_vol": 0.10,
        "vol_weeks": 26,
        "target_vol": 0.20,
        "cov_weeks": 52,
        "max_gross": 1.5,
        "top_n": 10,
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        pos = self.p["positions"]
        syms = [s for s in pos.columns if s in data.bars]
        pos = pos[syms]
        if self.p["mode"] == "top_n":
            ok = pos > 0 if self.p["long_only"] else pos.notna()
            rank = pos.where(ok).rank(axis=1, ascending=False, method="first")
            w = (rank <= self.p["top_n"]).astype(float) / self.p["top_n"]
            return w.reindex(columns=data.symbols, fill_value=0.0)
        pos = pos.clip(lower=0 if self.p["long_only"] else -1, upper=1)
        wk = weekly_closes(data, syms)
        ret = wk.pct_change(fill_method=None)
        vol = ret.rolling(self.p["vol_weeks"], min_periods=self.p["vol_weeks"] // 2).std() * np.sqrt(52)
        # Size on the full weekly grid so the covariance has history from the
        # first week with a position, then keep only the weeks with positions.
        first = pos.dropna(how="all").index[0]
        pos = pos.reindex(wk.index).loc[:pos.index[-1]]
        n = pos.notna().sum(axis=1).clip(lower=1)
        raw = (pos * self.p["asset_vol"] / vol.loc[pos.index]).div(n, axis=0).fillna(0.0)
        w = scale_to_vol(raw, ret.loc[pos.index], self.p["target_vol"], self.p["cov_weeks"], self.p["max_gross"])
        return w.loc[first:].fillna(0.0).reindex(columns=data.symbols, fill_value=0.0)
