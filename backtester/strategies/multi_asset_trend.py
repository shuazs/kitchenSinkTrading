"""Multi-asset trend following, the core of managed futures (CTA) funds.

Research basis:
- Moskowitz, Ooi and Pedersen (2012), "Time Series Momentum": each asset's
  own past 12-month return predicts its next month, across 58 futures.
- Hurst, Ooi and Pedersen (2017), "A Century of Evidence on Trend-Following
  Investing": positive in every decade since 1880, strongest in big equity
  sell-offs ("crisis alpha").
- Baltas and Kosowski (2013) and AQR practice: blend several lookbacks and
  size each position by inverse volatility, then scale the book to a target
  volatility.

Retail version: ETFs stand in for futures. Each week, every asset's signal is
the average sign of its returns over `lookbacks` trading days (so it runs
from -1 to +1), its weight is signal x (asset_vol / its own volatility) / N,
and the whole book is scaled so its forecast volatility (from a trailing
covariance of weekly returns) equals `target_vol`, capped at `max_gross`.

A small cash account cannot short futures. Negative weights go to the
inverse ETF listed in `inverse` (SH for SPY, TBF for TLT, ...) when it has a
price that day; otherwise they become cash unless `short="allow"` (then the
engine needs allow_short=True and a margin account).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy

TREND_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",  # equities
    "TLT", "IEF",  # Treasuries
    "GLD", "SLV", "DBC", "USO",  # commodities
    "UUP", "FXE", "FXY",  # currencies
    "BTC-USD",
]
INVERSE = {"SPY": "SH", "QQQ": "PSQ", "IWM": "RWM", "EFA": "EFZ", "EEM": "EUM", "TLT": "TBF", "IEF": "TBX", "GLD": "GLL", "UUP": "UDN"}


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last date of each calendar week present in the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W")).last().values)


def weekly_closes(data: MarketData, symbols) -> pd.DataFrame:
    """Close on the last bar of each week, carried forward over gaps so ETFs
    and 24/7 crypto line up on the same weekly grid."""
    close = data.field("close")[list(symbols)].ffill()
    return close.loc[week_ends(data.index)]


def scale_to_vol(raw: pd.DataFrame, weekly_ret: pd.DataFrame, target_vol: float, window: int, max_gross: float) -> pd.DataFrame:
    """Scale each row of weights so the book's forecast volatility, from the
    trailing `window`-week covariance, equals target_vol; cap gross exposure."""
    out = raw.copy() * np.nan
    for i in range(len(raw)):
        if i < window:
            continue
        w = raw.iloc[i].fillna(0.0)
        if not w.any():
            out.iloc[i] = 0.0
            continue
        hist = weekly_ret.iloc[i - window + 1 : i + 1]
        live = w.index[w != 0]
        cov = hist[live].fillna(0.0).cov().to_numpy() * 52
        vol = float(np.sqrt(max(w[live].to_numpy() @ cov @ w[live].to_numpy(), 0.0)))
        k = target_vol / vol if vol > 0 else 0.0
        gross = w.abs().sum() * k
        if gross > max_gross:
            k *= max_gross / gross
        out.iloc[i] = w * k
    return out


class MultiAssetTrend(Strategy):
    """Weekly vol-scaled time-series momentum across an ETF (and crypto) basket.

    universe: symbols to trade (default TREND_UNIVERSE, whatever is loaded).
    lookbacks: trend windows in trading days; the signal is their mean sign.
    asset_vol / vol_weeks: each position is sized to asset_vol / its own
        volatility over vol_weeks weekly returns, divided by the number of
        assets with a price.
    target_vol / cov_weeks: whole-book volatility target from a cov_weeks
        weekly covariance; set target_vol=None to skip book-level scaling.
    max_gross: cap on the sum of absolute weights.
    short: "inverse" (use the `inverse` ETF map, else cash), "cash" or "allow".
    band: skip a week's rebalance unless some weight moved by at least this.
    """

    name = "multi_asset_trend"
    params = {
        "universe": None,
        "lookbacks": (21, 63, 126, 252),
        "asset_vol": 0.10,
        "vol_weeks": 26,
        "target_vol": 0.15,
        "cov_weeks": 52,
        "max_gross": 1.5,
        "short": "inverse",
        "inverse": INVERSE,
        "band": 0.05,
    }

    def universe(self, data: MarketData) -> list[str]:
        inv = set(self.p["inverse"].values()) if self.p["short"] == "inverse" else set()
        uni = self.p["universe"] or TREND_UNIVERSE
        return [s for s in uni if s in data.bars and s not in inv]

    def signal(self, data: MarketData, uni: list[str]) -> pd.DataFrame:
        """Mean sign of trailing returns, measured on each symbol's own bars
        (so a 252-day lookback is a year of trading for ETFs and crypto alike)."""
        sig = {}
        for s in uni:
            c = data.bars[s]["close"].dropna()
            # Crypto trades every day, so convert trading-day lookbacks to its calendar.
            per_year = 365 if (c.index.dayofweek >= 5).mean() > 0.1 else 252
            votes = [np.sign(c.pct_change(round(n * per_year / 252), fill_method=None)) for n in self.p["lookbacks"]]
            sig[s] = (sum(votes) / len(votes)).reindex(data.index).ffill()
        return pd.DataFrame(sig).loc[week_ends(data.index)]

    def raw_weights(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        uni = self.universe(data)
        wk = weekly_closes(data, uni)
        ret = wk.pct_change(fill_method=None)
        vol = ret.rolling(self.p["vol_weeks"], min_periods=self.p["vol_weeks"] // 2).std() * np.sqrt(52)
        sig = self.signal(data, uni)
        n = vol.notna().sum(axis=1).clip(lower=1)
        raw = (sig * self.p["asset_vol"] / vol).div(n, axis=0)
        return raw, ret

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        raw, ret = self.raw_weights(data)
        if self.p["target_vol"]:
            w = scale_to_vol(raw, ret, self.p["target_vol"], self.p["cov_weeks"], self.p["max_gross"])
        else:
            gross = raw.abs().sum(axis=1)
            w = raw.mul((self.p["max_gross"] / gross).clip(upper=1.0), axis=0)
        w = w.dropna(how="all")
        out = self.map_shorts(w.fillna(0.0), data)
        return _apply_band(out, self.p["band"])

    def map_shorts(self, w: pd.DataFrame, data: MarketData) -> pd.DataFrame:
        out = pd.DataFrame(0.0, index=w.index, columns=data.symbols)
        out[w.columns] = w.clip(lower=0) if self.p["short"] != "allow" else w
        if self.p["short"] == "inverse":
            close = data.field("close").reindex(w.index)
            for s, inv in self.p["inverse"].items():
                if s in w and inv in data.bars:
                    short = (-w[s]).clip(lower=0)
                    out[inv] += short.where(close[inv].notna(), 0.0)
        return out


def _apply_band(w: pd.DataFrame, band: float) -> pd.DataFrame:
    """Keep the previous row unless some weight moved by at least `band`."""
    if not band:
        return w
    rows = [w.iloc[0]]
    for i in range(1, len(w)):
        cur = w.iloc[i]
        rows.append(cur if (cur - rows[-1]).abs().max() >= band else rows[-1])
    return pd.DataFrame(rows, index=w.index)
