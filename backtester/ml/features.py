"""Weekly feature panel: one row per (week end, symbol), one column per signal.

Each feature is a numeric version of a strategy already in the repo, measured
with data up to that week's last close. The target is the return to the next
week's last close divided by the asset's recent volatility, so assets with
different risk sit on one scale and a pooled model can learn from all of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData
from ..strategies.indicators import rsi
from ..strategies.multi_asset_trend import week_ends

# Per-asset features and the strategy each one comes from.
ASSET_FEATURES = {
    "mom_21": "MultiAssetTrend / TrendFilter: 1-month vol-scaled return",
    "mom_63": "3-month vol-scaled return",
    "mom_126": "6-month vol-scaled return",
    "mom_252": "12-month vol-scaled return",
    "macd_8_24": "EnsembleTrend-style fast moving-average crossover (Baz et al. 2015)",
    "macd_16_48": "medium crossover",
    "macd_32_96": "slow crossover",
    "chan_20": "Breakout: position in the 20-day high-low channel, -1 to 1",
    "chan_55": "position in the 55-day channel",
    "xs_rank": "CrossSectionalMomentum / HedgedMomentum: 12-1 month return rank in the universe",
    "vol_20": "VolTarget: log of 20-day annualised volatility",
    "vol_ratio": "20-day over 120-day volatility",
    "rsi_2": "RSIMeanReversion: RSI(2) scaled to -1..1",
    "season": "SameMonthSeasonality: average return of this calendar month in past years",
}
# Market-wide features, the same for every asset on a date.
MARKET_FEATURES = {
    "days_to_month_end": "TurnOfMonth: weekdays left in the month",
    "pension_gap": "PensionRebalance: SPY minus TLT return month to date",
    "vix_ratio": "VixTermStructure: VIX / VIX3M - 1 (positive = backwardation)",
    "vix_change": "log change in VIX over 20 days",
    "spy_trend": "TrendFilter: SPY over its 200-day average, minus 1",
    "curve": "FRED T10Y2Y: 10y minus 2y Treasury spread, %",
}
FEATURES = list(ASSET_FEATURES) + list(MARKET_FEATURES)
# Directions for the equal-weight baseline: +1 when a high value has meant
# higher returns in the strategy it came from, -1 for mean reversion.
SIGNAL_SIGNS = {
    "mom_21": 1, "mom_63": 1, "mom_126": 1, "mom_252": 1,
    "macd_8_24": 1, "macd_16_48": 1, "macd_32_96": 1,
    "chan_20": 1, "chan_55": 1, "xs_rank": 1, "rsi_2": -1, "season": 1,
}


def _per_year(index: pd.DatetimeIndex) -> int:
    return 365 if (index.dayofweek >= 5).mean() > 0.1 else 252


def asset_features(data: MarketData, symbols) -> dict[str, pd.DataFrame]:
    """Daily date x symbol frames of the per-asset features."""
    close = data.field("close")[list(symbols)]
    high = data.field("high")[list(symbols)].fillna(close)
    low = data.field("low")[list(symbols)].fillna(close)
    ppy = _per_year(data.index)
    k = ppy / 252  # crypto lookbacks in calendar days cover the same span
    d = lambda n: max(1, round(n * k))  # noqa: E731
    r = close.pct_change(fill_method=None)
    vol63 = r.rolling(d(63), min_periods=d(42)).std()
    out = {}
    for n in (21, 63, 126, 252):
        out[f"mom_{n}"] = close.pct_change(d(n), fill_method=None) / (vol63 * np.sqrt(d(n)))
    for s, lng in ((8, 24), (16, 48), (32, 96)):
        m = close.ewm(span=d(s), adjust=False).mean() - close.ewm(span=d(lng), adjust=False).mean()
        q = m / close.rolling(d(63)).std()
        out[f"macd_{s}_{lng}"] = q / q.rolling(d(252), min_periods=d(126)).std()
    for n in (20, 55):
        hi, lo = high.rolling(d(n)).max(), low.rolling(d(n)).min()
        out[f"chan_{n}"] = ((close - lo) / (hi - lo) * 2 - 1).where(hi > lo, 0.0)
    past = close.shift(d(21)) / close.shift(d(252)) - 1
    out["xs_rank"] = past.rank(axis=1, pct=True) - 0.5
    vol20 = r.rolling(d(20)).std()
    out["vol_20"] = np.log(vol20 * np.sqrt(ppy))
    out["vol_ratio"] = vol20 / r.rolling(d(120)).std()
    out["rsi_2"] = (rsi(close, 2) - 50) / 50
    out["season"] = same_month_mean(close)
    return out


def same_month_mean(close: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """For each date, the mean return of its calendar month in earlier years only."""
    monthly = close.resample("ME").last().pct_change(fill_method=None)
    out = {}
    for m in range(1, 13):
        sub = monthly[monthly.index.month == m]
        # shift so a month never sees its own year's return
        out[m] = sub.expanding(min_periods=min_years).mean().shift()
    table = pd.concat(out.values()).sort_index()
    key = close.index.to_period("M")
    looked_up = table.set_axis(table.index.to_period("M")).reindex(key)
    return looked_up.set_axis(close.index)


def market_features(data: MarketData, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Market-wide features on `index`. Uses SPY, TLT, ^VIX and ^VIX3M from
    `data.bars` when loaded and FRED's T10Y2Y from `data.extra['curve']`;
    anything missing comes out as NaN (the panel fills those with 0)."""
    def close_of(sym):
        if sym in data.bars:
            return data.bars[sym]["close"].reindex(index.union(data.bars[sym].index)).ffill().reindex(index)
        return pd.Series(np.nan, index=index)

    me = index + pd.offsets.MonthEnd(0)
    out = pd.DataFrame(index=index)
    out["days_to_month_end"] = np.busday_count(
        (index + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
        (me + pd.Timedelta(days=1)).values.astype("datetime64[D]"),
    )
    spy, tlt = close_of("SPY"), close_of("TLT")
    month = index.to_period("M")

    def mtd(s):
        prev = s.groupby(month).last().shift()
        return s / prev.reindex(month).set_axis(index) - 1

    out["pension_gap"] = mtd(spy) - mtd(tlt)
    vix, vix3m = close_of("^VIX"), close_of("^VIX3M")
    out["vix_ratio"] = vix / vix3m - 1
    out["vix_change"] = np.log(vix / vix.shift(20 if _per_year(index) == 252 else 28))
    if "SPY" in data.bars:
        s = data.bars["SPY"]["close"]
        out["spy_trend"] = (s / s.rolling(200).mean() - 1).reindex(index.union(s.index)).ffill().reindex(index)
    else:
        out["spy_trend"] = np.nan
    curve = data.extra.get("curve")
    if curve is not None:
        # FRED posts a day's value after the close; lag one day to be safe.
        c = curve.shift(1, freq="D")
        out["curve"] = c.reindex(index.union(c.index)).ffill().reindex(index)
    else:
        out["curve"] = np.nan
    return out


def build_panel(data: MarketData, symbols, clip_target: float = 3.0) -> pd.DataFrame:
    """Weekly (date, symbol) panel of FEATURES plus `ret` (raw next-week
    return) and `target` (ret over trailing weekly vol, clipped).

    Rows exist once a symbol has every per-asset feature; the final week's
    `ret` and `target` are NaN because that week has not happened yet.
    """
    symbols = [s for s in symbols if s in data.bars]
    daily = asset_features(data, symbols)
    weeks = week_ends(data.index)
    close = data.field("close")[symbols].ffill().loc[weeks]
    fwd = close.shift(-1) / close - 1
    # Weekly vol from daily vol so the scale is known at the decision date.
    rows_per_week = 7 if _per_year(data.index) == 365 else 5
    r = data.field("close")[symbols].pct_change(fill_method=None)
    wvol = (r.rolling(rows_per_week * 4).std() * np.sqrt(rows_per_week)).ffill().loc[weeks]
    frames = {name: f.ffill(limit=3).loc[weeks] for name, f in daily.items()}
    frames["ret"] = fwd
    frames["target"] = (fwd / wvol).clip(-clip_target, clip_target)
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    panel.index.names = ["date", "symbol"]
    # A symbol must have traded this week (no stale prices after delisting).
    last_trade = data.field("close")[symbols].notna().cumsum().loc[weeks].diff().stack(future_stack=True)
    live = last_trade.reindex(panel.index).fillna(1) > 0
    # Seasonality needs three years of history; until then it is neutral, so
    # young assets (most coins) are not held out of the panel for it.
    panel["season"] = panel["season"].fillna(0.0)
    panel = panel[live].dropna(subset=list(ASSET_FEATURES))
    mkt = market_features(data, weeks)
    panel = panel.join(mkt, on="date")
    panel[list(MARKET_FEATURES)] = panel[list(MARKET_FEATURES)].fillna(0.0)
    return panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
