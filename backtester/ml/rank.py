"""Monthly cross-sectional stock ranking with learning-to-rank.

Gu, Kelly and Xiu (2020, "Empirical Asset Pricing via Machine Learning")
found trees and neural nets beat linear models at ranking US stocks by next
month's return, using ~900 firm characteristics on 30,000 stocks since
1957. Poh, Lim, Zohren and Roberts (2021, "Building Cross-Sectional
Systematic Strategies by Learning to Rank") trained LambdaMART on momentum
features and roughly doubled the Sharpe of a plain momentum sort. Most of the
ML gain in both sits in small, illiquid stocks; this asks whether any is left
in the S&P 100 with only the repo's price signals, at a monthly horizon
rather than the contest's weekly one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .features import ASSET_FEATURES, asset_features

RANK_FEATURES = [f for f in ASSET_FEATURES if f != "season"] + ["mom_12_1", "rev_21"]


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().to_numpy())


def monthly_panel(data: MarketData, symbols) -> pd.DataFrame:
    """(date, symbol) rows at each month end: features, `ret` (next month's
    return, from this close to next month's last close) and `grade` (its
    cross-sectional quintile 0-4, the ranker's relevance label)."""
    symbols = [s for s in symbols if s in data.bars]
    feats = asset_features(data, symbols)
    close = data.field("close")[symbols]
    feats["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    feats["rev_21"] = -close.pct_change(21, fill_method=None)
    me = month_ends(data.index)
    c = close.ffill().loc[me]
    fwd = c.shift(-1) / c - 1
    live = close.notna().cumsum().loc[me].diff() > 0
    frames = {k: v.ffill(limit=3).loc[me] for k, v in feats.items() if k in RANK_FEATURES}
    frames["ret"] = fwd.where(live)
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    panel.index.names = ["date", "symbol"]
    panel = panel.astype(float).replace([np.inf, -np.inf], np.nan).dropna(subset=RANK_FEATURES)
    panel = panel[live.stack(future_stack=True).reindex(panel.index).fillna(False).astype(bool)]
    panel["grade"] = panel.groupby(level="date")["ret"].transform(
        lambda r: pd.qcut(r.rank(method="first"), 5, labels=False) if r.notna().sum() >= 5 else np.nan
    )
    return panel


def _zscore_by_date(X: pd.DataFrame) -> pd.DataFrame:
    g = X.groupby(level="date")
    return ((X - g.transform("mean")) / g.transform("std").replace(0, 1)).clip(-3, 3).fillna(0.0)


def walk_forward_scores(panel: pd.DataFrame, model: str, first_test: str, test_end: str, seed: int = 0) -> pd.Series:
    """Out-of-sample monthly scores, refit each calendar year on months whose
    next-month label ended before the year (one month purged).

    model: "momentum" (12-1 month return, the HedgedMomentum sort; no
    fitting), "ridge" (regression on cross-sectionally z-scored features and
    demeaned returns), "lambdarank" (LightGBM LambdaMART on quintile grades)
    or "gbm_reg" (LightGBM regression on demeaned returns).
    """
    X = _zscore_by_date(panel[RANK_FEATURES])
    y = panel["ret"] - panel.groupby(level="date")["ret"].transform("mean")
    dates = panel.index.get_level_values("date")
    out = []
    for year in range(pd.Timestamp(first_test).year, pd.Timestamp(test_end).year + 1):
        lo = pd.Timestamp(f"{year}-01-01")
        hi = min(pd.Timestamp(test_end), pd.Timestamp(f"{year}-12-31"))
        test = (dates >= lo) & (dates <= hi)
        train_dates = np.sort(dates[dates < lo].unique())[:-1]  # purge the month whose label runs into the year
        train = dates.isin(train_dates) & panel["grade"].notna().to_numpy()
        if not test.any():
            continue
        if model == "momentum":
            s = panel.loc[test, "mom_12_1"]
        elif model == "ridge":
            from sklearn.linear_model import Ridge

            m = Ridge(alpha=10.0).fit(X[train], y[train])
            s = pd.Series(m.predict(X[test]), index=X.index[test])
        elif model == "lambdarank":
            from lightgbm import LGBMRanker

            Xt = X[train].sort_index(level="date")
            grp = Xt.groupby(level="date").size().to_numpy()
            m = LGBMRanker(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
            m.fit(Xt, panel["grade"].reindex(Xt.index).astype(int), group=grp)
            s = pd.Series(m.predict(X[test]), index=X.index[test])
        elif model == "gbm_reg":
            from lightgbm import LGBMRegressor

            m = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
            m.fit(X[train], y[train])
            s = pd.Series(m.predict(X[test]), index=X.index[test])
        else:
            raise ValueError(model)
        out.append(s)
    return pd.concat(out).rename(model)


def rank_ic(scores: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Monthly Spearman correlation between score and next-month return."""
    df = pd.DataFrame({"s": scores, "r": panel["ret"].reindex(scores.index)}).dropna()
    return df.groupby(level="date").apply(lambda g: g["s"].rank().corr(g["r"].rank()))


class TopN(Strategy):
    """Hold the `n` highest-scored symbols in equal weight, rebalanced on the
    dates `scores` has (month ends here)."""

    name = "top_n"
    params = {"scores": None, "n": 10}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        s = self.p["scores"].unstack("symbol").reindex(columns=data.symbols)
        top = s.rank(axis=1, ascending=False, method="first") <= self.p["n"]
        return top.astype(float).div(top.sum(axis=1).clip(lower=1), axis=0)
