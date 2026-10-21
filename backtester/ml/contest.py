"""Run models through the folds and judge them fairly: same folds, same
sizing, same costs, plus the Deflated Sharpe ratio for the number of trials."""
from __future__ import annotations

import time
from math import erf, exp, sqrt

import numpy as np
import pandas as pd

from .splits import Fold


def out_of_sample_positions(panel: pd.DataFrame, make_model, folds: list[Fold], verbose: bool = False) -> pd.Series:
    """Fit a fresh model on each fold and keep only its test-period positions."""
    parts = []
    for f in folds:
        t = time.time()
        m = make_model().fit(panel, f)
        parts.append(m.predict(panel, f.test))
        if verbose:
            print(f"  {getattr(m, 'name', 'model')} fold {f.name}: {time.time() - t:.1f}s", flush=True)
    return pd.concat(parts).sort_index()


def out_of_sample_seed_positions(panel: pd.DataFrame, make_model, folds: list[Fold]) -> pd.DataFrame:
    """Like out_of_sample_positions, one column per seed (models with predict_seeds)."""
    parts = []
    for f in folds:
        m = make_model().fit(panel, f)
        parts.append(m.predict_seeds(panel, f.test))
    return pd.concat(parts).sort_index()


def information_coefficient(pos: pd.Series, panel: pd.DataFrame) -> float:
    """Mean weekly rank correlation between position and next week's vol-scaled return."""
    df = pd.DataFrame({"pos": pos, "y": panel["target"].reindex(pos.index)}).dropna()
    ic = df.groupby(level="date").apply(lambda g: g["pos"].rank().corr(g["y"].rank()) if len(g) > 2 else np.nan)
    return float(ic.mean())


def pooled_correlation(pos: pd.Series, panel: pd.DataFrame) -> float:
    """Correlation of position with next week's vol-scaled return across all rows
    (includes timing across weeks, which the cross-sectional IC leaves out)."""
    df = pd.DataFrame({"pos": pos, "y": panel["target"].reindex(pos.index)}).dropna()
    return float(df["pos"].corr(df["y"]))


def positions_frame(pos: pd.Series) -> pd.DataFrame:
    """(date, symbol) Series -> date x symbol frame for SignalBlend."""
    return pos.unstack("symbol")


def yearly_sharpe(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().dropna()
    ppy = len(r) / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    g = r.groupby(r.index.year)
    return g.mean() / g.std() * np.sqrt(ppy)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """Bailey and Lopez de Prado (2014): probability that the true Sharpe is
    above zero after allowing for `n_trials` tries, skew and fat tails.

    Works in per-period Sharpe units. `trial_sharpes` (per period) sets the
    spread of Sharpes across trials; without it a spread of 0.5 / sqrt(ppy)
    annualised-equivalent is assumed.
    """
    r = returns.dropna()
    T = len(r)
    if T < 10 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    ppy = T / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        v = float(np.var(trial_sharpes, ddof=1))
    else:
        v = (0.5 / sqrt(ppy)) ** 2
    if n_trials > 1:
        g = 0.5772156649
        sr0 = sqrt(v) * ((1 - g) * _norm_ppf(1 - 1 / n_trials) + g * _norm_ppf(1 - 1 / (n_trials * exp(1))))
    else:
        sr0 = 0.0
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return _norm_cdf((sr - sr0) * sqrt(T - 1) / denom)


__all__ = [
    "deflated_sharpe",
    "information_coefficient",
    "out_of_sample_positions",
    "out_of_sample_seed_positions",
    "pooled_correlation",
    "positions_frame",
    "yearly_sharpe",
]
