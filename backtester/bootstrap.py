"""Block-bootstrap confidence intervals for backtest statistics.

A backtest gives one Sharpe ratio from one history. The bootstrap asks how
much that number would move if history had dealt the same kind of days in a
different order: resample the daily returns with replacement many times,
recompute the statistic on every resample, and read the interval off the
percentiles (Efron 1979).

Returns are not independent from day to day (volatility clusters, trends
persist), so days are drawn in blocks rather than one at a time. This uses the
stationary bootstrap of Politis and Romano (1994, JASA): each block starts on
a random day and its length is geometric with mean `block`, wrapping around
the end of the sample. `block=1` is the plain i.i.d. bootstrap.

    rets = result.equity.pct_change().dropna()
    bootstrap_ci(rets)                    # CAGR, vol, Sharpe, max DD with 90% bands
    bootstrap_compare(rets, spy_rets)     # is the Sharpe gap to SPY more than luck?

What it does not do: it measures sampling noise only. It cannot see
overfitting from trying many variants (use `ml.contest.deflated_sharpe` and
the holdout for that), and it assumes the future draws from the same
distribution as the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import _periods_per_year

STATS = ("cagr", "volatility", "sharpe", "max_drawdown")


def stationary_indices(n: int, n_boot: int = 2000, block: float = 20, seed: int | None = 0) -> np.ndarray:
    """Row positions for `n_boot` resamples of a length-`n` series, shape
    (n_boot, n). Each position either continues the current block (next day,
    wrapping) or, with probability 1/block, jumps to a random day."""
    if n < 2:
        raise ValueError("need at least 2 observations")
    if block < 1:
        raise ValueError("block must be >= 1")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < 1.0 / block
    jump[:, 0] = True
    # Position of the latest jump at or before each column, then walk forward from it.
    cols = np.arange(n)
    last_jump = np.maximum.accumulate(np.where(jump, cols, 0), axis=1)
    start_at_jump = np.take_along_axis(starts, last_jump, axis=1)
    return (start_at_jump + cols - last_jump) % n


def _stats(r: np.ndarray, ppy: float, rf_p: float = 0.0) -> dict[str, np.ndarray]:
    """Statistics along the last axis, defined as in `metrics.summary`."""
    ex = r - rf_p
    sd = ex.std(axis=-1)
    growth = np.cumprod(1 + r, axis=-1)
    # The curve starts at 1 before the first return, so a first-day loss counts as drawdown.
    peak = np.maximum(np.maximum.accumulate(growth, axis=-1), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(sd > 0, ex.mean(axis=-1) / sd * np.sqrt(ppy), np.nan)
        cagr = np.where(growth[..., -1] > 0, growth[..., -1] ** (ppy / r.shape[-1]) - 1, np.nan)
    return {
        "cagr": cagr,
        "volatility": r.std(axis=-1) * np.sqrt(ppy),
        "sharpe": sharpe,
        "max_drawdown": (growth / peak - 1).min(axis=-1),
    }


def _clean(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        raise ValueError("need at least 2 returns")
    return r


def _ppy(index, periods_per_year: float | None) -> float:
    if periods_per_year is not None:
        return float(periods_per_year)
    return _periods_per_year(index) if isinstance(index, pd.DatetimeIndex) else 252.0


def bootstrap_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    block: float = 20,
    level: float = 0.90,
    rf: float = 0.0,
    periods_per_year: float | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Percentile confidence intervals for CAGR, volatility, Sharpe and max
    drawdown of a series of periodic returns (e.g. `equity.pct_change()`).

    Returns one row per statistic with `estimate` (the sample value), `lo`
    and `hi` (the `level` band) and `p_positive` (share of resamples above
    zero). `rf` is an annual rate. `attrs` records n_boot, block and level.
    """
    r = _clean(returns)
    ppy = _ppy(r.index, periods_per_year)
    x = r.to_numpy()
    point = _stats(x, ppy, rf / ppy)
    boot = _stats(x[stationary_indices(len(x), n_boot, block, seed)], ppy, rf / ppy)
    a = (1 - level) / 2
    out = pd.DataFrame(
        {
            "estimate": {k: float(point[k]) for k in STATS},
            "lo": {k: np.nanquantile(boot[k], a) for k in STATS},
            "hi": {k: np.nanquantile(boot[k], 1 - a) for k in STATS},
            "p_positive": {k: float(np.nanmean(boot[k] > 0)) for k in STATS},
        }
    ).loc[list(STATS)]
    out.attrs.update(n_boot=n_boot, block=block, level=level)
    return out


def bootstrap_compare(
    returns: pd.Series,
    benchmark: pd.Series,
    n_boot: int = 2000,
    block: float = 20,
    level: float = 0.90,
    rf: float = 0.0,
    periods_per_year: float | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Paired bootstrap of strategy minus benchmark. Both series are resampled
    on the same days, so their correlation is kept and the common market noise
    cancels: the band on the difference is much tighter than the two separate
    bands would suggest.

    Returns one row per statistic with `strategy`, `benchmark`, `diff`, the
    `lo`/`hi` band on the difference and `p_better` (share of resamples where
    the strategy's statistic is higher; for max_drawdown higher = shallower).
    """
    both = pd.concat([pd.Series(returns), pd.Series(benchmark)], axis=1, join="inner").dropna().astype(float)
    if len(both) < 2:
        raise ValueError("need at least 2 overlapping returns")
    ppy = _ppy(both.index, periods_per_year)
    s, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    idx = stationary_indices(len(both), n_boot, block, seed)
    ps, pb = _stats(s, ppy, rf / ppy), _stats(b, ppy, rf / ppy)
    bs, bb = _stats(s[idx], ppy, rf / ppy), _stats(b[idx], ppy, rf / ppy)
    a = (1 - level) / 2
    rows = {}
    for k in STATS:
        d = bs[k] - bb[k]
        rows[k] = {
            "strategy": float(ps[k]),
            "benchmark": float(pb[k]),
            "diff": float(ps[k] - pb[k]),
            "lo": np.nanquantile(d, a),
            "hi": np.nanquantile(d, 1 - a),
            "p_better": float(np.nanmean(d > 0)),
        }
    out = pd.DataFrame(rows).T.loc[list(STATS)]
    out.attrs.update(n_boot=n_boot, block=block, level=level, n_obs=len(both))
    return out
