"""Parameter sweeps and walk-forward tests, with the overfitting caveats built in."""
from __future__ import annotations

import itertools

import pandas as pd

from .core import Backtester, MarketData


def sweep(strategy_cls, grid: dict, data: MarketData, bt: Backtester, start=None, end=None, **fixed) -> pd.DataFrame:
    """Run every combination in `grid`. The number of rows is the number of
    variants tried: the more you try, the more the best one is luck."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**fixed, **dict(zip(keys, combo))}
        res = bt.run(strategy_cls(**params), data, start, end)
        rows.append({**dict(zip(keys, combo)), **res.stats().drop(["start", "end"]).to_dict()})
    out = pd.DataFrame(rows)
    out.attrs["n_variants"] = len(out)
    return out


def walk_forward(
    strategy_cls,
    grid: dict,
    data: MarketData,
    bt: Backtester,
    train_years: int = 5,
    test_years: int = 1,
    metric: str = "sharpe",
    **fixed,
) -> tuple[pd.Series, pd.DataFrame]:
    """Pick the best params on each training window, trade them on the next
    test window, and stitch the out-of-sample test windows together.

    Returns (out-of-sample equity curve, table of chosen params per window).
    Warm-up: each test run starts at the test window but the strategy sees
    all prior data, so long lookbacks are filled.
    """
    idx = data.index
    start = idx[0]
    picks, curves = [], []
    capital = bt.initial_capital
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if train_end >= idx[-1]:
            break
        table = sweep(strategy_cls, grid, data, bt, start, train_end, **fixed)
        best = table.sort_values(metric, ascending=False).iloc[0]
        params = {k: best[k] for k in grid}
        params = {k: (int(v) if isinstance(v, float) and float(v).is_integer() and isinstance(grid[k][0], int) else v) for k, v in params.items()}
        test_bt = Backtester(**{**bt.__dict__, "initial_capital": capital})
        res = test_bt.run(strategy_cls(**fixed, **params), data, train_end, min(test_end, idx[-1]))
        curves.append(res.equity)
        capital = res.equity.iloc[-1]
        picks.append({"train_start": start.date(), "test_start": train_end.date(), **params, f"train_{metric}": best[metric], f"test_{metric}": res.stats()[metric]})
        start = start + pd.DateOffset(years=test_years)
    equity = pd.concat(curves)
    equity = equity[~equity.index.duplicated(keep="last")]
    return equity, pd.DataFrame(picks)
