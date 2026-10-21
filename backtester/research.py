"""Parameter sweeps, walk-forward tests and event studies, with the overfitting caveats built in."""
from __future__ import annotations

import itertools

import numpy as np

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


def event_study(data: MarketData, events: pd.DataFrame, horizons=(5, 21, 63, 126), benchmark: str | None = None, shift: int = 1) -> pd.DataFrame:
    """Return of each event's stock from the open `shift` trading days after
    the event's signal day (1 = the next open, as the engine would trade;
    negative values start before the event, which only a forecaster could
    do) to the close `h` trading days later, for each horizon `h`.

    `events` needs columns date and symbol. With `benchmark`, adds the same
    window's benchmark return and the excess ("abnormal") return. Events
    whose window runs past the data are left as NaN for that horizon.
    """
    idx = data.index
    opens = data.field("open")
    closes = data.field("close").ffill()
    rows = []
    for e in events.itertuples():
        if e.symbol not in data.bars:
            continue
        i = idx.searchsorted(pd.Timestamp(e.date)) + shift
        if i < 0 or i >= len(idx) or pd.isna(opens.at[idx[i], e.symbol]):
            continue
        row = {"date": e.date, "symbol": e.symbol}
        for h in horizons:
            j = i + h - 1
            if j >= len(idx):
                continue
            row[f"r{h}"] = closes.iat[j, closes.columns.get_loc(e.symbol)] / opens.at[idx[i], e.symbol] - 1
            if benchmark:
                b = closes[benchmark].iat[j] / opens[benchmark].iat[i] - 1
                row[f"x{h}"] = row[f"r{h}"] - b
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_events(study: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, hit rate and t-stat for each return column of `event_study`."""
    cols = [c for c in study.columns if c[0] in "rx" and c[1:].isdigit()]
    out = {}
    for c in cols:
        s = study[c].dropna()
        out[c] = {"n": len(s), "mean": s.mean(), "median": s.median(), "hit_rate": (s > 0).mean(),
                  "t": s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 1 else np.nan}
    return pd.DataFrame(out).T
