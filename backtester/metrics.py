"""Performance statistics for backtest results."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    # Crypto trades 365 days a year; infer from the data rather than assume.
    if len(index) < 3:
        return TRADING_DAYS
    span_years = (index[-1] - index[0]).days / 365.25
    return len(index) / span_years if span_years > 0 else TRADING_DAYS


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(result, benchmark: pd.Series | None = None, rf: float | pd.Series = 0.0) -> pd.Series:
    eq = result.equity
    rets = eq.pct_change().dropna()
    ppy = _periods_per_year(eq.index)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if isinstance(rf, pd.Series):
        rf_p = rf.reindex(rets.index).ffill().fillna(0.0) / ppy
    else:
        rf_p = rf / ppy
    ex = rets - rf_p
    # Downside deviation: root mean square of the shortfall below zero, over all periods.
    downside = np.sqrt((ex.clip(upper=0) ** 2).mean()) * np.sqrt(ppy)
    dd = drawdown(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    traded = (result.trades["qty"].abs() * result.trades["fill_price"]).sum() if len(result.trades) else 0.0
    avg_eq = eq.mean()

    out = {
        "start": eq.index[0].date(),
        "end": eq.index[-1].date(),
        "final_equity": eq.iloc[-1],
        "total_return": eq.iloc[-1] / result.initial_capital - 1,
        "cagr": cagr,
        "volatility": rets.std(ddof=0) * np.sqrt(ppy),
        "sharpe": ex.mean() / ex.std(ddof=0) * np.sqrt(ppy) if ex.std(ddof=0) > 0 else np.nan,
        "sortino": ex.mean() * ppy / downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "exposure": (result.weights.abs().sum(axis=1) > 0.01).mean(),
        "n_trades": len(result.trades),
        "annual_turnover": traded / avg_eq / years if years > 0 else np.nan,
        "total_costs": result.costs.sum(),
        "cost_drag_pct_per_year": result.costs.sum() / avg_eq / years if years > 0 else np.nan,
    }
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill()
        b_rets = b.pct_change().dropna()
        b_eq = b / b.iloc[0]
        b_cagr = b_eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        joined = pd.concat([rets, b_rets], axis=1).dropna()
        beta = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1] / joined.iloc[:, 1].var() if len(joined) > 2 else np.nan
        out.update(
            {
                "bench_cagr": b_cagr,
                "bench_max_drawdown": drawdown(b_eq).min(),
                "bench_sharpe": b_rets.mean() / b_rets.std(ddof=0) * np.sqrt(ppy),
                "beta": beta,
            }
        )
    return pd.Series(out, name=result.strategy)
