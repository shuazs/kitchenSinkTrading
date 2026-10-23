"""Combine several strategy backtests into one fund with sleeve weights,
calendar or drift-band rebalancing, and an optional drawdown brake.

Each sleeve is run on its own (any `Result`, or any equity curve), then the
fund holds a dollar amount in each sleeve and grows it by the sleeve's daily
return. This is how a small account actually runs several strategies: one
sub-account per strategy, topped up or trimmed back to target now and then.

    rets = sleeve_returns({"upro": res_upro, "trend": res_trend, "crypto": res_crypto})
    fund = combine(rets, {"upro": 0.4, "trend": 0.4, "crypto": 0.2}, rebalance="QE", band=0.05)
    fund.stats()

Days a sleeve doesn't trade (weekends for ETFs) count as a 0% return, so a
crypto sleeve can sit next to stock sleeves on one calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Result

CASH = "cash"


def sleeve_returns(sleeves: dict) -> pd.DataFrame:
    """Daily returns of each sleeve on the union of their calendars.

    Values may be `Result`s or equity Series. Each column starts on its
    sleeve's first bar; days before that are NaN (the sleeve didn't exist).
    """
    eqs = {k: (v.equity if isinstance(v, Result) else v) for k, v in sleeves.items()}
    idx = sorted(set().union(*[e.index for e in eqs.values()]))
    out = {}
    for k, e in eqs.items():
        e = e.reindex(idx)
        r = e.ffill().pct_change()
        r[e.ffill().isna()] = np.nan
        r.iloc[e.index.get_loc(e.first_valid_index())] = 0.0
        out[k] = r
    return pd.DataFrame(out, index=pd.DatetimeIndex(idx))


def combine(
    returns: pd.DataFrame,
    weights: dict,
    initial_capital: float = 5_000.0,
    rebalance: str | None = "QE",
    band: float | None = None,
    cost_bps: float = 5.0,
    cash_rate: float | pd.Series = 0.0,
    dd_brake: float | None = None,
    dd_resume: float | None = None,
    brake_scale: float = 0.5,
    start=None,
    end=None,
) -> Result:
    """Simulate a fund holding `weights` of each sleeve (the rest in cash).

    rebalance: pandas period alias ("ME", "QE", "YE") to reset to target at
        each period's last bar, or None for never (except via `band`).
    band: also reset all sleeves when any sleeve drifts more than this many
        weight points from its (current) target.
    cost_bps: cost per dollar moved between sleeves (both legs' spread and
        commissions, on top of the sleeves' own trading costs).
    cash_rate: annual rate earned on the cash share (float or daily Series).
    dd_brake: when the fund's drawdown from its peak reaches this (e.g.
        -0.15), scale every sleeve to `brake_scale` of its target, the rest
        to cash, until the drawdown recovers above `dd_resume` (default half
        of `dd_brake`). Decided on close t, applied from t+1.
    """
    w = pd.Series(weights, dtype=float)
    if (w < 0).any() or w.sum() > 1 + 1e-9:
        raise ValueError("weights must be >= 0 and sum to at most 1")
    rets = returns[list(w.index)].loc[start:end]
    rets = rets.loc[rets.notna().all(axis=1).idxmax():].fillna(0.0)
    idx = rets.index
    if isinstance(cash_rate, pd.Series):
        cr = cash_rate.reindex(idx).ffill().fillna(0.0)
    else:
        cr = pd.Series(cash_rate, index=idx)
    gaps = pd.Series(idx, index=idx).diff().dt.days.fillna(1).clip(lower=1)
    cash_growth = (1 + cr) ** (gaps / 365.25) - 1

    if rebalance:
        period_ends = set(pd.Series(idx, index=idx).resample(rebalance).last().dropna())
    else:
        period_ends = set()
    resume = dd_resume if dd_resume is not None else (dd_brake / 2 if dd_brake else None)

    names = list(w.index) + [CASH]
    vals = np.zeros((len(idx), len(names)))
    costs = np.zeros(len(idx))
    target = np.append(w.values, 1 - w.sum())
    hold = target * initial_capital
    peak = initial_capital
    brake_on = False
    r = rets.values
    for i, d in enumerate(idx):
        if i > 0:
            hold = hold * np.append(1 + r[i], 1 + cash_growth.iloc[i])
        nav = hold.sum()
        peak = max(peak, nav)
        dd = nav / peak - 1
        cur = np.append(w.values * (brake_scale if brake_on else 1.0), 0.0)
        cur[-1] = 1 - cur[:-1].sum()
        want_brake = brake_on
        if dd_brake is not None:
            if not brake_on and dd <= dd_brake:
                want_brake = True
            elif brake_on and dd >= resume:
                want_brake = False
        if want_brake != brake_on:
            brake_on = want_brake
            cur = np.append(w.values * (brake_scale if brake_on else 1.0), 0.0)
            cur[-1] = 1 - cur[:-1].sum()
            reset = True
        else:
            drift = np.abs(hold / nav - cur)[:-1].max() if nav > 0 else 0.0
            reset = d in period_ends or (band is not None and drift > band)
        if reset and nav > 0:
            new = cur * nav
            moved = np.abs(new - hold)[:-1].sum()
            c = moved * cost_bps / 1e4
            costs[i] = c
            hold = cur * (nav - c)
        vals[i] = hold

    alloc = pd.DataFrame(vals, index=idx, columns=names)
    equity = alloc.sum(axis=1)
    return Result(
        equity=equity,
        positions=alloc,
        weights=alloc[list(w.index)].div(equity, axis=0),
        trades=pd.DataFrame(columns=["qty", "fill_price"]),
        costs=pd.Series(costs, index=idx),
        initial_capital=initial_capital,
        strategy="fund(" + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()) + ")",
    )

