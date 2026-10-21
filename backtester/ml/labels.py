"""Trade events and their labels, for meta-labeling (Lopez de Prado 2018,
"Advances in Financial Machine Learning", ch. 3).

A primary strategy decides the side (here: long on a breakout). Each entry is
an event. Two labels are offered for the secondary model:

- `trade`: did the primary trade itself, held to the primary's own exit, earn
  more than `min_ret`? This is the question a filter really asks.
- `triple_barrier`: from the entry, which comes first: a profit-take at
  `pt` x daily vol x sqrt(horizon), a stop at `sl` x the same, or the end of
  `horizon` days (labelled by the sign of the return then).

Every event keeps its `end` date (the last close its label depends on), so
walk-forward splits can purge events whose labels overlap the test period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entries_and_exits(held: pd.DataFrame) -> pd.DataFrame:
    """Trades from a date x symbol boolean "position on" frame.

    `entry` is the signal close (the engine buys at the next open); `exit` is
    the close the position turns off (sold at the next open), or the last
    date for trades still open.
    """
    rows = []
    last = held.index[-1]
    for sym in held.columns:
        h = held[sym].fillna(False).astype(bool).to_numpy()
        on = np.flatnonzero(h & ~np.r_[False, h[:-1]])
        off = np.flatnonzero(~h & np.r_[False, h[:-1]])
        for i in on:
            later = off[off > i]
            j = later[0] if len(later) else None
            rows.append((held.index[i], sym, held.index[j] if j is not None else last, j is None))
    out = pd.DataFrame(rows, columns=["entry", "symbol", "exit", "open"])
    return out.sort_values(["entry", "symbol"]).reset_index(drop=True)


def trade_returns(trades: pd.DataFrame, opens: pd.DataFrame, cost: float = 0.0) -> pd.Series:
    """Return of each trade from the open after `entry` to the open after
    `exit`, less `cost` (round trip, as a fraction)."""
    idx = opens.index
    out = np.full(len(trades), np.nan)
    for k, t in enumerate(trades.itertuples()):
        o = opens[t.symbol]
        i = idx.searchsorted(t.entry, side="right")
        j = idx.searchsorted(t.exit, side="right")
        seg = o.iloc[i : j + 1].dropna()
        if len(seg) >= 2:
            out[k] = seg.iloc[-1] / seg.iloc[0] - 1 - cost
    return pd.Series(out, index=trades.index, name="ret")


def triple_barrier(
    trades: pd.DataFrame,
    close: pd.DataFrame,
    daily_vol: pd.DataFrame,
    pt: float = 2.0,
    sl: float = 1.0,
    horizon: int = 20,
) -> pd.DataFrame:
    """First barrier touched within `horizon` bars after each entry.

    Returns columns tb_ret (return at the touch), tb_label (1 profit-take or
    positive at the vertical barrier, 0 otherwise) and tb_end (touch date).
    """
    idx = close.index
    res = []
    for t in trades.itertuples():
        i = idx.get_loc(t.entry)
        c = close[t.symbol].iloc[i : i + horizon + 1].dropna()
        v = daily_vol.at[t.entry, t.symbol]
        if len(c) < 2 or not np.isfinite(v) or v <= 0:
            res.append((np.nan, np.nan, t.entry))
            continue
        width = v * np.sqrt(horizon)
        path = c.iloc[1:] / c.iloc[0] - 1
        up = path[path >= pt * width]
        dn = path[path <= -sl * width]
        first_up = up.index[0] if len(up) else None
        first_dn = dn.index[0] if len(dn) else None
        if first_up is not None and (first_dn is None or first_up <= first_dn):
            res.append((path[first_up], 1.0, first_up))
        elif first_dn is not None:
            res.append((path[first_dn], 0.0, first_dn))
        else:
            res.append((path.iloc[-1], float(path.iloc[-1] > 0), path.index[-1]))
    return pd.DataFrame(res, columns=["tb_ret", "tb_label", "tb_end"], index=trades.index)
