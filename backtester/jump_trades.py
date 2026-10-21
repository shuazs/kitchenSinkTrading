"""Trading an asset after a sudden move in a prediction market.

A jump is a period where a signal (such as the Polymarket Iran escalation
index) moves at least `threshold`. The trade goes in the jump's direction at
the open of the first asset bar that starts after the jump period ends, and
exits at the close `hold` bars later. Bars are hourly and indexed by their
start, so a jump measured over [t, t+1h) can only be acted on from t+1h:
nothing is traded on a price that came before the signal was known.

`gap` is how far the asset had already moved between its last close before
or during the jump and our entry; it is reported, never earned. A jump while
the asset is closed (oil futures on a Saturday) usually shows up as a gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def jump_events(signal: pd.Series, threshold: float, min_gap: pd.Timedelta = pd.Timedelta(hours=6)) -> pd.DataFrame:
    """Periods where |signal| >= threshold, keeping one per `min_gap` window. Columns: time, direction, size."""
    rows, last = [], None
    for t, v in signal[signal.abs() >= threshold].items():
        if last is None or t - last >= min_gap:
            rows.append({"time": t, "direction": int(np.sign(v)), "size": float(v)})
            last = t
    return pd.DataFrame(rows, columns=["time", "direction", "size"])


def jump_trades(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    hold: int = 3,
    delay: int = 0,
    cost_bps: float = 6.0,
    period: pd.Timedelta = pd.Timedelta(hours=1),
) -> pd.DataFrame:
    """One row per event: entry/exit times, whether the asset traded during the jump, gap and net return.

    `bars` needs open and close columns. `delay` skips that many extra bars
    before entering (a slow trader). `cost_bps` is the round-trip cost.
    Returns are log returns signed by the jump direction.
    """
    rows = []
    idx = bars.index
    for ev in events.itertuples(index=False):
        j = idx.searchsorted(ev.time + period) + delay
        if j + hold - 1 >= len(idx) or j == 0:
            continue
        ref = bars["close"].iloc[j - 1]
        entry, exit_ = bars["open"].iloc[j], bars["close"].iloc[j + hold - 1]
        rows.append({
            "time": ev.time,
            "direction": ev.direction,
            "size": ev.size,
            "asset_open": ev.time in idx,
            "entry_time": idx[j],
            "exit_time": idx[j + hold - 1],
            "gap": ev.direction * np.log(entry / ref),
            "ret": ev.direction * np.log(exit_ / entry) - cost_bps / 1e4,
        })
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> pd.Series:
    r = trades["ret"]
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else np.nan
    return pd.Series({
        "trades": n,
        "mean_bp": r.mean() * 1e4 if n else np.nan,
        "t_stat": r.mean() / sd * np.sqrt(n) if n > 1 and sd > 0 else np.nan,
        "hit_rate": (r > 0).mean() if n else np.nan,
        "worst_bp": r.min() * 1e4 if n else np.nan,
        "total_pct": r.sum() * 100,
    })


def grid(bars: pd.DataFrame, signal: pd.Series, thresholds, holds, open_only: bool = True, **kw) -> pd.DataFrame:
    """summarize() for every threshold x hold, to see whether a result is a lone lucky cell."""
    out = []
    for th in thresholds:
        ev = jump_events(signal, th)
        if open_only:
            ev = ev[ev["time"].isin(bars.index)]
        for h in holds:
            tr = jump_trades(bars, ev, hold=h, **kw)
            out.append({"threshold": th, "hold": h, **summarize(tr).to_dict()})
    return pd.DataFrame(out)
