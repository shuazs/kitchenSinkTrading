"""Weather-signal research helpers (thread "Find weather-based trading edges").

Data comes from `backtester.data.weather` (GHCN actuals, GEFS forecasts, ISD
hourly obs, IBTrACS hurricanes). The functions here turn it into signals:

- `run_revisions`: change in a forecast quantity between two GEFS runs, averaged
  over target days `a..b` ahead (the "weather revision" natural-gas traders watch).
- `trading_day_revisions`: the same between consecutive trading days' runs.
- `kalshi_bounds`: settle range of Kalshi daily-high markets from their subtitles.
- `locked_side`: the side of a Kalshi high-temperature bracket that the
  observed max-so-far already guarantees.
- `event_returns`: open t+1 -> close t+h returns after event days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_revisions(table: pd.DataFrame, value: str, windows: dict, inits=None) -> pd.DataFrame:
    """Mean change of `value` for the same target dates between each init and the previous one.

    table: long frame with columns init, date, value (one row per run and target day).
    windows: {"name": (a, b)} lead days counted from the newer run's date.
    inits: optional subset/order of runs to compare (e.g. trading days only).
    """
    piv = table.pivot_table(index="init", columns="date", values=value)
    inits = piv.index if inits is None else pd.DatetimeIndex(inits).intersection(piv.index)
    out = {k: [] for k in windows}
    for i in range(1, len(inits)):
        t, tp = inits[i], inits[i - 1]
        for k, (a, b) in windows.items():
            dates = pd.date_range(t.normalize() + pd.Timedelta(days=a), t.normalize() + pd.Timedelta(days=b))
            dates = dates.intersection(piv.columns)
            x = piv.loc[t, dates] - piv.loc[tp, dates]
            out[k].append(x.mean() if len(dates) and x.notna().sum() >= 0.8 * len(dates) else np.nan)
    return pd.DataFrame(out, index=inits[1:])


def trading_day_revisions(table: pd.DataFrame, value: str, days: pd.DatetimeIndex, windows: dict) -> pd.DataFrame:
    """`run_revisions` between the runs of consecutive trading days, indexed by trading day."""
    inits = pd.DatetimeIndex(sorted(table["init"].unique()))
    inits = inits[inits.normalize().isin(days)]
    r = run_revisions(table, value, windows, inits)
    r.index = r.index.normalize()
    return r.reindex(days)


def rolling_z(x: pd.Series | pd.DataFrame, window: int = 250, min_periods: int = 60):
    """x divided by its trailing standard deviation (known before t)."""
    return x / x.rolling(window, min_periods=min_periods).std().shift(1)


def kalshi_bounds(markets: pd.DataFrame) -> pd.DataFrame:
    """Add kind ('ge', 'le', 'br'), a, b (inclusive whole deg F) from `yes_sub_title`.

    '87° or higher' -> ge, a=87; '81° or lower' -> le, b=81; '82° to 83°' -> br, a=82, b=83.
    """
    sub = markets["yes_sub_title"].fillna("")
    lo = sub.str.extract(r"(-?\d+(?:\.\d)?)°? or (?:higher|above)")[0].astype(float)
    hi = sub.str.extract(r"(-?\d+(?:\.\d)?)°? or (?:below|lower)")[0].astype(float)
    br = sub.str.extract(r"(-?\d+)°? to (-?\d+)").astype(float)
    out = markets.copy()
    out["kind"] = np.where(lo.notna(), "ge", np.where(hi.notna(), "le", np.where(br[0].notna(), "br", None)))
    out["a"] = np.where(lo.notna(), lo, np.where(hi.notna(), -999, br[0]))
    out["b"] = np.where(lo.notna(), 999, np.where(hi.notna(), hi, br[1]))
    return out


def locked_side(kind, a, b, max_so_far, margin: float = 1.0):
    """'yes' / 'no' / None: the outcome the observed max-so-far already guarantees.

    A daily high only rises, so 'ge a' is settled Yes once max >= a, and every
    bracket or 'le' contract whose top b is passed (max >= b + 1) is settled No.
    `margin` (deg F) guards against the METAR-to-climate-report rounding gap.
    """
    kind, a, b, m = (np.asarray(v) for v in (kind, a, b, max_so_far))
    yes = (kind == "ge") & (m >= a + margin - 0.5)
    no = (kind != "ge") & (m >= b + 1 + margin - 0.5)
    return np.where(yes, "yes", np.where(no, "no", None))


def event_returns(open_: pd.Series, close: pd.Series, events, hold: int, cost: float = 0.0) -> pd.Series:
    """Return from the open after each event day to the close `hold` sessions later, minus `cost`."""
    days = close.index
    pos = days.searchsorted(pd.DatetimeIndex(events))
    keep = pos + hold < len(days)
    pos = pos[keep]
    r = close.iloc[pos + hold].to_numpy() / open_.iloc[pos + 1].to_numpy() - 1 - cost
    return pd.Series(r, index=days[pos])
