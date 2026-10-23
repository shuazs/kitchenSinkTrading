import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


class PreFOMCOverlay(Strategy):
    """Add `add` weight in `symbol` around scheduled FOMC statements.

    Lucca and Moench (2015): US stocks earned most of their excess return in
    the 24 hours before FOMC statements. Targets set at the close two days and
    one day before the statement are traded at the next open, so the extra
    exposure is held from the open of the day before to the open after the
    statement. `base` is any strategy (default: its own targets of zero);
    total weight in `symbol` is capped at `cap`. Dates in `extra[dates_key]`.
    """

    name = "pre_fomc_overlay"
    params = {"base": None, "symbol": "UPRO", "add": 0.5, "cap": 1.0, "dates_key": "fomc"}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        if self.p["base"] is not None:
            w = self.p["base"].generate_targets(data).reindex(idx).ffill().fillna(0.0)
        else:
            w = pd.DataFrame(0.0, index=idx, columns=data.symbols)
        flag = np.zeros(len(idx))
        for d in pd.DatetimeIndex(data.extra[self.p["dates_key"]]):
            i = idx.searchsorted(d)
            if i < len(idx) and idx[i] == d and i >= 2:
                flag[i - 2 : i] = 1.0
        w = w.copy()
        sym = self.p["symbol"]
        w[sym] = np.minimum(w[sym].to_numpy() + self.p["add"] * flag, self.p["cap"])
        return w


class CalendarOverlay(Strategy):
    """Scale `base`'s weight in `symbol` by a per-day multiplier.

    `extra[mult_key]` is a Series indexed by the trading day the exposure is
    held (1.0 = unchanged); the target is set at the previous close, so a
    day's multiplier only uses the calendar. Weight is capped at `cap`.
    """

    name = "calendar_overlay"
    params = {"base": None, "symbol": "UPRO", "mult_key": "mult", "cap": 1.0}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        w = self.p["base"].generate_targets(data).reindex(idx).ffill().fillna(0.0).copy()
        mult = data.extra[self.p["mult_key"]].reindex(idx).fillna(1.0).shift(-1).fillna(1.0)
        sym = self.p["symbol"]
        w[sym] = np.minimum(w[sym] * mult, self.p["cap"])
        return w


def opex_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly option expiry: third Friday (previous trading day if a holiday)."""
    out = []
    for m in pd.period_range(index[0], index[-1], freq="M"):
        fridays = pd.date_range(m.start_time, m.end_time, freq="W-FRI")
        d = fridays[2]
        i = index.searchsorted(d, side="right") - 1
        if i >= 0 and index[i].to_period("M") == m:
            out.append(index[i])
    return pd.DatetimeIndex(out)
