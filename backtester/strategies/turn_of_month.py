import pandas as pd

from ..core import MarketData, Strategy


def month_position(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Trading day number from the start of the month (1 = first) and from
    the end (1 = last). Exchange holidays are published in advance, so
    knowing which day is the month's last is not lookahead."""
    s = pd.Series(1, index=index)
    g = s.groupby(index.to_period("M"))
    from_start = g.cumsum()
    from_end = g.transform("sum") - from_start + 1
    return from_start, from_end


class TurnOfMonth(Strategy):
    """Hold `symbol` over the turn of the month, else `risk_off` (or cash).

    Lakonishok and Smidt (1988), McConnell and Xu (2008): most of the equity
    premium arrives from the last trading day of a month through the first
    few of the next. Signals form at the close and fill at the next open, so
    with the defaults the position is bought at the open of the last trading
    day and sold at the open of trading day `exit_day + 1`.

    enter_days: trading days before month end to be invested (1 = last day).
    exit_day: last trading day of the new month that is held.
    signal, ma: optional trend filter; only enter while `signal` closes above
        its `ma`-day average on the entry signal day.
    """

    name = "turn_of_month"
    params = {"symbol": "SPY", "enter_days": 1, "exit_day": 3, "risk_off": None, "signal": None, "ma": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        from_start, from_end = month_position(idx)
        # Weight set at close t applies from open t+1, hence the one-day shift.
        on = (from_end <= self.p["enter_days"] + 1) | (from_start < self.p["exit_day"])
        if self.p["ma"]:
            close = data.bars[self.p["signal"] or self.p["symbol"]]["close"].reindex(idx).ffill()
            trend = close > close.rolling(self.p["ma"]).mean()
            # Decide once per window, on its first signal day, and keep it.
            start = on & ~on.shift(fill_value=False)
            allowed = trend.where(start).ffill().fillna(False).astype(bool)
            on = on & allowed

        w = pd.DataFrame(0.0, index=idx, columns=data.symbols)
        w[self.p["symbol"]] = on.astype(float)
        if self.p["risk_off"]:
            w[self.p["risk_off"]] = (~on).astype(float)
        return w
