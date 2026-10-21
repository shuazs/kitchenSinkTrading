import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel, MarketData
from backtester.data.synthetic import synthetic_bars, synthetic_market
from backtester.strategies import SectorRotation, TurnOfMonth, VixTermStructure, VolTarget
from backtester.strategies.turn_of_month import month_position

FREE = CostModel(spread_bps=0, slippage_bps=0)


def test_month_position_counts_trading_days():
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    start, end = month_position(idx)
    assert start.loc["2024-01-01"] == 1 and end.loc["2024-01-31"] == 1
    assert end.loc["2024-01-30"] == 2 and start.loc["2024-02-06"] == 4



def test_month_position_does_not_treat_last_bar_as_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-01-17")
    start, end = month_position(idx)
    assert end.iloc[-1] == 11  # Jan 17 to Jan 31 is 11 weekdays
    assert start.iloc[-1] == 13

def test_turn_of_month_holds_last_day_through_day_three():
    data = synthetic_market(["SPY"], n=120)
    res = Backtester(costs=FREE).run(TurnOfMonth(), data)
    idx = data.index
    start, end = month_position(idx)
    held = res.positions["SPY"] > 0
    # Held through the close of the last day and days 1-3; flat by day 5.
    assert held[end == 1].iloc[1:].all()
    assert held[start <= 3].iloc[1:].all()
    assert not held[(start >= 5) & (end >= 3)].any()
    buys = res.trades[(res.trades.qty > 0) & (res.trades.date > idx[10])]
    assert set(end.loc[buys.date]) == {1}


def test_turn_of_month_trend_filter_blocks_entries_in_downtrend():
    bars = synthetic_bars(n=400, mu=-0.8, sigma=0.08, seed=3)
    data = MarketData({"SPY": bars})
    res = Backtester(costs=FREE).run(TurnOfMonth(ma=50), data, start=data.index[100])
    assert len(res.trades) == 0


def test_vix_term_structure_follows_contango():
    idx = pd.bdate_range("2020-01-01", periods=60)
    flat = lambda v: pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": 0}, index=idx)
    vix = np.where(np.arange(60) < 30, 15.0, 30.0)
    data = MarketData({"SVXY": flat(np.full(60, 50.0)), "^VIX": flat(vix), "^VIX3M": flat(np.full(60, 20.0))})
    w = VixTermStructure().generate_targets(data)["SVXY"]
    assert (w.iloc[:30] == 1).all() and (w.iloc[30:] == 0).all()
    res = Backtester(costs=FREE).run(VixTermStructure(), data)
    assert res.positions["SVXY"].iloc[29] > 0 and res.positions["SVXY"].iloc[31] == 0


def test_vol_target_scales_down_when_volatility_rises():
    calm = synthetic_bars(n=200, sigma=0.08, seed=1)
    wild = synthetic_bars(n=200, sigma=0.6, seed=2, start_price=calm["close"].iloc[-1])
    wild.index = pd.bdate_range(calm.index[-1] + pd.offsets.BDay(), periods=200)
    data = MarketData({"X": pd.concat([calm, wild])})
    w = VolTarget(symbol="X", target_vol=0.15, band=0.0).generate_targets(data)["X"]
    assert w.iloc[150:200].mean() == pytest.approx(1.0)
    assert w.iloc[-50:].mean() < 0.4
    banded = VolTarget(symbol="X", target_vol=0.15, band=0.2).generate_targets(data)["X"]
    assert banded.diff().abs().gt(0).sum() < w.diff().abs().gt(0).sum()


def test_sector_rotation_picks_strongest_positive_sectors():
    data = synthetic_market(["A", "B", "C", "D", "SAFE"], n=400, seed=5)
    strat = SectorRotation(universe=["A", "B", "C", "D"], safe="SAFE", top_n=2, lookbacks=(63,))
    w = strat.generate_targets(data)
    assert np.allclose(w.sum(axis=1), 1.0) or (w.sum(axis=1) == 0).any()
    close = data.field("close")
    d = w.index[-1]
    mom = close.loc[:d].iloc[-1] / close.loc[:d].iloc[-64] - 1
    expected = [s for s in mom[["A", "B", "C", "D"]].nlargest(2).index if mom[s] > 0]
    assert set(w.loc[d][w.loc[d] > 0].index) - {"SAFE"} == set(expected)
