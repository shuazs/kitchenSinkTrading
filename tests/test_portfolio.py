import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel
from backtester.data.synthetic import synthetic_market
from backtester.portfolio import combine, sleeve_returns
from backtester.strategies import BuyAndHold


def _rets(a, b, n=10):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"a": a, "b": b}, index=idx)


def test_no_rebalance_is_buy_and_hold_of_sleeves():
    r = _rets(0.01, -0.01)
    res = combine(r, {"a": 0.5, "b": 0.5}, rebalance=None, cost_bps=0)
    a = 2_500 * 1.01 ** 9
    b = 2_500 * 0.99 ** 9
    assert res.equity.iloc[-1] == pytest.approx(a + b)
    assert res.positions["cash"].iloc[-1] == pytest.approx(0.0)


def test_band_rebalances_back_to_target_and_charges_cost():
    r = _rets(0.0, 0.0)
    r.iloc[1, 0] = 0.5  # sleeve a jumps: 2500 -> 3750, weight 0.6
    res = combine(r, {"a": 0.5, "b": 0.5}, rebalance=None, band=0.05, cost_bps=10)
    nav = 6_250
    moved = 2 * (3_750 - nav / 2)
    assert res.costs.iloc[1] == pytest.approx(moved * 10 / 1e4)
    assert res.weights.iloc[1]["a"] == pytest.approx(0.5)


def test_cash_share_earns_rate_and_weights_validated():
    r = _rets(0.0, 0.0, n=253)
    res = combine(r, {"a": 0.5}, rebalance=None, cost_bps=0, cash_rate=0.04)
    assert res.positions["cash"].iloc[-1] > 2_500 * 1.035
    with pytest.raises(ValueError):
        combine(r, {"a": 0.7, "b": 0.7})


def test_drawdown_brake_halves_then_restores():
    r = _rets(0.0, 0.0, n=8)
    r["a"] = [0, -0.3, 0, 0, 1.0, 0, 0, 0]
    res = combine(r, {"a": 1.0}, rebalance=None, cost_bps=0, dd_brake=-0.2, dd_resume=-0.01)
    w = res.weights["a"]
    assert w.iloc[0] == pytest.approx(1.0)
    assert w.iloc[1] == pytest.approx(0.5)  # braked on the close of the drop
    assert w.iloc[3] == pytest.approx(0.5)
    assert res.equity.iloc[4] == pytest.approx(1_750 * 2 + 1_750)  # only half rode the rebound
    assert w.iloc[4] == pytest.approx(1.0)  # back above the resume level, full weight again


def test_sleeve_returns_mixes_calendars_and_matches_engine():
    data = synthetic_market(["X", "Y"], seed=3)
    bt = Backtester(5_000, costs=CostModel())
    rx = bt.run(BuyAndHold(symbols=["X"]), data)
    ry = bt.run(BuyAndHold(symbols=["Y"]), data)
    weekdays = ry.equity[ry.equity.index.dayofweek < 4]  # a sleeve with gaps
    rets = sleeve_returns({"x": rx, "y": weekdays})
    assert rets.index.equals(rx.equity.index)
    one = combine(rets, {"x": 1.0}, rebalance=None, cost_bps=0)
    assert np.allclose(one.equity.values, rx.equity.values)
