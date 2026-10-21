import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel, MarketData
from backtester.data.synthetic import synthetic_market
from backtester.memecoin_calls import K, V0_SOL, buy, mc_to_v, sell, simulate_calls
from backtester.strategies import Breakout, BuyAndHold, CrossSectionalMomentum, Sleeves, VolTargetBreakout

FREE = CostModel(spread_bps=0, slippage_bps=0)


def trending_market(n=200):
    idx = pd.date_range("2024-01-01", periods=n)
    bars = {}
    for sym, drift in [("BTC-USD", 0.002), ("UP", 0.01), ("FLAT", 0.0), ("DOWN", -0.01)]:
        close = 100 * np.exp(drift * np.arange(n))
        bars[sym] = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    return MarketData(bars)


def test_momentum_picks_strongest_coins_with_positive_return():
    data = trending_market()
    w = CrossSectionalMomentum(top_n=2, lookback=10, trend_ma=20).generate_targets(data)
    last = w.iloc[-1]
    assert last["UP"] == pytest.approx(0.5)
    assert last["BTC-USD"] == pytest.approx(0.5)
    assert last["DOWN"] == 0 and last["FLAT"] == 0
    assert (w.index[1] - w.index[0]).days == 7


def test_momentum_goes_to_cash_when_btc_below_trend():
    data = trending_market()
    btc = data.bars["BTC-USD"]
    data.bars["BTC-USD"] = btc.iloc[::-1].set_axis(btc.index)
    w = CrossSectionalMomentum(top_n=2, lookback=10, trend_ma=20).generate_targets(data)
    assert w.iloc[-1].sum() == 0


def test_vol_target_breakout_shrinks_positions_in_volatile_markets():
    data = synthetic_market(["A", "B"], n=600, seed=3)
    # No position until the volatility window has filled.
    plain = Breakout(entry=20, exit=10).generate_targets(data).iloc[40:]
    scaled = VolTargetBreakout(entry=20, exit=10, target_vol=0.05).generate_targets(data).iloc[40:]
    held = plain > 0
    assert ((scaled > 0) == held).all().all()
    s, p = scaled.where(held).stack().dropna(), plain.where(held).stack().dropna()
    assert (s <= p + 1e-12).all()
    assert s.mean() < p.mean()


def test_sleeves_split_capital_by_weight():
    data = synthetic_market(["A", "B", "C"], n=300)
    s = Sleeves(sleeves=[(BuyAndHold(), ["A", "B"], 0.9), (BuyAndHold(), ["C"], 0.1)])
    w = s.generate_targets(data).iloc[0]
    assert w.to_dict() == pytest.approx({"A": 0.45, "B": 0.45, "C": 0.1})
    res = Backtester(costs=FREE).run(s, data)
    assert res.weights.iloc[1].sum() == pytest.approx(1.0, abs=1e-6)


def test_bonding_curve_round_trip_loses_only_to_price_impact():
    v = mc_to_v(28.0)
    assert v == pytest.approx(V0_SOL, rel=1e-3)
    tokens, v_after = buy(v, 1.0)
    assert sell(v_after, tokens) == pytest.approx(1.0)
    assert sell(v, tokens) < 1.0
    assert K / v_after < K / v


def test_simulate_calls_resolves_each_exit():
    sol = pd.Series(100.0, index=pd.date_range("2026-10-01", periods=3))
    calls = pd.DataFrame({
        "time": pd.to_datetime(["2026-10-01 10:00", "2026-10-01 11:00", "2026-10-02 12:00"]),
        "mc": [10_000, 10_000, 10_000],
        "peak": [3.0, 1.2, 1.1],
        "graduated": [0, 1, 0],
        "terminal_rs": [0.0, np.nan, 0.0],
    })
    r = simulate_calls(calls, sol, take_profit=2.0, delay_premium=0.0, fee=0.0, tx_cost_usd=0.0)
    assert list(r["exit"]) == ["take_profit", "flat", "terminal"]
    assert 0.9 < r["ret"].iloc[0] < 1.0
    assert -0.05 < r["ret"].iloc[1] < 0
    assert r["ret"].iloc[2] < -0.5
    stopped = simulate_calls(calls, sol, take_profit=2.0, stop_loss=0.5, delay_premium=0.0, fee=0.0, tx_cost_usd=0.0)
    assert stopped["exit"].iloc[2] == "stop"
    assert stopped["ret"].iloc[2] == pytest.approx(-0.5, abs=0.03)
