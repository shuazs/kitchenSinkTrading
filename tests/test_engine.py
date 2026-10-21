import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel, MarketData, Strategy, sweep, walk_forward
from backtester.data.synthetic import synthetic_bars, synthetic_market
from backtester.strategies import Breakout, BuyAndHold, DualMomentum, RSIMeanReversion, TrendFilter

FREE = CostModel(spread_bps=0, slippage_bps=0)


class OneSignal(Strategy):
    """Go all in on `day` (signal at that close)."""

    params = {"day": 10}

    def generate_targets(self, data):
        w = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        w.iloc[self.p["day"] :, 0] = 1.0
        return w


def test_buy_and_hold_matches_price_without_costs():
    data = synthetic_market(["A"], n=500)
    res = Backtester(10_000, costs=FREE).run(BuyAndHold(), data)
    bars = data.bars["A"]
    expected = 10_000 * bars["close"].iloc[-1] / bars["open"].iloc[1]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_signal_executes_next_open():
    data = synthetic_market(["A"], n=50)
    res = Backtester(costs=FREE).run(OneSignal(day=10), data)
    first = res.trades.iloc[0]
    assert first["date"] == data.index[11]
    assert first["fill_price"] == pytest.approx(data.bars["A"]["open"].iloc[11])


def test_costs_reduce_returns_and_are_reported():
    data = synthetic_market(["A"], n=300)
    free = Backtester(costs=FREE).run(RSIMeanReversion(trend_ma=20), data)
    costly = Backtester(costs=CostModel(pct_fee=0.004, spread_bps=20, slippage_bps=10)).run(
        RSIMeanReversion(trend_ma=20), data
    )
    assert len(free.trades) > 5
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.costs.sum() > 0


def test_whole_shares_and_no_overdraft():
    bars = synthetic_bars(n=100, start_price=1_000)
    data = MarketData({"X": bars})
    bt = Backtester(5_000, fractional=False, costs=CostModel(commission=1))
    res = bt.run(BuyAndHold(), data)
    held = res.positions["X"].iloc[-1]
    assert held == np.floor(held) and held <= 5
    cash = res.equity - res.positions["X"] * bars["close"]
    assert (cash >= -1e-6).all()


def test_short_clipped_unless_allowed():
    class Short(Strategy):
        def generate_targets(self, data):
            return pd.DataFrame(-1.0, index=data.index[:1], columns=data.symbols)

    data = synthetic_market(["A"], n=50)
    assert Backtester().run(Short(), data).trades.empty
    assert (Backtester(allow_short=True).run(Short(), data).positions["A"].iloc[-1]) < 0


def test_cash_rate_accrues():
    class Cash(Strategy):
        def generate_targets(self, data):
            return pd.DataFrame(0.0, index=data.index[:1], columns=data.symbols)

    data = synthetic_market(["A"], n=261, freq="B")
    res = Backtester(1_000, cash_rate=0.05).run(Cash(), data)
    assert res.equity.iloc[-1] == pytest.approx(1_000 * 1.05 ** ((data.index[-1] - data.index[0]).days / 365))


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        TrendFilter(moving_average=50)


@pytest.mark.parametrize(
    "strategy",
    [
        TrendFilter(risk_on="AAA", risk_off="BBB", ma=50),
        Breakout(entry=20, exit=10),
        Breakout(entry=20, exit=10, max_positions=1),
        RSIMeanReversion(),
        DualMomentum(safe="CCC", lookback=126),
    ],
)
def test_strategies_run_and_respect_weights(strategy):
    data = synthetic_market(["AAA", "BBB", "CCC"], n=800)
    res = Backtester().run(strategy, data)
    assert res.weights.sum(axis=1).max() <= 1.01
    stats = res.stats(benchmark=data.bars["AAA"]["close"])
    assert np.isfinite(stats["final_equity"])


def test_sweep_and_walk_forward():
    data = synthetic_market(["AAA"], n=252 * 8)
    bt = Backtester()
    table = sweep(Breakout, {"entry": [10, 20], "exit": [5, 10]}, data, bt)
    assert table.attrs["n_variants"] == 4
    equity, picks = walk_forward(Breakout, {"entry": [10, 20]}, data, bt, train_years=3, test_years=1, exit=10)
    assert len(picks) >= 3 and equity.index.is_monotonic_increasing


def test_on_change_skips_drift_trades():
    data = synthetic_market(["A", "B"], n=300)
    hold = BuyAndHold()
    on_change = Backtester(costs=FREE).run(hold, data)
    daily = Backtester(costs=FREE, rebalance="daily").run(hold, data)
    assert len(on_change.trades) == 2
    assert len(daily.trades) > 100
