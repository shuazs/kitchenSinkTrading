import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel, MarketData
from backtester.data.fx import remove_spikes, total_return_index
from backtester.data.synthetic import synthetic_market
from backtester.strategies import BasisCarry, FXCarry, HedgedMomentum
from backtester.strategies.basis import last_fridays

FREE = CostModel(spread_bps=0, slippage_bps=0)


def flat(values, idx):
    return pd.DataFrame({"open": values, "high": values, "low": values, "close": values, "volume": 0.0}, index=idx)


def test_remove_spikes_drops_bad_ticks_but_keeps_real_jumps():
    idx = pd.bdate_range("2020-01-01", periods=6)
    s = pd.Series([1.0, 1.0, 1.2, 1.0, 1.0, 1.2], index=idx)  # spike on day 3, real jump on day 6
    out = remove_spikes(s)
    assert out.iloc[2] == 1.0 and out.iloc[5] == 1.2


def test_total_return_index_adds_interest_to_flat_spot():
    idx = pd.date_range("2020-01-01", periods=366, freq="D")
    tr = total_return_index(pd.Series(1.0, index=idx), pd.Series(0.05, index=idx))
    assert tr.iloc[-1] / tr.iloc[0] == pytest.approx(1.05, rel=1e-3)


def test_fx_carry_longs_high_yielders_and_shorts_low_yielders():
    data = synthetic_market(["HI", "MID", "LO"], n=300, sigma=0.08, seed=4)
    data.extra["rate_gap"] = pd.DataFrame({"HI": 0.04, "MID": 0.0, "LO": -0.03}, index=data.index)
    w = FXCarry(n_long=1, n_short=1, target_vol=0.10).generate_targets(data)
    last = w.iloc[-1]
    assert last["HI"] > 0 and last["LO"] < 0 and last["MID"] == 0
    res = Backtester(costs=FREE, allow_short=True, max_leverage=3).run(FXCarry(n_long=1, n_short=1), data)
    assert res.positions["LO"].iloc[-1] < 0


def test_fx_carry_trend_filter_drops_legs_against_the_trend():
    idx = pd.bdate_range("2020-01-01", periods=200)
    up, down = np.linspace(100, 130, 200), np.linspace(100, 70, 200)
    data = MarketData({"HI": flat(down, idx), "LO": flat(up, idx), "X": flat(up, idx)})
    data.extra["rate_gap"] = pd.DataFrame({"HI": 0.04, "LO": -0.03, "X": 0.0}, index=idx)
    w = FXCarry(n_long=1, n_short=1, trend=63, vol_lookback=20).generate_targets(data)
    # HI is falling and LO rising, so carry says long HI / short LO but trend vetoes both.
    assert (w.iloc[-3:].abs().sum(axis=1) == 0).all()


def test_last_fridays():
    assert list(last_fridays("2024-01-01", "2024-03-01").strftime("%Y-%m-%d")) == ["2024-01-26", "2024-02-23", "2024-03-29"]


def basis_inputs(ann):
    idx = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    exp = last_fridays(idx[0], idx[-1])
    nxt = exp[np.searchsorted(exp.values, idx.values, side="left")]
    df = pd.DataFrame({"expiry": nxt}, index=idx)
    df["dte"] = (df["expiry"] - df.index).dt.days
    df["ann_basis"] = pd.Series(ann, index=idx).where(df["dte"] >= 7)
    return df


def test_basis_carry_earns_basis_on_hedged_share_only_when_above_hurdle():
    inputs = basis_inputs(0.12)
    res = BasisCarry(margin=0.25, roll_cost_bps=0, spot_cost_bps=0, spot_fee=0).run(inputs, cash_rate=0.04)
    years = (res.equity.index[-1] - res.equity.index[0]).days / 365
    growth = (res.equity.iloc[-1] / res.equity.iloc[0]) ** (1 / years) - 1
    assert growth == pytest.approx(0.12 * 0.8, abs=0.005)
    skip = BasisCarry(hurdle=0.10, roll_cost_bps=0, spot_cost_bps=0).run(inputs, cash_rate=0.04)
    years = (skip.equity.index[-1] - skip.equity.index[0]).days / 365
    assert (skip.equity.iloc[-1] / skip.equity.iloc[0]) ** (1 / years) - 1 == pytest.approx(0.04, abs=0.002)


def test_hedged_momentum_picks_winners_and_sizes_hedge_inside_equity():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(0)
    mkt = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
    bars = {"SPY": flat(mkt, idx), "SH": flat(10000 / mkt, idx)}
    for k, drift in enumerate([0.002, 0.001, 0.0, -0.001]):
        bars[f"S{k}"] = flat(mkt * np.cumprod(np.full(400, 1 + drift)), idx)
    data = MarketData(bars)
    strat = HedgedMomentum(universe=["S0", "S1", "S2", "S3"], top_n=2, hedge="SH")
    last = strat.generate_targets(data).iloc[-1]
    assert last["S0"] > 0 and last["S1"] > 0 and last["S2"] == 0 and last["S3"] == 0
    assert last["SH"] > 0 and last.sum() == pytest.approx(1.0)
