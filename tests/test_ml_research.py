import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from backtester import Backtester, CostModel, MarketData
from backtester.data.synthetic import synthetic_market
from backtester.ml.labels import entries_and_exits, trade_returns, triple_barrier
from backtester.ml.metalabel import Logistic, MetaFiltered, decide, event_table, held_mask, walk_forward_probs
from backtester.ml.rank import monthly_panel, walk_forward_scores
from backtester.ml.regime import JumpModel, RegimeSwitch, jump_features
from backtester.ml.vol_forecast import ForecastVolTarget, future_log_var, vol_features, walk_forward_vol
from backtester.strategies import Breakout, BuyAndHold

SYMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


@pytest.fixture(scope="module")
def market():
    return synthetic_market(SYMS, n=2200)


def test_entries_and_exits():
    idx = pd.date_range("2020-01-01", periods=8)
    held = pd.DataFrame({"A": [0, 1, 1, 0, 0, 1, 1, 1]}, index=idx).astype(bool)
    t = entries_and_exits(held)
    assert list(t["entry"]) == [idx[1], idx[5]]
    assert list(t["exit"]) == [idx[3], idx[7]]
    assert list(t["open"]) == [False, True]
    opens = pd.DataFrame({"A": [10, 10, 11, 12, 13, 13, 13, 13]}, index=idx, dtype=float)
    # Bought at the open after entry (11), sold at the open after exit (13).
    assert trade_returns(t.iloc[:1], opens).iloc[0] == pytest.approx(13 / 11 - 1)


def test_triple_barrier_hits_first_barrier():
    idx = pd.date_range("2020-01-01", periods=6)
    close = pd.DataFrame({"A": [100, 101, 99, 106, 90, 90]}, index=idx, dtype=float)
    vol = pd.DataFrame({"A": 0.01}, index=idx)
    trades = pd.DataFrame({"entry": [idx[0]], "symbol": ["A"]})
    # width = 1% * sqrt(4) = 2%: stop at -2% would need 98; profit at +4% = 104 on day 3.
    tb = triple_barrier(trades, close, vol, pt=2, sl=1, horizon=4)
    assert tb["tb_label"].iloc[0] == 1 and tb["tb_end"].iloc[0] == idx[3]


def test_meta_filtered_with_every_trade_kept_matches_slots(market):
    uni = list(SYMS)
    primary = Breakout(entry=20, exit=10)
    trades = event_table(market, primary, uni)
    keep = MetaFiltered(primary=primary, universe=uni, trades=trades.assign(mult=1.0), slots=4)
    w = keep.generate_targets(market)
    held = held_mask(primary, market, uni)
    expected = (held.astype(float) / 4).div((held.sum(axis=1) / 4).clip(lower=1), axis=0)
    pd.testing.assert_frame_equal(w[uni], expected, check_names=False)
    skip = MetaFiltered(primary=primary, universe=uni, trades=trades.assign(mult=0.0), slots=4)
    assert skip.generate_targets(market).abs().sum().sum() == 0


def test_event_features_do_not_see_the_future(market):
    uni = list(SYMS)
    trades = event_table(market, Breakout(), uni)
    cut = trades["entry"].iloc[len(trades) // 2]
    bars = {s: df.copy() for s, df in market.bars.items()}
    for df in bars.values():
        df.loc[df.index > cut, ["open", "high", "low", "close"]] *= 2.0
    other = event_table(MarketData(bars), Breakout(), uni)
    cols = ["mom_63", "macd_16_48", "chan_55", "vol_20", "brk_strength", "breadth"]
    a = trades[trades["entry"] <= cut].set_index(["entry", "symbol"])[cols]
    b = other.set_index(["entry", "symbol"]).loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_probs_train_only_on_finished_trades(market):
    trades = event_table(market, Breakout(entry=10, exit=5), list(SYMS))
    seen = []

    class Spy(Logistic):
        def fit(self, X, y):
            seen.append(X["end"].max())
            return super().fit(X, y)

    first = str(trades["entry"].dt.year.min() + 2)
    probs = walk_forward_probs(trades, Spy, f"{first}-01-01", str(trades["entry"].max().date()), min_train=20)
    assert len(probs) and probs["p"].between(0, 1).all()
    years = sorted(probs["entry"].dt.year.unique())
    for y, last_end in zip(years, seen):
        assert last_end < pd.Timestamp(f"{y}-01-01")
    assert set(decide(probs, "skip_q50").unique()) <= {0.0, 1.0}


def test_jump_model_finds_persistent_regimes_and_online_states_do_not_look_ahead():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    calm = np.r_[np.ones(1000), np.zeros(500), np.ones(1000), np.zeros(500)].astype(bool)
    r = np.where(calm, rng.normal(0.0008, 0.007, 3000), rng.normal(-0.0015, 0.025, 3000))
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    f = jump_features(close)
    m = JumpModel(lam=50).fit(f, close.pct_change())
    truth = pd.Series(np.where(calm, 0, 1), index=idx).reindex(f.index)
    assert (m.train_states_ == truth).mean() > 0.9
    on = m.online_states(f)
    on_short = m.online_states(f.iloc[:1800])
    pd.testing.assert_series_equal(on.iloc[:1800], on_short)


def test_regime_switch_holds_bull_strategy_only_in_regime_zero(market):
    idx = market.index
    states = pd.Series(np.where(np.arange(len(idx)) % 200 < 100, 0, 1), index=idx)
    w = RegimeSwitch(states=states, bull=BuyAndHold(symbols=["AAA"])).generate_targets(market)
    assert (w.loc[states == 0, "AAA"] == 1).all() and (w.loc[states == 1].abs().sum(axis=1) == 0).all()


def test_vol_forecasts_are_out_of_sample(market):
    bars = market.bars["AAA"]
    X, y = vol_features(bars), future_log_var(bars, 21)
    f = walk_forward_vol(X, y, "har", "2018-01-01", str(bars.index[-1].date()))
    assert f.index.min() >= pd.Timestamp("2018-01-01") and (f > 0).all()
    # Rewriting data after a date must not change forecasts before that year.
    bars2 = bars.copy()
    cut = pd.Timestamp("2020-06-30")
    bars2.loc[bars2.index > cut, ["open", "high", "low", "close"]] *= np.linspace(1, 3, (bars2.index > cut).sum())[:, None]
    f2 = walk_forward_vol(vol_features(bars2), future_log_var(bars2, 21), "har", "2018-01-01", str(bars.index[-1].date()))
    pd.testing.assert_series_equal(f.loc[:"2019-12-31"], f2.loc[:"2019-12-31"])
    s = ForecastVolTarget(symbol="AAA", leverage=1, target_vol=0.1, forecast=f)
    res = Backtester(5_000, costs=CostModel()).run(s, market, start="2018-01-01")
    assert res.equity.iloc[-1] > 0


def test_monthly_panel_and_ranking(market):
    panel = monthly_panel(market, SYMS)
    assert panel["grade"].dropna().between(0, 4).all()
    s = walk_forward_scores(panel, "ridge", "2018-01-01", str(market.index[-1].date()))
    assert s.index.get_level_values("date").min() >= pd.Timestamp("2018-01-01")
