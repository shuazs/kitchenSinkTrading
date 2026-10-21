import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from backtester import Backtester, CostModel, MarketData
from backtester.data.synthetic import synthetic_market
from backtester.ml import FEATURES, build_panel, holdout_fold, walk_forward_folds
from backtester.ml.contest import deflated_sharpe, out_of_sample_positions, positions_frame
from backtester.ml.models import EqualWeight, RidgeModel
from backtester.strategies import SignalBlend

SYMS = ("AAA", "BBB", "CCC", "DDD")


@pytest.fixture(scope="module")
def market():
    return synthetic_market(SYMS, n=2200)


@pytest.fixture(scope="module")
def panel(market):
    return build_panel(market, SYMS)


def test_panel_features_do_not_see_the_future(market, panel):
    cut = panel.index.get_level_values("date")[len(panel) // 2]
    bars = {}
    for s, df in market.bars.items():
        df = df.copy()
        later = df.index > cut
        df.loc[later, ["open", "high", "low", "close"]] *= 3.0  # rewrite the future
        bars[s] = df
    other = build_panel(MarketData(bars), SYMS)
    upto = panel.index.get_level_values("date") <= cut
    a = panel[upto][FEATURES]
    b = other.loc[a.index, FEATURES]
    pd.testing.assert_frame_equal(a, b)
    # Labels before the cut week are untouched; the cut week's label looks ahead by design.
    before = panel.index.get_level_values("date") < cut
    pd.testing.assert_series_equal(panel[before]["ret"].iloc[:-len(SYMS)], other.loc[panel[before].index, "ret"].iloc[:-len(SYMS)])


def test_panel_target_is_next_week_return(market, panel):
    d = panel.index.get_level_values("date").unique()
    close = market.field("close")["AAA"]
    expect = close.loc[d[11]] / close.loc[d[10]] - 1
    assert panel.loc[(d[10], "AAA"), "ret"] == pytest.approx(expect)
    assert panel.loc[(d[-1], "AAA"), ["ret", "target"]].isna().all()


def test_walk_forward_folds_are_purged_and_ordered(panel):
    dates = panel.index.get_level_values("date").unique()
    folds = walk_forward_folds(dates, "2018-01-01", "2020-06-30", purge=2)
    assert [f.name for f in folds] == ["2018", "2019", "2020"]
    for f in folds:
        assert f.train.max() < f.val.min() and f.val.max() < f.test.min()
        between = dates[(dates > f.val.max()) & (dates < f.test.min())]
        assert len(between) == 2
        between = dates[(dates > f.train.max()) & (dates < f.val.min())]
        assert len(between) == 2
    assert folds[-1].test.max() <= pd.Timestamp("2020-06-30")
    assert len(folds[1].train) > len(folds[0].train)  # expanding window


def test_holdout_fold_trains_only_before_the_holdout(panel):
    dates = panel.index.get_level_values("date").unique()
    f = holdout_fold(dates, "2020-01-01")
    assert f.fit_all.max() < pd.Timestamp("2020-01-01") <= f.test.min()


def _planted(panel, seed=0):
    """Panel whose target is driven by mom_63, so a model has something to find."""
    rng = np.random.default_rng(seed)
    p = panel.copy()
    z = (p["mom_63"] - p["mom_63"].mean()) / p["mom_63"].std()
    p["target"] = (0.5 * z + rng.normal(0, 1, len(p))).where(p["target"].notna())
    return p


def test_ridge_finds_a_planted_signal(panel):
    p = _planted(panel)
    folds = walk_forward_folds(p.index.get_level_values("date"), "2018-01-01", "2020-06-30")
    pos = out_of_sample_positions(p, RidgeModel, folds)
    assert pos.corr(p["target"].reindex(pos.index)) > 0.3


def test_equal_weight_is_mean_of_signed_signals(panel):
    folds = walk_forward_folds(panel.index.get_level_values("date"), "2019-01-01", "2019-12-31")
    m = EqualWeight().fit(panel, folds[0])
    pos = m.predict(panel, folds[0].test)
    assert pos.notna().all() and np.isclose(pos.std(), 1.0, atol=0.5)


def test_lightgbm_runs(panel):
    pytest.importorskip("lightgbm")
    from backtester.ml.models import GradientBoosting

    p = _planted(panel)
    folds = walk_forward_folds(p.index.get_level_values("date"), "2019-01-01", "2019-12-31")
    pos = GradientBoosting().fit(p, folds[0]).predict(p, folds[0].test)
    assert pos.corr(p["target"].reindex(pos.index)) > 0.2


def test_nets_run_and_give_one_column_per_seed(panel):
    pytest.importorskip("torch")
    from backtester.ml.models import LSTMMomentum, MLP

    p = _planted(panel)
    f = walk_forward_folds(p.index.get_level_values("date"), "2019-01-01", "2019-12-31")[0]
    mlp = MLP(seeds=2, epochs=5).fit(p, f)
    seeds = mlp.predict_seeds(p, f.test)
    assert seeds.shape == (len(f.test) * len(SYMS), 2)
    pd.testing.assert_series_equal(seeds.mean(axis=1), mlp.predict(p, f.test), check_names=False)
    lstm = LSTMMomentum(seeds=1, epochs=3, window=8).fit(p, f)
    pos = lstm.predict(p, f.test)
    assert len(pos) == len(f.test) * len(SYMS) and pos.notna().all()


def test_signal_blend_top_n_picks_the_best_scores(market, panel):
    dates = panel.index.get_level_values("date").unique()[-30:]
    scores = pd.DataFrame({"AAA": 2.0, "BBB": 1.0, "CCC": -1.0, "DDD": 0.5}, index=dates)
    w = SignalBlend(positions=scores, mode="top_n", top_n=2).generate_targets(market)
    assert (w["AAA"] == 0.5).all() and (w["BBB"] == 0.5).all() and (w[["CCC", "DDD"]] == 0).all().all()


def test_signal_blend_risk_parity_is_long_only_and_capped(market, panel):
    dates = panel.index.get_level_values("date").unique()[-60:]
    scores = pd.DataFrame({"AAA": 5.0, "BBB": -1.0, "CCC": 0.5, "DDD": 1.0}, index=dates)
    w = SignalBlend(positions=scores, max_gross=1.2).generate_targets(market)
    assert (w["BBB"] == 0).all()
    assert (w.abs().sum(axis=1) <= 1.2 + 1e-9).all()
    assert w.index[0] == dates[0]
    res = Backtester(5_000, costs=CostModel(spread_bps=0, slippage_bps=0), max_leverage=1.2).run(
        SignalBlend(positions=scores, max_gross=1.2), market, start=dates[0])
    assert res.equity.iloc[-1] > 0


def test_positions_frame_unstacks_symbols(panel):
    s = pd.Series(1.0, index=panel.index[:8])
    assert set(positions_frame(s).columns) <= set(SYMS)


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 2000), index=pd.bdate_range("2015-01-01", periods=2000))
    one, many = deflated_sharpe(r, 1), deflated_sharpe(r, 100)
    assert 0 <= many < one <= 1
    assert one > 0.9


def test_kalshi_weekly_features_sum_changes_and_lag():
    from backtester.data.kalshi import KALSHI_FEATURES, weekly_features

    days = pd.date_range("2023-01-02", "2023-01-20", freq="D")
    daily = pd.DataFrame({"cpi_chg": 0.0, "recession_p": 0.3}, index=days)
    daily.loc["2023-01-05", "cpi_chg"] = 2.0
    daily.loc["2023-01-13", "cpi_chg"] = 50.0  # clipped to 10
    daily.loc["2023-01-13":, "recession_p"] = 0.5  # from a Friday on: lagged into the next week
    weeks = pd.DatetimeIndex(["2022-12-30", "2023-01-06", "2023-01-13", "2023-01-20", "2023-03-31"])
    w = weekly_features(daily, weeks)
    assert list(w.columns) == KALSHI_FEATURES
    assert w.loc["2023-01-06", "k_cpi"] == 2.0
    assert w.loc["2023-01-13", "k_cpi"] == 0.0 and w.loc["2023-01-20", "k_cpi"] == 10.0
    assert w.loc["2023-01-13", "k_recession_p"] == 0.3 and w.loc["2023-01-20", "k_recession_chg"] == pytest.approx(0.2)
    assert w.loc[["2022-12-30", "2023-03-31"]].isna().all().all()  # outside the file's history


def test_extra_features_join_the_panel_and_models_can_use_them(market, panel):
    weeks = panel.index.get_level_values("date").unique()
    extra = pd.DataFrame({"x_new": np.arange(len(weeks), dtype=float)}, index=weeks)
    p = build_panel(market, SYMS, extra=extra.iloc[::2])  # every other week missing -> 0
    assert "x_new" in p and (p["x_new"] == 0).any() and p["x_new"].notna().all()
    folds = walk_forward_folds(p.index.get_level_values("date"), "2019-01-01", "2019-12-31")
    m = RidgeModel(features=FEATURES + ["x_new"]).fit(p, folds[0])
    assert m.model.coef_.shape == (len(FEATURES) + 1,)
