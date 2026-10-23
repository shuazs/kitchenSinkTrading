import numpy as np
import pandas as pd
import pytest

from backtester import COSTS, Backtester
from backtester.combo_search import (Moments, SleeveSearch, TimingSearch, combine, deflated_sharpe, evaluate,
                                     half_year_blocks, open_to_open, pbo, rule_specs, subset_masks, timing_returns,
                                     unit_cost)
from backtester.data.synthetic import synthetic_market
from backtester.strategies import TrendFilter


def test_subset_masks_and_rules_count_every_distinct_combination():
    m = subset_masks(4)
    assert len(m) == 15 and len({tuple(r) for r in m}) == 15
    # 4 singles, 6 pairs x 3 rules, 5 larger subsets x 4 rules
    assert len(rule_specs(m)) == 4 + 18 + 20


def test_combine_rules():
    s = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]])
    mask = np.array([[True, True, True]] * 4)
    out = combine(s, mask, np.array(["and", "or", "majority", "share"]))
    np.testing.assert_allclose(out[:, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(out[:, 1], [1, 1, 0, 1])
    np.testing.assert_allclose(out[:, 2], [1, 0, 0, 1])
    np.testing.assert_allclose(out[:, 3], [2 / 3, 1 / 3, 0, 1])


def test_timing_returns_match_the_engine():
    d = synthetic_market(["AAA"], n=1200, seed=3)
    strat = TrendFilter(risk_on="AAA", signal="AAA", ma=50)
    eng = Backtester(5_000, costs=COSTS["etf"]).run(strat, d).equity
    on = strat.generate_targets(d)["AAA"].reindex(d.index).ffill().fillna(0.0).to_numpy()[:, None]
    r = open_to_open(d.bars["AAA"], d.index).fillna(0.0).to_numpy()
    net = timing_returns(on, r, np.zeros(len(r)), unit_cost(COSTS["etf"]))[:, 0]
    # The engine marks at the close, this at the open: compare total growth.
    assert np.prod(1 + net) == pytest.approx(eng.iloc[-1] / 5_000, rel=0.02)


def test_moments_sharpe_matches_direct():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    x = rng.normal(0.0005, 0.01, (1000, 3))
    blocks, names = half_year_blocks(idx)
    mom = Moments.from_returns(x, blocks, names)
    sel = blocks < 3
    direct = x[sel].mean(0) / x[sel].std(0) * np.sqrt(252)
    np.testing.assert_allclose(mom.sharpe(np.arange(3)), direct, rtol=1e-6)


def test_overfitting_checks_flag_noise_and_pass_a_real_edge():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=2520)
    blocks, names = half_year_blocks(idx)
    noise = rng.normal(0, 0.01, (2520, 300))
    mom = Moments.from_returns(noise, blocks, names)
    assert pbo(mom, np.arange(len(names)), 100)["pbo"] > 0.3
    sr = mom.sharpe(np.arange(len(names)))
    lucky = noise[:, int(np.argmax(sr))]
    assert deflated_sharpe(lucky, sr)["dsr"] < 0.5
    real = noise.copy()
    real[:, 0] += 0.0015  # Sharpe ~2.4
    mom = Moments.from_returns(real, blocks, names)
    sr = mom.sharpe(np.arange(len(names)))
    assert pbo(mom, np.arange(len(names)), 100)["pbo"] < 0.1
    assert deflated_sharpe(real[:, 0], sr)["dsr"] > 0.95


def test_searches_run_end_to_end():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2012-01-01", "2016-12-31")
    sig = pd.DataFrame(rng.random((len(idx), 4)) > 0.3, index=idx, columns=list("abcd"))
    ret = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    cash = pd.Series(0.0, index=idx)
    ts = TimingSearch(sig, {"X": (ret, 0.0003, None)}, cash)
    res = evaluate(ts, "2016-07-01", min_train_years=2, pbo_splits=20)
    assert len(ts) == 42 and len(res["wf_picks"]) == 5
    assert res["wf_returns"].index.min() >= pd.Timestamp("2014-01-01")
    assert res["wf_returns"].index.max() < pd.Timestamp("2016-07-01")
    sleeves = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), 5)), index=idx, columns=list("vwxyz"))
    ss = SleeveSearch(sleeves, cash)
    assert len(ss) == 2 * 31
    single = ss.specs.index[(ss.specs["size"] == 1) & (ss.specs.signals == "v")][0]
    np.testing.assert_allclose(ss.returns(int(single)), sleeves["v"])
    evaluate(ss, "2016-07-01", min_train_years=2, pbo_splits=20)
