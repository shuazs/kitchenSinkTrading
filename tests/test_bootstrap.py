import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from backtester.bootstrap import bootstrap_ci, bootstrap_compare, stationary_indices
from backtester.data.synthetic import synthetic_market
from backtester.strategies import BuyAndHold


def _rets(n=1500, mean=0.0, vol=0.01, seed=1):
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mean, vol, n), index=idx)


def _ar1(n=3000, phi=0.5, seed=2):
    e = np.random.default_rng(seed).normal(0, 0.01, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x, index=pd.bdate_range("2010-01-01", periods=n))


def test_indices_shape_range_and_seed():
    a = stationary_indices(100, n_boot=50, block=10, seed=3)
    assert a.shape == (50, 100)
    assert a.min() >= 0 and a.max() < 100
    assert np.array_equal(a, stationary_indices(100, n_boot=50, block=10, seed=3))
    assert not np.array_equal(a, stationary_indices(100, n_boot=50, block=10, seed=4))


def test_blocks_are_consecutive_days_with_the_asked_mean_length():
    n = 500
    idx = stationary_indices(n, n_boot=200, block=20, seed=0)
    continues = (np.diff(idx, axis=1) % n) == 1
    # A step continues the block with probability 1 - 1/block (a jump can also land on the next day by chance).
    assert continues.mean() == pytest.approx(1 - 1 / 20, abs=0.01)
    iid = stationary_indices(n, n_boot=200, block=1, seed=0)
    assert ((np.diff(iid, axis=1) % n) == 1).mean() < 0.02


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        stationary_indices(1)
    with pytest.raises(ValueError):
        stationary_indices(10, block=0.5)
    with pytest.raises(ValueError):
        bootstrap_ci(pd.Series([0.01]))


def test_constant_returns_have_no_uncertainty():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=300))
    ci = bootstrap_ci(r, n_boot=200, periods_per_year=252)
    assert ci.loc["cagr", "estimate"] == pytest.approx(1.001 ** 252 - 1)
    assert ci.loc["cagr", "lo"] == pytest.approx(ci.loc["cagr", "hi"])
    assert ci.loc["max_drawdown", "estimate"] == 0.0
    assert ci.loc["volatility", "hi"] == pytest.approx(0.0, abs=1e-12)


def test_known_drawdown_including_a_first_day_loss():
    r = pd.Series([-0.10, 0.05, -0.20, 0.30], index=pd.bdate_range("2020-01-01", periods=4))
    ci = bootstrap_ci(r, n_boot=50, periods_per_year=252)
    assert ci.loc["max_drawdown", "estimate"] == pytest.approx(0.9 * 1.05 * 0.8 - 1)


def test_estimates_match_engine_summary():
    data = synthetic_market(["AAA"], n=252 * 4, seed=5)
    res = Backtester(5_000).run(BuyAndHold(symbols=["AAA"]), data)
    stats = res.stats()
    ci = bootstrap_ci(res.equity.pct_change().dropna(), n_boot=100)
    assert ci.loc["max_drawdown", "estimate"] == pytest.approx(stats["max_drawdown"])
    # summary() annualises from the equity index, which has one more row than the returns.
    for k in ("volatility", "sharpe"):
        assert ci.loc[k, "estimate"] == pytest.approx(stats[k], rel=2e-3)
    assert ci.loc["cagr", "estimate"] == pytest.approx(stats["cagr"], abs=0.005)


def test_interval_brackets_estimate_and_widens_with_level():
    r = _rets(mean=0.0004)
    narrow = bootstrap_ci(r, n_boot=500, level=0.5)
    wide = bootstrap_ci(r, n_boot=500, level=0.95)
    for k in ("cagr", "volatility", "sharpe"):
        assert wide.loc[k, "lo"] < wide.loc[k, "estimate"] < wide.loc[k, "hi"]
        assert wide.loc[k, "hi"] - wide.loc[k, "lo"] > narrow.loc[k, "hi"] - narrow.loc[k, "lo"]
    assert wide.attrs == {"n_boot": 500, "block": 20, "level": 0.95}


def test_no_edge_interval_covers_zero_and_strong_edge_excludes_it():
    flat = bootstrap_ci(_rets(mean=0.0, seed=4), n_boot=500)
    assert flat.loc["sharpe", "lo"] < 0 < flat.loc["sharpe", "hi"]
    strong = bootstrap_ci(_rets(mean=0.002), n_boot=500)
    assert strong.loc["sharpe", "lo"] > 0
    assert strong.loc["sharpe", "p_positive"] == 1.0


def test_blocks_widen_the_band_when_returns_are_autocorrelated():
    r = _ar1()
    iid = bootstrap_ci(r, n_boot=1000, block=1)
    blocked = bootstrap_ci(r, n_boot=1000, block=20)
    width = lambda ci: ci.loc["cagr", "hi"] - ci.loc["cagr", "lo"]
    assert width(blocked) > 1.3 * width(iid)


def test_compare_with_itself_is_zero():
    r = _rets(mean=0.0003)
    cmp = bootstrap_compare(r, r, n_boot=200)
    assert (cmp[["diff", "lo", "hi", "p_better"]] == 0).all().all()


def test_compare_is_paired_so_a_small_steady_edge_shows():
    bench = _rets(mean=0.0003)
    better = bench + 0.0002  # same noise, 5%/yr more drift
    cmp = bootstrap_compare(better, bench, n_boot=500)
    assert cmp.loc["sharpe", "p_better"] == 1.0
    assert cmp.loc["cagr", "lo"] > 0
    # Separate bands overlap heavily: only the paired test can tell the two apart.
    assert bootstrap_ci(better, n_boot=500).loc["sharpe", "lo"] < bootstrap_ci(bench, n_boot=500).loc["sharpe", "hi"]


def test_compare_aligns_on_common_dates():
    a = _rets(n=600)
    b = _rets(n=600, seed=9).iloc[100:]
    cmp = bootstrap_compare(a, b, n_boot=100)
    assert cmp.attrs["n_obs"] == 500
    assert set(cmp.columns) == {"strategy", "benchmark", "diff", "lo", "hi", "p_better"}
