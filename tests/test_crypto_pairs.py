import numpy as np
import pandas as pd

from backtester.crypto_pairs import (
    adf_t,
    band_positions,
    rolling_pairs,
    rolling_spread_z,
    rotation_returns,
    select_pairs,
    simulate_pair,
    trade_list,
)


def _coint_pair(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.02, n))
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.8 * s[i - 1] + rng.normal(0, 0.02)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    walk = np.cumsum(rng.normal(0, 0.02, n))
    return pd.DataFrame({"Y": np.exp(5 + x + s), "X": np.exp(4 + x), "W": np.exp(3 + walk)}, idx)


def test_adf_separates_stationary_from_walk():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=500))
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.5 * ar[i - 1] + rng.normal()
    assert adf_t(ar) < -5 < adf_t(walk)


def test_band_positions_enter_exit_stop():
    z = [0, -2.5, -1.0, 0.1, 0, 2.2, 5.0, 0, 3.0, 2.0, 0.4]
    pos = band_positions(z, entry=2.0, exit=0.5, stop=4.0)
    assert list(pos) == [0, 1, 1, 0, 0, -1, 0, 0, -1, -1, 0]
    # a trade is never opened beyond the stop
    assert band_positions([4.5], 2.0, 0.5, 4.0)[0] == 0


def test_simulate_pair_no_lookahead_and_costs():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    ry = pd.Series([0, 0.10, 0.0, 0.0, 0.0], idx)
    rx = pd.Series([0, 0.0, 0.05, 0.0, 0.0], idx)
    pos = np.array([1, 1, 0, 0, 0])
    res = simulate_pair(ry, rx, pos, 1.0, cost_bps=0)
    # signal at bar 0 is held from bar 1: +0.5 * 10% on bar 1, then flat after bar 2
    assert np.isclose(res["gross"].iloc[1], 0.05)
    assert np.isclose(res["gross"].iloc[2], -0.025)
    assert res["gross"].iloc[3:].abs().sum() == 0
    res = simulate_pair(ry, rx, pos, 1.0, cost_bps=100)
    # gross 1 in and 1 out: two fills of 1% each
    assert np.isclose(res["cost"].sum(), 0.02)
    t = trade_list(res)
    assert len(t) == 1 and np.isclose(t["net"].iloc[0], 0.05 - 0.025 - 0.02)


def test_rotation_long_only():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    ry = pd.Series([0, 0.1, 0.1, 0.1], idx)
    rx = pd.Series([0, 0.0, 0.0, 0.0], idx)
    res = rotation_returns(ry, rx, np.array([0, 1, -1, 0]), cost_bps=0)
    assert (res[["wy", "wx"]] >= 0).all().all()
    assert np.allclose(res["gross"], [0, 0.05, 0.1, 0.0])


def test_rolling_spread_z_uses_past_only():
    df = _coint_pair()
    lp = np.log(df)
    a = rolling_spread_z(lp["Y"], lp["X"], 60, 30)
    lp2 = lp.copy()
    lp2.iloc[400:] *= 3
    b = rolling_spread_z(lp2["Y"], lp2["X"], 60, 30)
    pd.testing.assert_series_equal(a["z"].iloc[:400], b["z"].iloc[:400])


def test_select_and_rolling_pairs_find_cointegrated_pair():
    df = _coint_pair()
    picks = select_pairs(np.log(df.iloc[:200]), n_pairs=3)
    assert picks and {picks[0]["y"], picks[0]["x"]} == {"Y", "X"}
    out = rolling_pairs(df, formation=150, trading=30, n_pairs=1, cost_bps=0)
    assert out["returns"].index[0] == df.index[150]
    assert len(out["trades"]) > 3 and out["gross"].sum() > 0
