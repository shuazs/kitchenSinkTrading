import numpy as np
import pandas as pd
import pytest

norm = pytest.importorskip("scipy.stats").norm

from backtester import kalshi_vol as kv


def test_ladder_quantiles_recover_normal_quartiles():
    K = np.arange(4900, 5101, 10.0)
    p_above = 1 - norm.cdf((K - 5000) / 40)
    q25, q50, q75 = kv.ladder_quantiles(K, p_above)
    assert q50 == pytest.approx(5000, abs=1)
    assert (q75 - q25) / 40 == pytest.approx(kv.IQR_TO_SD, rel=0.03)


def test_ladder_quantiles_nan_outside_strikes():
    K = np.array([5000.0, 5010.0, 5020.0])
    assert np.isnan(kv.ladder_quantiles(K, np.array([0.6, 0.5, 0.45]))[0])


def test_iron_fly_credit_bounds_and_payoff():
    credit = kv.iron_fly_price(500, 500, 3, 200, 0.15)
    assert 0 < credit < 3
    assert kv.iron_fly_payoff(500.5, 500, 3) == pytest.approx(0.5)
    assert kv.iron_fly_payoff(520, 500, 3) == 3


def _panel(n=120, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n)
    idx = days + pd.Timedelta(hours=12, minutes=30)
    vix = rng.uniform(10, 20, n)
    sd = vix / 100 * np.sqrt(210 / kv.MIN_PER_YEAR)
    price = np.full(n, 5000.0)
    close = price * np.exp(rng.normal(0, sd))
    half = 0.674 * sd * price * rng.uniform(0.7, 1.3, n)
    return pd.DataFrame({"price": price, "close": close, "minutes_left": 210, "rv": sd ** 2,
                         "q25": price - half, "q50": price, "q75": price + half, "n_strikes": 10,
                         "vix1d": vix}, index=idx)


def test_signal_uses_only_earlier_days():
    P = kv.add_signal(_panel())
    assert not P.low.iloc[:60].any()
    # changing the last day must not change earlier flags
    Q = _panel(); Q.iloc[-1, Q.columns.get_loc("q75")] *= 1.5
    assert (kv.add_signal(Q).low.iloc[:-1] == P.low.iloc[:-1]).all()


def test_fair_mult_near_one_when_vix_is_true_vol():
    P = _panel(n=2000)
    assert kv.fair_iv_mult(P) == pytest.approx(1.0, abs=0.1)


def test_backtest_sizes_to_risk_and_tracks_equity():
    P = kv.add_signal(_panel())
    tr = kv.iron_fly_backtest(P, 1.0, rule="always", times=("12:30",))
    assert 0 < len(tr) <= 60  # skips days once equity cannot fund one fly
    assert np.allclose(tr.equity.diff().iloc[1:], tr.pnl.iloc[1:])
    assert (tr.pnl >= -0.1 * 5_000 * 3).all()
