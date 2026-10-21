import math

import numpy as np
import pandas as pd
import pytest

from backtester.strategies.spreads import CreditSpreads, smile_iv, strike_for_delta_smile, term_factor
from backtester.strategies.wheel import bs_delta

FREE = dict(commission=0.0, spread_pct=0.0, min_spread=0.0)
FLAT = dict(atm_mult=1.0, skew_scale=0.0)


def frame(prices, iv=0.20, rate=0.0, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"close": prices, "iv": iv, "rate": rate, "dividend": 0.0}, index=idx)


def gbm(n, vol, seed, s0=100.0, drift=0.0):
    rng = np.random.default_rng(seed)
    steps = rng.normal((drift - 0.5 * vol**2) / 252, vol / math.sqrt(252), n)
    return s0 * np.exp(np.cumsum(steps))


def test_term_factor_flat_and_interpolated():
    assert term_factor({30: 0.2}, 45) == 1.0
    curve = {9: 0.15, 30: 0.20, 91: 0.25}
    assert term_factor(curve, 30) == pytest.approx(1.0)
    assert 1.0 < term_factor(curve, 60) < 1.25
    assert term_factor(curve, 9) == pytest.approx(0.75)


def test_smile_puts_rich_calls_cheap_and_delta_strikes():
    S, T, atm = 100.0, 40 / 365, 0.18
    vol_at = lambda K: smile_iv(S, K, T, atm, 0.344, 0.016, 0.164)
    assert vol_at(90) > vol_at(95) > atm > vol_at(105)
    K = strike_for_delta_smile(S, T, 0.0, 0.0, 0.25, False, 0.5, vol_at)
    assert K < S and abs(bs_delta(S, K, T, 0, 0, vol_at(K), False)) == pytest.approx(0.25, abs=0.02)


def test_rising_market_keeps_credit_and_risk_is_capped():
    data = frame(np.linspace(100, 130, 400))
    strat = CreditSpreads(**FREE, **FLAT, rebase=False, width_pct=0.05)
    res = strat.run(data, 5_000)
    c = strat.closed_
    assert len(c) >= 10 and (c["pnl"] > 0).all()
    assert res.equity.iloc[-1] > 5_000
    assert (c["max_loss"] * c["contracts"] <= 0.10 * 6_000).all()


def test_crash_loses_at_most_the_width_and_equity_stays_positive():
    prices = np.r_[np.full(30, 100.0), np.linspace(100, 40, 40), np.full(200, 40.0)]
    for structure in ("put_spread", "iron_condor"):
        strat = CreditSpreads(structure=structure, max_positions=3, entry_spacing=2, risk_per_trade=0.5,
                              rebase=False, width_pct=0.05, **FLAT)
        res = strat.run(frame(prices, iv=0.3), 5_000)
        c = strat.closed_
        width = 5.0
        assert (c["exit_value"] <= width + 1e-9).all()
        assert (res.equity > 0).all()
        assert c["pnl"].min() < 0


def test_premium_over_realized_vol_pays_and_under_it_loses():
    """Implied 20% vs realized 12% should earn money; implied 20% vs realized 32% should lose."""
    out = {}
    for name, realized in (("rich", 0.12), ("cheap", 0.32)):
        pnl = 0.0
        for seed in range(6):
            data = frame(gbm(1500, realized, seed), iv=0.20)
            strat = CreditSpreads(**FREE, **FLAT, rebase=False, take_profit=1.0, exit_dte=None)
            strat.run(data, 100_000)
            pnl += strat.closed_["pnl"].sum()
        out[name] = pnl
    assert out["rich"] > 0 > out["cheap"]


def test_costs_rebase_and_params():
    data = frame(100 + 5 * np.sin(np.arange(600) / 20))
    free = CreditSpreads(**FREE).run(data, 5_000)
    costly = CreditSpreads().run(data, 5_000)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1] and costly.costs.sum() > 0
    stats = costly.stats(benchmark=data["close"])
    assert {"cagr", "sharpe", "max_drawdown"} <= set(stats.index)
    with pytest.raises(TypeError):
        CreditSpreads(delta=0.2)
    with pytest.raises(ValueError):
        CreditSpreads(structure="strangle")
