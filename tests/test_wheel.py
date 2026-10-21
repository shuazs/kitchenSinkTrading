import numpy as np
import pandas as pd
import pytest

from backtester.strategies.wheel import Wheel, bs_delta, bs_price, monthly_expiries, strike_for_delta

FREE = dict(commission=0.0, spread_pct=0.0, min_spread=0.0)


def frame(prices, iv=0.25, rate=0.0, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"close": prices, "iv": iv, "rate": rate, "dividend": 0.0}, index=idx)


def test_put_call_parity_and_delta():
    S, K, T, r, q, v = 50.0, 48.0, 30 / 365, 0.04, 0.02, 0.3
    c, p = bs_price(S, K, T, r, q, v, True), bs_price(S, K, T, r, q, v, False)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), rel=1e-9)
    K25 = strike_for_delta(S, T, r, q, v, 0.25, call=False, step=0.01)
    assert K25 < S and bs_delta(S, K25, T, r, q, v, False) == pytest.approx(-0.25, abs=0.002)
    Kc = strike_for_delta(S, T, r, q, v, 0.30, call=True, step=0.01)
    assert Kc > S and bs_delta(S, Kc, T, r, q, v, True) == pytest.approx(0.30, abs=0.002)


def test_monthly_expiries_are_third_fridays_or_prior_trading_day():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    idx = idx[idx != "2021-04-16"]  # pretend April's third Friday is a holiday
    exps = monthly_expiries(idx)
    assert pd.Timestamp("2021-01-15") in exps and pd.Timestamp("2021-03-19") in exps
    assert pd.Timestamp("2021-04-15") in exps


def test_rising_market_keeps_premium_without_assignment():
    data = frame(np.linspace(40, 60, 300))
    res = Wheel(**FREE).run(data, 5_000)
    assert (res.positions["XLF"] == 0).all()
    assert res.equity.iloc[-1] > 5_000
    assert len(res.trades) >= 12


def test_crash_assigns_shares_then_sells_calls_and_never_overdraws():
    prices = np.r_[np.full(40, 45.0), np.linspace(45, 30, 20), np.full(120, 30.0)]
    res = Wheel(**FREE).run(frame(prices), 5_000)
    assert res.positions["XLF"].max() == 100
    assert res.trades["symbol"].str.endswith("C").any()
    cash = res.equity - res.positions["XLF"] * frame(prices)["close"]
    assert (cash > -1e-9).all()


def test_account_too_small_sits_in_cash_earning_rate():
    data = frame(np.full(260, 80.0), rate=0.05)
    res = Wheel().run(data, 5_000)
    assert len(res.trades) == 0
    assert res.equity.iloc[-1] == pytest.approx(5_000 * 1.05 ** ((data.index[-1] - data.index[0]).days / 365))


def test_costs_reduce_returns_and_stats_work():
    data = frame(40 + 3 * np.sin(np.arange(500) / 15))
    free = Wheel(**FREE).run(data, 5_000)
    costly = Wheel().run(data, 5_000)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1] and costly.costs.sum() > 0
    stats = costly.stats(benchmark=data["close"])
    assert {"cagr", "sharpe", "max_drawdown", "beta"} <= set(stats.index)
    with pytest.raises(TypeError):
        Wheel(delta=0.2)
