import numpy as np
import pandas as pd
import pytest

from backtester import shortterm as st


def hourly_bars(days=60, seed=0, drift=0.0):
    """yfinance-style RTH hourly bars (09:30 ... 15:30 starts), New York tz."""
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        idx += [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=i) for i in range(7)]
    idx = pd.DatetimeIndex(idx).tz_localize(st.NY)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.003, len(idx))))
    return pd.DataFrame({"Open": np.r_[100, close[:-1]], "High": close, "Low": close, "Close": close,
                         "Volume": 1.0}, index=idx)


def test_bar_end_closes_maps_last_bar_to_the_close():
    s = st.bar_end_closes(hourly_bars(2))
    assert s.index[0].strftime("%H:%M") == "10:30"
    assert s.index[6].strftime("%H:%M") == "16:00"


def test_panel_targets_and_features_line_up():
    b = hourly_bars(30)
    P = st.hourly_panel(b, vix=hourly_bars(30, seed=1), vix9d=hourly_bars(30, seed=2))
    c = st.bar_end_closes(b)
    row = P.iloc[20]
    t = P.index[20]
    assert row.price == pytest.approx(c[t])
    close = c[t.normalize() + pd.Timedelta("16h")]
    assert row.y_close == pytest.approx(np.log(close / row.price))
    assert row.minutes_left == 16 * 60 - (t.hour * 60 + t.minute)
    prev = c[c.index.normalize() < t.normalize()].iloc[-1]
    assert row.day_ret == pytest.approx(np.log(row.price / prev))


def test_walk_forward_never_trains_on_test_month():
    P = st.hourly_panel(hourly_bars(120))
    seen = []

    def fit(X, y):
        seen.append(X.index.max())
        return lambda Xn: np.zeros(len(Xn))
    oos = st.walk_forward(P, ["day_ret", "hour"], "y_close", fit, "2024-03-01", min_train=50)
    assert len(oos) and len(seen) == oos.index.tz_localize(None).to_period("M").nunique()
    months = sorted(oos.index.tz_localize(None).to_period("M").unique())
    for last_train, m in zip(seen, months):
        assert last_train.tz_localize(None) < m.start_time


def test_bs_put_call_parity_and_expiry():
    S, K = 600.0, 601.0
    c, p = st.bs_price(S, K, 120, 0.15, True), st.bs_price(S, K, 120, 0.15, False)
    assert c - p == pytest.approx(S - K, abs=1e-9)
    assert st.bs_price(S, K, 0, 0.15, False) == 1.0


def test_zero_dte_signal_and_against_are_mirror_trades():
    P = st.hourly_panel(hourly_bars(40), vix9d=hourly_bars(40, seed=3)).dropna(subset=["vix9d", "close"])
    P["vix9d"] = 15.0
    P["pred"] = np.sign(P.y_close)  # perfect foresight
    for q in (50, 80, 90):
        P[f"train_q{q}"] = 0.0
    kw = dict(hours=("10:30",), capital=1e7, risk_frac=0.001)  # sizing never skips a day
    win = st.zero_dte(P, 0.6, **kw)
    lose = st.zero_dte(P, 0.6, mode="against", **kw)
    assert len(win) == len(lose) > 0
    assert (win.side != lose.side).all()
    assert win.ret_on_premium.mean() > 0 > lose.ret_on_premium.mean()


def test_fair_iv_mult_breaks_even():
    P = st.hourly_panel(hourly_bars(80), vix9d=hourly_bars(80, seed=4)).dropna(subset=["close"])
    P["vix9d"] = 20.0
    m = st.fair_iv_mult(P)
    K = P.price.round()
    cost = np.mean([st.bs_price(S, k, mn, m * 0.2, True) + st.bs_price(S, k, mn, m * 0.2, False)
                    for S, k, mn in zip(P.price, K, P.minutes_left)])
    assert cost == pytest.approx(np.abs(P.close - K).mean(), rel=1e-3)
