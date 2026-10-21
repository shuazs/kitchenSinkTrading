from statistics import NormalDist

import numpy as np
import pandas as pd

from backtester import kalshi_minute as km


def _close(n=600, seed=1):
    idx = pd.date_range("2025-03-04 14:30", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(5000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))), idx)


def test_trade_implied_recovers_spot_from_consistent_prices():
    close = _close()
    sig = km.realised_sigma(close)
    close_time = pd.Timestamp("2025-03-04 19:30", tz="UTC")
    rows, mk = [], []
    for i, t in enumerate(close.index[100:400:7]):
        tau = (close_time - t - pd.Timedelta(seconds=30)).total_seconds() / 60
        if tau > 60:
            continue
        K = round(close[t] / 5) * 5 + 5
        p = NormalDist().cdf(np.log(close[t] / K) / (sig[t] * np.sqrt(tau)))
        tick = f"KXINXU-25MAR04H1700-T{K}"
        mk.append({"ticker": tick, "yes_sub_title": f"{K} or above", "close_time": close_time})
        rows.append({"ticker": tick, "count": 10, "yes_price": p * 100, "taker_side": "yes",
                     "created_time": t + pd.Timedelta(seconds=30)})
    f = km.trade_implied(pd.DataFrame(rows), pd.DataFrame(mk).drop_duplicates("ticker"), close, pmin=0.01, pmax=0.99)
    assert len(f) > 3
    assert f.z.abs().max() < 1e-6
    assert (f.flow > 0).all()


def test_returns_around_is_forward_looking():
    close = _close()
    f = pd.DataFrame({"z": 1.0, "flow": 1}, index=close.index[100:110])
    R = km.returns_around(f, close)
    t = close.index[100]
    sig = km.realised_sigma(close)[t]
    assert np.isclose(R.loc[t, "fwd1"], np.log(close.iloc[101] / close.iloc[100]) / sig)
    assert np.isclose(R.loc[t, "past1"], np.log(close.iloc[100] / close.iloc[99]) / sig)


def test_zero_dte_proxy_costs_make_flat_market_lose():
    idx = pd.date_range("2025-03-04 15:00", periods=200, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.Series(6000 * np.exp(np.cumsum(rng.normal(0, 1e-5, 200))), idx)
    r = km.zero_dte_proxy(pd.Series(1.0, idx[30:150:10]), close, hold=5)
    assert len(r) > 5
    assert (r.ret < 0).mean() > 0.8
