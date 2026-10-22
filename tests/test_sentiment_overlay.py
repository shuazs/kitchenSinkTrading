import numpy as np
import pandas as pd

from backtester import Backtester, CostModel
from backtester.data import sentiment_indices as si
from backtester.data.synthetic import synthetic_market
from backtester.sentiment_overlay import SentimentOverlay, bucket_returns, threshold_mult
from backtester.strategies import BuyAndHold, VolTarget

FREE = CostModel(spread_bps=0, slippage_bps=0)


def mood_closes(n=600, seed=0):
    data = synthetic_market(si.MOOD_SYMBOLS, seed=seed, n=n)
    close = data.field("close")
    close["^VIX"] = 15 + 5 * np.sin(np.arange(n) / 20)
    close["^VIX3M"] = 17.0
    return close


def test_mood_index_is_bounded_and_ignores_the_future():
    close = mood_closes()
    mood = si.mood_index(close)
    assert mood.between(0, 100).all() and len(mood) > 300
    changed = close.copy()
    changed.iloc[450:] *= 1.5
    assert np.allclose(si.mood_index(changed).loc[: close.index[449]], mood.loc[: close.index[449]])


def test_mood_term_component_reads_backwardation_as_fear():
    close = mood_closes()
    comp = si.mood_components(close)
    assert (comp["term"][close["^VIX"] > close["^VIX3M"]] < 0).all()


def test_daily_funding_sums_the_three_prints():
    ts = pd.date_range("2024-01-01", periods=6, freq="8h") + pd.Timedelta("3ms")
    raw = pd.DataFrame({"timestamp": ts, "funding_rate": [1e-4, 2e-4, 3e-4, -1e-4, 0.0, 1e-4]})
    daily = si.daily_funding(raw)
    assert list(daily.index) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert np.allclose(daily, [6e-4, 0.0])


def test_consumer_sentiment_is_lagged(monkeypatch):
    s = pd.Series([70.0, 80.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    monkeypatch.setattr("backtester.data.fred.load_series", lambda _id: s.copy())
    out = si.consumer_sentiment(lag_months=1)
    assert out.loc["2024-02-01"] == 70.0 and out.loc["2024-03-01"] == 80.0


def test_bucket_returns_finds_a_signal_that_knows_the_future():
    data = synthetic_market(["SPY"], n=800)
    close = data.bars["SPY"]["close"]
    peek = np.log(close.shift(-5) / close)  # cheating signal: the forward return itself
    t = bucket_returns(peek, close, 5)
    assert t["mean_pct"].is_monotonic_increasing
    assert t["mean_pct"].iloc[0] < 0 < t["mean_pct"].iloc[-1]
    assert t["days"].sum() == peek.notna().sum()


def test_threshold_mult():
    s = pd.Series([5.0, 50.0, 95.0, np.nan])
    assert list(threshold_mult(s, low=10, high=90, low_mult=2, high_mult=0.5)) == [2, 1, 0.5, 1]


def test_overlay_with_unit_multiplier_matches_base():
    data = synthetic_market(["SPY", "UPRO"], n=300)
    data.extra["m"] = pd.Series(1.0, index=data.index)
    base = VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.2)
    bt = Backtester(costs=FREE)
    a = bt.run(base, data).equity
    b = bt.run(SentimentOverlay(base=base, mult_key="m", symbols=["UPRO"]), data).equity
    assert np.allclose(a, b)


def test_overlay_scales_moves_rest_to_risk_off_and_caps():
    data = synthetic_market(["SPY", "SHY"], n=60)
    mult = pd.Series(1.0, index=data.index)
    mult.iloc[20:30] = 0.25
    mult.iloc[40:] = 3.0
    data.extra["m"] = mult
    w = SentimentOverlay(base=BuyAndHold(symbols=["SPY"]), mult_key="m", symbols=["SPY"],
                         risk_off="SHY").generate_targets(data)
    assert np.allclose(w["SPY"].iloc[20:30], 0.25) and np.allclose(w["SHY"].iloc[20:30], 0.75)
    assert np.allclose(w["SPY"].iloc[40:], 1.0) and np.allclose(w["SHY"].iloc[40:], 0.0)


def test_overlay_trades_the_open_after_the_signal():
    data = synthetic_market(["SPY"], n=60)
    sig = pd.Series(50.0, index=data.index)
    sig.iloc[30:] = 5.0  # extreme fear from the close of bar 30
    data.extra["mood"] = sig
    strat = SentimentOverlay(base=BuyAndHold(symbols=["SPY"]), signal_key="mood", low=10, low_mult=0.0)
    res = Backtester(costs=FREE).run(strat, data)
    pos = res.positions["SPY"]
    assert pos.iloc[30] > 0 and pos.iloc[31] == 0
    sells = res.trades[res.trades.qty < 0]
    assert sells.date.min() == data.index[31]
