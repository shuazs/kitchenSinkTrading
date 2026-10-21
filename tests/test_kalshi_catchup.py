import numpy as np
import pandas as pd

from backtester.kalshi_catchup import catchup_backtest, digital_fair_value, known_prices, summarize, taker_fee


def test_taker_fee_rounds_up_to_the_cent():
    assert taker_fee(100, 0.5) == 1.75  # 0.07 * 100 * 0.25
    assert taker_fee(1, 0.5) == 0.02  # 0.0175 rounds up
    assert taker_fee(100, 0.5, rate=0.035) == 0.88


def test_digital_fair_value_is_half_at_the_money_and_monotone():
    assert abs(float(digital_fair_value(100.0, 100.0, 0.001, 30)) - 0.5) < 1e-12
    v = digital_fair_value(np.array([99.0, 100.0, 101.0]), 100.0, 0.001, 30)
    assert v[0] < 0.5 < v[2]


def test_known_prices_shifts_to_bar_end():
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2025-01-01 14:00", periods=2, freq="min", tz="UTC"))
    assert known_prices(bars).index[0] == pd.Timestamp("2025-01-01 14:01", tz="UTC")


def _setup(stale_after: bool):
    """Flat underlying with small noise, then a +0.5% jump at 14:30; the ATM contract's ask stays at 50c."""
    idx = pd.date_range("2025-01-01 13:00", "2025-01-01 15:00", freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.0003, len(idx))))
    p[idx >= "2025-01-01 14:30"] *= 1.005
    px = pd.Series(p, idx)
    K = float(px[px.index < "2025-01-01 14:30"].iloc[-1])
    rows = [("2025-01-01 14:20:00", 50, "yes", 1000)]  # last ask before the jump
    if stale_after:
        rows.append(("2025-01-01 14:30:10", 50, "yes", 300))  # someone still lifts the stale ask
    rows.append(("2025-01-01 14:50:00", 90, "no", 1000))  # bid later
    tr = pd.DataFrame(rows, columns=["created_time", "yes_price", "taker_side", "count"])
    tr["created_time"] = pd.to_datetime(tr.created_time, utc=True).astype("datetime64[ns, UTC]")
    tr = tr.assign(ticker="EV-T1", event_ticker="EV", K=K, result="yes",
                   close_time=pd.Timestamp("2025-01-01 15:00", tz="UTC").as_unit("ns"))
    return px, tr


def test_catchup_buys_the_stale_ask_and_caps_size_at_printed_volume():
    px, tr = _setup(stale_after=True)
    r = catchup_backtest(px, tr, k=1, z=3.0, latency=5, max_tau=60)
    f = r[r.qty > 0]
    assert len(f) == 1 and f.qty.iat[0] == 300 and f.fill_px.iat[0] == 0.5
    assert f.pnl.iat[0] == 300 * 0.5 - taker_fee(300, 0.5)
    assert f.pnl_x15.iat[0] == 300 * (0.9 - 0.5) - taker_fee(300, 0.5) - taker_fee(300, 0.9)
    assert summarize(r)["fills"] == 1


def test_catchup_does_not_fill_without_a_later_print_or_inside_latency():
    px, tr = _setup(stale_after=False)
    assert summarize(catchup_backtest(px, tr, k=1, z=3.0))["fills"] == 0
    px, tr = _setup(stale_after=True)
    assert summarize(catchup_backtest(px, tr, k=1, z=3.0, latency=30))["fills"] == 0
