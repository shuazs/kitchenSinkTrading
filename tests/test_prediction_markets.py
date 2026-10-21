import numpy as np
import pandas as pd

from backtester.data.polymarket import escalation_index, hourly_prices, iran_sign, snapshot_change
from backtester.jump_trades import grid, jump_events, jump_trades, summarize


def test_iran_sign_reads_the_question():
    assert iran_sign("US strikes Iran by February 28, 2026?", "US strikes Iran by...?") == 1
    assert iran_sign("US x Iran ceasefire by April 7?", "US x Iran ceasefire by...?") == -1
    assert iran_sign("Strait of Hormuz traffic returns to normal by end of June?") == -1
    assert iran_sign("Will Iran close the Strait of Hormuz before July?") == 1
    assert iran_sign("Khamenei out as Supreme Leader of Iran by March 31?") == 0
    assert iran_sign("Will Iran win the 2026 FIFA World Cup?", "World Cup Winner") == 0
    assert iran_sign("Fed decision in January?") == 0


def test_hourly_prices_take_the_last_fill_and_drop_dust():
    t = pd.to_datetime(["2026-04-07 10:05", "2026-04-07 10:50", "2026-04-07 10:55", "2026-04-07 12:10"])
    fills = pd.DataFrame({"time": t, "market_id": "a", "price": [0.2, 0.3, 0.9, 0.4], "usd_amount": [50, 50, 5, 50]})
    p, v = hourly_prices(fills)
    assert p.loc["2026-04-07 10:00", "a"] == 0.3  # the $5 fill is ignored
    assert p.loc["2026-04-07 11:00", "a"] == 0.3  # carried forward
    assert v.loc["2026-04-07 10:00", "a"] == 100


def test_escalation_index_signs_and_weights():
    idx = pd.date_range("2026-04-07", periods=3, freq="h")
    prices = pd.DataFrame({"strike": [0.2, 0.3, 0.3], "ceasefire": [0.5, 0.5, 0.3]}, index=idx)
    volume = pd.DataFrame({"strike": [300.0, 0, 0], "ceasefire": [100.0, 0, 0]}, index=idx)
    s = escalation_index(prices, volume, pd.Series({"strike": 1, "ceasefire": -1}))
    assert np.isclose(s.iloc[1], 0.75 * 0.1)  # strike up 10 pts, 75% of volume
    assert np.isclose(s.iloc[2], 0.25 * 0.2)  # ceasefire down 20 pts reads as escalation


def test_jump_events_keep_one_per_cooldown():
    idx = pd.date_range("2026-06-12", periods=12, freq="h")
    sig = pd.Series(0.0, index=idx)
    sig.iloc[[1, 3, 9]] = [0.06, -0.08, -0.05]
    ev = jump_events(sig, 0.05)
    assert list(ev["time"]) == [idx[1], idx[9]] and list(ev["direction"]) == [1, -1]


def test_jump_trades_enter_after_the_jump_hour():
    idx = pd.date_range("2026-06-12", periods=8, freq="h")
    close = pd.Series([100, 103, 104, 105, 106, 107, 108, 109.0], index=idx)
    bars = pd.DataFrame({"open": close.shift(1).fillna(100) + 0.5, "close": close})
    ev = pd.DataFrame({"time": [idx[1]], "direction": [1], "size": [0.06]})
    tr = jump_trades(bars, ev, hold=2, cost_bps=0)
    row = tr.iloc[0]
    assert row["entry_time"] == idx[2] and row["exit_time"] == idx[3] and row["asset_open"]
    assert np.isclose(row["ret"], np.log(105 / 103.5))
    assert np.isclose(row["gap"], np.log(103.5 / 103))
    short = jump_trades(bars, ev.assign(direction=-1), hold=2, cost_bps=10).iloc[0]
    assert np.isclose(short["ret"], -np.log(105 / 103.5) - 0.001)


def test_closed_market_jump_trades_at_reopen():
    idx = pd.DatetimeIndex(["2026-06-12 20:00", "2026-06-14 22:00", "2026-06-14 23:00"])
    bars = pd.DataFrame({"open": [70.0, 75.0, 76.0], "close": [70.0, 76.0, 75.0]}, index=idx)
    ev = pd.DataFrame({"time": [pd.Timestamp("2026-06-13 18:00")], "direction": [1], "size": [0.3]})
    row = jump_trades(bars, ev, hold=2, cost_bps=0).iloc[0]
    assert not row["asset_open"] and row["entry_time"] == idx[1]
    assert np.isclose(row["gap"], np.log(75 / 70)) and np.isclose(row["ret"], np.log(75 / 75))


def test_grid_and_summary_shapes():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=500, freq="h")
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, 500))), index=idx)
    bars = pd.DataFrame({"open": c.shift(1).bfill(), "close": c})
    sig = pd.Series(rng.normal(0, 0.02, 500), index=idx)
    g = grid(bars, sig, [0.03, 0.05], [1, 3])
    assert len(g) == 4 and (g["trades"] > 0).all()
    assert summarize(jump_trades(bars, jump_events(sig, 0.05), hold=1))["trades"] > 0


def test_snapshot_change_uses_weekly_volume():
    old = pd.DataFrame({"id": ["1", "2"], "price": [0.2, 0.6]})
    new = pd.DataFrame({
        "id": ["1", "2"], "question": ["US strikes Iran by June 30?", "US x Iran ceasefire by June 30?"],
        "event_title": ["", ""], "price": [0.3, 0.5], "volume_1wk": [1.0, 3.0], "volume": [2e6, 2e6],
    })
    assert np.isclose(snapshot_change(old, new), 0.25 * 0.1 + 0.75 * 0.1)
