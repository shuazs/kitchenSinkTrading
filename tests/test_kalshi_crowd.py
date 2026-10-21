import numpy as np
import pandas as pd

from backtester.kalshi_crowd import Ladder, cap_capital, kalshi_fee, label_prints, move_triggers, move_z, simulate

T0 = pd.Timestamp("2025-01-06 14:00", tz="UTC")


def prints(rows):
    x = pd.DataFrame(rows, columns=["ticker", "count", "yes_price", "taker_side", "created_time"])
    x["event_ticker"], x["close_time"], x["result"], x["K"] = "EV", T0 + pd.Timedelta("90min"), "yes", 100.0
    return x


def test_fee_rounds_up_to_the_cent():
    assert kalshi_fee(0.07, 100, 0.5) == 1.75
    assert kalshi_fee(0.07, 1, 0.5) == 0.02


def test_big_move_triggers_once_with_direction():
    idx = pd.date_range("2025-01-01", periods=60 * 24 * 40, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-4, len(idx))
    r[-500:-485] = -2e-3  # a sharp 15-minute drop near the end
    s = pd.Series(100 * np.exp(np.cumsum(r)), idx)
    z = move_z(s, windows={"15m": 15, "1h": 60})
    tr = move_triggers(z, "15m", threshold=5)
    assert len(tr) == 1 and tr.iloc[0] == -1
    assert idx[-500] <= tr.index[0] <= idx[-485]


def test_label_gives_continuation_side_price():
    x = prints([("A", 10, 30, "yes", T0 + pd.Timedelta("5min")), ("A", 10, 30, "no", T0 + pd.Timedelta("3h"))])
    trig = pd.Series([-1.0], [T0])
    z = pd.DataFrame({"r1h": [0.01]}, [T0 - pd.Timedelta("1h")])
    lab = label_prints(x, trig, z)
    hot, cold = lab.iloc[0], lab.iloc[1]
    assert hot.hot and hot.d == -1 and np.isclose(hot.q, 0.7) and hot.win == 0 and hot.tk_cont == 0
    assert not cold.hot and cold.d == 1 and np.isclose(cold.q, 0.3) and cold.win == 1


def test_taker_fill_capped_at_printed_size_and_settles():
    x = prints([("A", 5, 50, "yes", T0 - pd.Timedelta("1min")),
                ("A", 40, 52, "yes", T0 + pd.Timedelta("2min")),
                ("A", 900, 60, "yes", T0 + pd.Timedelta("3min")),  # above last + 5c: not taken
                ("A", 30, 48, "no", T0 + pd.Timedelta("4min"))])
    r = simulate(Ladder(x), T0, 1, "follow", "mid", "taker", 0.07)
    assert r["qty"] == 40 and np.isclose(r["px"], 0.52)
    assert np.isclose(r["pnl"], 40 * 0.48 - kalshi_fee(0.07, 40, 0.52))
    m = simulate(Ladder(x), T0, 1, "follow", "mid", "maker", 0.07)  # resting bid at 0.50, half of the 30 sold at 0.48
    assert m["qty"] == 15 and np.isclose(m["px"], 0.50)


def test_capital_cap_skips_overlapping_trades():
    t = pd.DataFrame({"t": [T0, T0 + pd.Timedelta("1min"), T0 + pd.Timedelta("2h")],
                      "close": [T0 + pd.Timedelta("1h")] * 2 + [T0 + pd.Timedelta("3h")], "cost": [4000.0, 2000.0, 4000.0]})
    assert list(cap_capital(t).cost) == [4000.0, 4000.0]
