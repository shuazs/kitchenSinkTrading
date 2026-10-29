import numpy as np
import pandas as pd

from backtester import kalshi_macro as km

R = pd.Timestamp("2025-03-12 12:30", tz="UTC")  # 08:30 ET


def h(x):
    return R - pd.Timedelta(hours=x)


def test_flow_signals_and_take():
    m = pd.DataFrame({"ticker": ["E-T1", "E-T2"], "event_ticker": ["E", "E"], "series": ["KXCPI"] * 2,
                      "result": ["yes", "no"], "close_time": [R - pd.Timedelta(minutes=5)] * 2})
    m = km.release_events(m)
    assert len(m) == 2 and list(m["y"]) == [1, 0]
    p = pd.DataFrame({
        "ticker": ["E-T1", "E-T1", "E-T1", "E-T1", "E-T2"],
        "count": [10, 2, 5, 5, 4], "yes_price": [40, 45, 50, 52, 30],
        "taker_side": ["yes", "no", "yes", "no", "no"],
        "created_time": [h(20), h(10), h(0.5), h(0.4), h(0.6)],
    })
    s = km.flow_signals(m, p, window_h=24).set_index("ticker")
    assert s.loc["E-T1", "imb"] == 8 and np.isclose(s.loc["E-T1", "imb_share"], 8 / 12)
    assert s.loc["E-T1", "yes_ask"] == 0.50 and np.isclose(s.loc["E-T1", "no_ask"], 0.48)
    t = km.take(s.reset_index(), pd.Series(["yes"]))
    assert np.isclose(t["net"].iloc[0], 1 - 0.5 - 0.07 * 0.25)
    assert km.stats(t)["trades"] == 1
