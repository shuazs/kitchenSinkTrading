import numpy as np
import pandas as pd

from backtester.mention_models import (
    bankroll, count_hits, history_features, snapshots, speaker_of, summary, trades, variants,
)

T0 = pd.Timestamp("2025-06-01 12:00", tz="UTC")


def h(x):
    return T0 + pd.Timedelta(hours=x)


def test_variants_and_hits():
    assert variants("Airball / Air Ball") == ["airball", "air ball"]
    assert count_hits("Tariff", "Tariffs are great. TARIFF! tariffing") == 2
    assert count_hits("Air Ball", "that was an air  ball") == 1
    assert count_hits("ICE / Immigration and Customs Enforcement", "the ICE's agents; nice") == 1
    assert speaker_of("KXTRUMPMENTIONB") == "TRUMP" and speaker_of("KXSECPRESSMENTION") == "SECPRESS"


def toy():
    m = pd.DataFrame({
        "ticker": ["E1-A", "E1-B", "E2-A"], "event_ticker": ["E1", "E1", "E2"], "series": ["KXFOOMENTION"] * 3,
        "title": ["t"] * 3, "yes_sub_title": ["Alpha", "Beta", "Alpha"], "result": ["yes", "no", "no"],
        "open_time": [h(0), h(0), h(20)], "close_time": [h(10), h(10), h(30)],
    })
    p = pd.DataFrame({
        "ticker": ["E1-A", "E1-A", "E1-A", "E1-B", "E1-B", "E2-A", "E2-A", "E2-A"],
        "count": [5, 7, 3, 4, 6, 2, 9, 9],
        "yes_price": [40, 45, 50, 30, 35, 60, 55, 70],
        "taker_side": ["yes", "no", "yes", "yes", "no", "yes", "no", "yes"],
        "created_time": [h(1), h(3), h(4), h(1), h(2.5), h(21), h(23), h(24)],
    })
    return m, p


def test_snapshots_use_prints_on_each_side():
    m, p = toy()
    s = snapshots(m, p, hours=2).set_index("ticker")
    a = s.loc["E1-A"]
    assert a["p0"] == 0.40 and a["yes_ask"] == 0.50 and np.isclose(a["no_ask"], 0.55)
    assert np.isclose(s.loc["E1-B", "no_ask"], 0.65) and np.isnan(s.loc["E1-B", "yes_ask"])
    assert s.loc["E2-A", "y"] == 0 and np.isclose(s.loc["E2-A", "no_ask"], 0.45)


def test_history_is_causal():
    m, p = toy()
    s = history_features(snapshots(m, p, hours=2)).set_index("ticker")
    assert s.loc["E1-A", "n_sw"] == 0  # nothing settled before E1
    assert s.loc["E2-A", "n_sw"] == 1  # E1 "Alpha" settled Yes at h10 < h22
    assert s.loc["E2-A", "r_sw"] > s.loc["E1-A", "r_sw"]


def test_trades_and_bankroll():
    m, p = toy()
    s = snapshots(m, p, hours=2)
    t = trades(s, np.full(len(s), 0.1), thr=0.0, side="no")
    assert set(t["side"]) == {"no"}
    e2 = t[t["ticker"] == "E2-A"].iloc[0]
    assert np.isclose(e2["net"], 1 - 0.45 - 0.07 * 0.45 * 0.55)
    assert summary(t)["trades"] == len(t)
    eq = bankroll(t)
    assert len(eq) == t["event_ticker"].nunique() and eq.iloc[-1] != 5000
