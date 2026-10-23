import numpy as np
import pandas as pd

from backtester import Backtester, CostModel
from backtester.data import cot
from backtester.data.synthetic import synthetic_market
from backtester.strategies import CotPositioning

FREE = CostModel(spread_bps=0, slippage_bps=0)


def fake_pit(weeks=200, seed=0):
    rng = np.random.default_rng(seed)
    tuesdays = pd.date_range("2015-01-06", periods=weeks, freq="7D")
    rows = []
    for code, sym in [("088691", "GLD"), ("084691", "SLV"), ("067651", "USO"), ("023651", "UNG")]:
        spec = np.cumsum(rng.normal(0, 1000, weeks))
        for i, t in enumerate(tuesdays):
            known = t + pd.Timedelta(days=3, hours=15, minutes=30)
            if i == 50:
                known += pd.Timedelta(days=10)  # a late (shutdown) release
            rows.append({"entity_id": code, "event_date": t.date(), "open_interest": 100_000.0,
                         "net_noncommercial": spec[i], "net_commercial": -spec[i], "net_nonreportable": 0.0,
                         "knowledge_date": known.tz_localize("America/New_York").tz_convert("UTC")})
    return pd.DataFrame(rows)


def test_panel_dates_reports_by_publication_not_by_tuesday():
    panel = cot.weekly_panel(fake_pit())
    known = panel["known"]
    assert (known.index.dayofweek == 1).all()
    lag = (known - known.index).dt.days
    assert lag.iloc[0] == 3 and lag.iloc[50] == 13


def test_signals_and_trading_day_placement():
    panel = cot.weekly_panel(fake_pit())
    sig = cot.positioning_signals(panel)
    assert np.allclose(sig["hedging_pressure"].dropna(), -sig["commercial_index"].dropna())
    # speculators buying -> flow1 negative (contrarian)
    d = panel["net_noncommercial"]["GLD"].diff()
    f = sig["flow1"]["GLD"]
    ok = f.notna() & (d != 0)
    assert (np.sign(f[ok]) == -np.sign(d[ok])).all()
    days = pd.bdate_range("2015-01-01", "2019-12-31")
    placed = cot.to_trading_days(sig["flow1"], panel["known"], days)
    assert (placed.index.dayofweek == 4).mean() > 0.95  # Friday release -> Friday row, trades Monday open
    assert placed.index.is_monotonic_increasing


def test_strategy_holds_top_n_and_shorts_bottom_n():
    data = synthetic_market(["A", "B", "C", "D"], seed=1, n=300)
    idx = data.index[::5]
    sig = pd.DataFrame({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}, index=idx)
    data.extra["cot_signal"] = sig
    w = CotPositioning(top_n=2).generate_targets(data)
    assert np.allclose(w.iloc[-1][["A", "B", "C", "D"]], [0.5, 0.5, 0, 0])
    w = CotPositioning(top_n=1, short=True).generate_targets(data)
    assert np.allclose(w.iloc[-1][["A", "B", "C", "D"]], [0.5, 0, 0, -0.5])
    res = Backtester(5_000, costs=FREE, allow_short=True).run(CotPositioning(top_n=1, short=True), data)
    assert res.positions["A"].iloc[-1] > 0 and res.positions["D"].iloc[-1] < 0
