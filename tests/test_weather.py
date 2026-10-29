import numpy as np
import pandas as pd

from backtester.data import weather as w
from backtester import weather_trades as wt


def test_degree_days_and_demand_index():
    h, c = w.degree_days(np.array([60.0, 65.0, 75.0]))
    assert list(h) == [5, 0, 0] and list(c) == [0, 0, 10]
    idx = pd.date_range("2024-01-01", periods=2)
    cities = list(w.weights().index)
    t = pd.DataFrame(55.0, index=idx, columns=cities)
    d = w.demand_index(t)
    assert np.allclose(d["hdd"], 10) and np.allclose(d["cdd"], 0) and np.allclose(d["dd"], 10)


def test_normals_and_anomaly():
    idx = pd.date_range("1991-01-01", "2020-12-31")
    seasonal = 50 + 20 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f = pd.DataFrame({"X": seasonal}, index=idx)
    a = w.anomaly(f + 3, w.normals(f))
    assert abs(a["X"].mean() - 3) < 0.2


def test_parse_isd_keeps_metar_temperatures():
    base = "0210725053947282025010100517+40779-073969FM-15+0048KNYC V0300505N00315013115MN0160935N5+00895+00395100525"
    speci = base.replace("FM-15", "FM-16")
    synop = base.replace("FM-15", "SY-MT")
    missing = base[:87] + "+9999" + base[92:]
    d = w.parse_isd([base, speci, synop, missing])
    assert len(d) == 2
    assert abs(d["temp_f"].iloc[0] - (8.9 * 9 / 5 + 32)) < 1e-9


def test_gefs_daily_leads():
    init = pd.Timestamp("2024-01-01 12:00")
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in (6, 18, 30)])
    p = pd.DataFrame({"init": init, "valid": valid, "NYC": [40.0, 30.0, 50.0]})
    d = w.gefs_daily(p)
    assert list(d["lead"]) == [0, 1]
    assert d["NYC"].tolist() == [40.0, 40.0]


def test_run_revisions_same_target_dates():
    rows = []
    for i, init in enumerate(pd.date_range("2024-01-01", periods=3)):
        for lead in range(0, 6):
            rows.append({"init": init, "date": init + pd.Timedelta(days=lead), "v": 10.0 + i})
    r = wt.run_revisions(pd.DataFrame(rows), "v", {"1-3": (1, 3)})
    assert np.allclose(r["1-3"], 1.0) and len(r) == 2


def test_kalshi_bounds_and_locks():
    m = pd.DataFrame({"yes_sub_title": ["87° or higher", "81.0° or lower", "82° to 83°"]})
    b = wt.kalshi_bounds(m)
    assert b["kind"].tolist() == ["ge", "le", "br"]
    assert b["a"].tolist()[0] == 87 and b["b"].tolist()[1] == 81 and b["b"].tolist()[2] == 83
    side = wt.locked_side(b["kind"], b["a"], b["b"], [88.0, 83.0, 84.0], margin=1.0)
    assert list(side) == ["yes", "no", None]


def test_event_returns_trade_after_the_event():
    idx = pd.bdate_range("2024-01-01", periods=6)
    o = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    c = o + 0.5
    r = wt.event_returns(o, c, [idx[1]], hold=2)
    assert r.index[0] == idx[1]
    assert abs(r.iloc[0] - (c.iloc[3] / o.iloc[2] - 1)) < 1e-12
