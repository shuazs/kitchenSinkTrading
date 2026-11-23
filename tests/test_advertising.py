import numpy as np
import pandas as pd

from backtester.data import advertising as A


def test_signals_growth_and_gap_filter():
    end = pd.to_datetime(["2019-12-31", "2020-12-31", "2022-12-31"])
    p = pd.DataFrame({"cik": 1, "end": end, "ad": [50e6, 60e6, 90e6], "rev": [500e6, 540e6, 700e6]})
    s = A.signals(p)
    # 2021 is missing, so the 2022 row has no consecutive prior year and is dropped
    assert list(s.end.dt.year) == [2020]
    assert np.isclose(s.ad_g.iloc[0], np.log(1.2))
    assert np.isclose(s.intensity.iloc[0], 60 / 540)
    assert np.isclose(s.excess_g.iloc[0], np.log(1.2) - np.log(540 / 500))


def test_signals_min_ad_filter():
    end = pd.to_datetime(["2019-12-31", "2020-12-31"])
    p = pd.DataFrame({"cik": 2, "end": end, "ad": [1e6, 2e6], "rev": [10e6, 11e6]})
    assert A.signals(p).empty
