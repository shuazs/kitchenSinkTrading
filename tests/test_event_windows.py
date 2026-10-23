import numpy as np
import pandas as pd

from backtester import event_windows as ew


def test_calendar_helpers():
    assert ew.russell_recon(2024) == pd.Timestamp("2024-06-28")
    assert ew.russell_recon(2018) == pd.Timestamp("2018-06-22")  # last Friday is the 29th
    assert ew.last_friday(2026, 10) == pd.Timestamp("2026-10-30")
    assert ew.third_friday(2026, 11) == pd.Timestamp("2026-11-20")
    assert ew.vix_expiry(2026, 10) == pd.Timestamp("2026-10-21")


def test_anchor_and_nth_business_day():
    idx = pd.bdate_range("2026-01-01", "2026-03-31")
    assert ew.anchor_positions(idx, ["2026-01-03"]) == [idx.get_loc(pd.Timestamp("2026-01-05"))]
    assert ew.anchor_positions(idx, ["2026-01-03"], "on_or_before") == [idx.get_loc(pd.Timestamp("2026-01-02"))]
    assert list(ew.nth_business_day(idx, 2)) == [pd.Timestamp(d) for d in ("2026-01-02", "2026-02-03", "2026-03-03")]
    assert ew.nth_business_day(idx, -1)[0] == pd.Timestamp("2026-01-30")


def test_event_returns_mask_and_placebo():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(100 * np.cumprod(np.full(60, 1.001)), index=idx)
    bench = pd.Series(100 * np.cumprod(np.full(60, 1.0005)), index=idx)
    r = ew.event_returns(close, [10, 30], -2, 1, bench)
    assert np.allclose(r.values, 1.001 ** 3 - 1.0005 ** 3)
    assert list(r.index) == [idx[10], idx[30]]
    assert ew.event_returns(close, [0, 59], -2, 1).empty  # windows off the ends are dropped
    m = ew.window_mask(idx, [10], -2, 1)
    assert m.sum() == 3 and m.iloc[9:12].all()
    assert len(ew.placebo(close, [20, 40], -1, 0)) == 4
    st = ew.split_stats(pd.Series(dtype=float))
    assert (st["n"] == 0).all()
