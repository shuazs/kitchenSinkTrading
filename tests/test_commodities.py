import numpy as np
import pandas as pd

from backtester.commodities import (expiry_mask, ratio_positions, realized_carry, season_returns, storage_day_returns,
                                    storage_report_days, wasde_revisions)


def test_storage_report_days_thursday_or_holiday_wednesday():
    idx = pd.bdate_range("2025-11-17", "2025-12-05").drop(pd.Timestamp("2025-11-27"))  # Thanksgiving
    days = storage_report_days(idx, start="2025-01-01")
    assert list(days.strftime("%m-%d")) == ["11-20", "11-26", "12-04"]


def test_storage_day_returns_short_and_cost():
    idx = pd.bdate_range("2025-01-06", periods=5)  # Mon..Fri
    close = pd.Series([10.0, 10.0, 10.0, 9.0, 9.0], index=idx)
    r = storage_day_returns(close, storage_report_days(idx, "2025-01-01"), cost=0.001)
    assert len(r) == 1 and np.isclose(r.iloc[0], -np.log(0.9) - 0.001)


def test_ratio_positions_fade_and_exit():
    lr = pd.Series(np.r_[np.zeros(60) + np.random.default_rng(0).normal(0, 0.01, 60), [0.2, 0.1, -0.05, 0.0]])
    pos = ratio_positions(lr, lookback=60, z_in=2.0)
    assert pos.iloc[60] == -1  # spike up -> short the ratio
    assert pos.iloc[62] == 0   # back below the mean -> out


def test_realized_carry_is_roll_yield():
    idx = pd.bdate_range("2020-01-01", periods=300)
    fut = pd.DataFrame({"X": np.full(300, 50.0)}, index=idx)          # flat front month
    etf = pd.DataFrame({"X": 10 * np.exp(-0.0004 * np.arange(300))}, index=idx)  # loses 4 bp/day to rolls
    c = realized_carry(etf, fut, pd.Series(0.0, index=idx), lookback=252)
    assert np.isclose(c["X"].iloc[-1], -0.0004 * 252)


def test_season_returns_mask_zeroes_roll_days():
    idx = pd.bdate_range("2020-06-25", "2020-10-05")
    close = pd.Series(100.0, index=idx)
    close[idx >= "2020-07-14"] = 110.0  # a roll gap on the 14th
    m = expiry_mask(idx, (7, 9))
    assert np.isclose(season_returns(close, "07-01", "09-30").loc[2020], np.log(1.1))
    assert season_returns(close, "07-01", "09-30", m).loc[2020] == 0


def test_wasde_revisions():
    rows = []
    for d, v in [("2020-01-10", 100.0), ("2020-02-10", 90.0)]:
        rows.append({"Region": "United States", "Attribute": "Ending Stocks", "ProjEstFlag": "Proj.",
                     "ReportTitle": "U.S. Feed Grain and Corn Supply and Use", "Commodity": "Corn",
                     "MarketYear": "2019/20", "Value": v, "release": pd.Timestamp(d)})
    out = wasde_revisions(pd.DataFrame(rows), commodities=("Corn",))
    assert len(out) == 1 and np.isclose(out.revision.iloc[0], -0.1)
