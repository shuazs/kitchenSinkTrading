import numpy as np
import pandas as pd

from backtester import MarketData
from backtester.data.fundamentals import as_of
from backtester.strategies import FundamentalAllocator
from backtester.strategies.fundamental_allocator import macro_stress

N_STOCKS = 60
IDX = pd.bdate_range("2012-01-02", periods=700)


def bars(drift=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    px = 50 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(len(IDX))))
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6, "raw_close": px}, index=IDX)


def fundamentals(flip_after=None):
    """Stock i has ROE rising with i. With `flip_after`, a later filing
    reverses the ranking, but only becomes public on that date."""
    rows = []
    for i in range(N_STOCKS):
        for item, val in [("net_income", 10 + i), ("equity", 100), ("assets", 200), ("liabilities", 100),
                          ("op_cash_flow", 10 + i), ("gross_profit", 50 + i), ("shares", 1e6)]:
            rows.append((i + 1, item, pd.Timestamp("2011-12-31"), pd.Timestamp("2012-01-01"), float(val)))
            if flip_after is not None and item in ("net_income", "op_cash_flow", "gross_profit"):
                rows.append((i + 1, item, pd.Timestamp("2013-06-30"), pd.Timestamp(flip_after), float(val - 2 * i)))
    return pd.DataFrame(rows, columns=["cik", "item", "end", "available", "val"])


def market(flip_after=None, spy_drift=0.0003, unrate=None):
    stocks = {f"S{i}": bars(seed=i) for i in range(N_STOCKS)}
    etfs = {s: bars(seed=100 + k, drift=spy_drift if s == "SPY" else 0.0002) for k, s in enumerate(["SPY", "QQQ", "EFA", "EEM", "IJR", "IEF", "BIL"])}
    extra = {
        "members": pd.Series([frozenset(stocks)], index=[IDX[0]]),
        "fundamentals": fundamentals(flip_after),
        "cik": {f"S{i}": i + 1 for i in range(N_STOCKS)},
        "sector": {f"S{i}": "tech" if i % 2 else "health" for i in range(N_STOCKS)},
    }
    if unrate is not None:
        extra["macro"] = pd.DataFrame({"UNRATE": unrate})
    return MarketData({**stocks, **etfs}, extra)


QUALITY_ONLY = dict(stock_etf=None, factors={"quality": 1.0}, macro=False, sleeve_trend=False, vol_target=None, min_history=200)


def test_as_of_ignores_values_before_they_are_public():
    long = fundamentals(flip_after="2013-09-01")
    got = as_of(long, pd.DatetimeIndex(["2013-08-30", "2013-09-02"]), [5])
    assert got["net_income"].loc["2013-08-30", 5] == 14
    assert got["net_income"].loc["2013-09-02", 5] == 14 - 8


def test_picks_top_quality_with_sector_cap():
    w = FundamentalAllocator(sleeves={"stocks": 1.0}, top_n=10, max_sector=0.5, **QUALITY_ONLY).generate_targets(market())
    last = w.iloc[-1]
    held = sorted(last[last > 0].index, key=lambda s: int(s[1:]))
    assert len(held) == 10 and np.allclose(last[held], 0.1)
    # The best names are the high numbers; the cap forces 5 per sector.
    assert sum(int(s[1:]) % 2 for s in held) == 5
    assert min(int(s[1:]) for s in held) >= N_STOCKS - 12


def test_new_filing_changes_picks_only_after_it_is_public():
    w = FundamentalAllocator(sleeves={"stocks": 1.0}, top_n=10, buffer=1.0, **QUALITY_ONLY).generate_targets(market(flip_after="2013-09-01"))
    before = w.loc[:"2013-08-31"].iloc[-1]
    after = w.loc["2013-10-01":].iloc[0]
    assert before[[f"S{i}" for i in range(50, 60)]].sum() > 0.99
    assert after[[f"S{i}" for i in range(10)]].sum() > 0.99


def test_default_holds_qqq_intl_and_small_caps():
    w = FundamentalAllocator(min_history=200, macro=False, sleeve_trend=False, vol_target=None).generate_targets(market())
    last = w.iloc[-1]
    assert np.isclose(last["QQQ"], 0.8) and np.isclose(last["IJR"], 0.1) and np.isclose(last[["EFA", "EEM"]].sum(), 0.1)


def test_weights_are_long_only_and_fully_invested():
    w = FundamentalAllocator(stock_etf=None, min_history=200, vol_target=0.05).generate_targets(market())
    assert (w >= -1e-12).all().all()
    # Fully invested once the defensive assets have 6 months of history.
    assert np.allclose(w.loc["2012-08-01":].sum(axis=1), 1.0)
    assert (w.sum(axis=1) <= 1 + 1e-9).all()


def test_weak_economy_and_downtrend_goes_fully_defensive():
    months = pd.date_range("2010-01-01", "2015-01-01", freq="MS")
    unrate = pd.Series(np.linspace(4, 9, len(months)), index=months)  # steadily rising
    data = market(spy_drift=-0.002, unrate=unrate)
    assert macro_stress(data, IDX).iloc[-1]
    w = FundamentalAllocator(stock_etf=None, min_history=200).generate_targets(data)
    last = w.iloc[-1]
    assert last[["IEF", "BIL"]].sum() > 0.999
