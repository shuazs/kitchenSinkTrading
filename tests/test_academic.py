import numpy as np
import pandas as pd

from backtester import Backtester, CostModel, MarketData
from backtester import intraday
from backtester.data import edgar
from backtester.data.synthetic import synthetic_market
from backtester.strategies.crypto import EnsembleTrend
from backtester.strategies.earnings import EarningsPremium, announcement_windows
from backtester.strategies.insider import classify_insiders, cluster_signals
from backtester.strategies.seasonality import SameMonthSeasonality

FREE = CostModel(spread_bps=0, slippage_bps=0)


def _bars(close: np.ndarray, start="2020-01-01", freq="D", volume=1e7) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq=freq)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx)


def test_ensemble_trend_signal_is_full_in_a_steady_uptrend():
    up = _bars(100 * 1.01 ** np.arange(400))
    s = EnsembleTrend(lookbacks=(5, 20, 60)).signal(up[["close"]])
    assert s["close"].iloc[100:].eq(1.0).all()
    down = _bars(100 * 0.99 ** np.arange(400))
    assert EnsembleTrend(lookbacks=(5, 20, 60)).signal(down[["close"]])["close"].eq(0.0).all()


def test_ensemble_trend_universe_waits_for_history_and_volume():
    rng = np.random.default_rng(0)
    old = _bars(100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 500))))
    new = _bars(100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 500))))
    new.iloc[:300] = np.nan  # listed 300 days later
    thin = _bars(100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 500))), volume=1.0)
    data = MarketData({"OLD": old, "NEW": new, "THIN": thin})
    member = EnsembleTrend(min_history=100, universe_days=1, top_n=5).universe(data)
    assert not member["THIN"].any()
    assert not member["NEW"].iloc[:399].any() and member["NEW"].iloc[-1]
    w = EnsembleTrend(min_history=100, top_n=2, target_vol=10.0).generate_targets(data)
    assert (w.sum(axis=1) <= 1 + 1e-9).all() and (w["THIN"] == 0).all()


def test_classify_insiders_routine_opportunistic_unclassified():
    rows = []
    for y in (2015, 2016, 2017):
        rows.append(("routine", f"{y}-04-10"))  # every April
        rows.append(("opp", f"{y}-{(y - 2014) * 3:02d}-10"))  # different months
    rows += [("routine", "2018-06-01"), ("opp", "2018-06-01"), ("new", "2018-06-01")]
    t = pd.DataFrame(rows, columns=["owner", "trans_date"])
    t["filed"] = pd.to_datetime(t["trans_date"]) + pd.Timedelta(days=2)
    kinds = classify_insiders(t).query("trans_date == '2018-06-01'").set_index("owner")["trader_type"]
    assert kinds.to_dict() == {"routine": "routine", "opp": "opportunistic", "new": "unclassified"}


def test_cluster_signals_can_keep_only_opportunistic_buyers():
    rows = []
    for y in (2015, 2016, 2017):
        rows += [("a", f"{y}-04-10", "S"), ("b", f"{y}-0{y - 2013}-10", "S")]
    rows += [("a", "2018-06-01", "P"), ("b", "2018-06-02", "P")]
    t = pd.DataFrame(rows, columns=["owner", "trans_date", "code"])
    t["filed"] = pd.to_datetime(t["trans_date"])
    t["symbol"], t["value"] = "XYZ", 200_000
    both = cluster_signals(t, min_insiders=2, min_value=100_000)
    assert len(both) == 1
    assert cluster_signals(t, min_insiders=2, trader_types=("opportunistic",)).empty
    solo = cluster_signals(t, min_insiders=1, trader_types=("opportunistic",))
    assert solo["filed"].tolist() == [pd.Timestamp("2018-06-02")]


def test_earnings_dates_shifts_after_close_filings(monkeypatch):
    subs = pd.DataFrame({
        "form": ["8-K", "8-K", "8-K", "10-Q"],
        "items": ["2.02,9.01", "2.02", "5.02", ""],
        "filingDate": pd.to_datetime(["2024-01-25", "2024-04-25", "2024-05-01", "2024-05-02"]),
        "acceptanceDateTime": ["2024-01-25T21:30:00.000Z", "2024-04-25T11:00:00.000Z",
                               "2024-05-01T12:00:00.000Z", "2024-05-02T12:00:00.000Z"],
    })
    monkeypatch.setattr(edgar, "submissions", lambda ticker, refresh=False: subs)
    d = edgar.earnings_dates("XYZ")
    # 21:30 UTC = 16:30 New York, after the close -> next session; 11:00 UTC is pre-market.
    assert d["date"].tolist() == [pd.Timestamp("2024-01-26"), pd.Timestamp("2024-04-25")]


def test_earnings_premium_holds_the_window_then_returns_to_idle():
    data = synthetic_market(["AAA", "SPY"], n=600)
    idx = data.index
    last_year = idx[100]
    data.extra["earnings_dates"] = {"AAA": [last_year]}
    strat = EarningsPremium(pre_days=3, post_days=1, max_positions=2, idle="SPY")
    res = Backtester(costs=FREE).run(strat, data)
    k = idx.searchsorted(last_year + pd.Timedelta(days=364))
    held = res.positions["AAA"] > 0
    # Bought at the open 3 sessions before day k, held through k's close, sold next open.
    assert held.iloc[k - 3: k + 1].all()
    assert not held.iloc[: k - 3].any() and not held.iloc[k + 1:].any()
    assert abs(res.weights["SPY"].iloc[k - 2] - 0.5) < 0.05
    w = announcement_windows({"AAA": [last_year]}, idx, pre_days=3, post_days=1)
    assert w[["entry", "exit"]].iloc[0].tolist() == [k - 4, k]


def test_seasonality_picks_the_asset_that_rises_every_january():
    idx = pd.bdate_range("2010-01-01", "2020-12-31")
    jan = np.where(idx.month == 1, 0.004, -0.0002)
    flat = np.full(len(idx), 0.0001)
    data = MarketData({
        "JAN": _bars(100 * np.exp(np.cumsum(jan)), freq="B", start="2010-01-01").set_axis(idx),
        "FLAT": _bars(100 * np.exp(np.cumsum(flat)), freq="B", start="2010-01-01").set_axis(idx),
    })
    w = SameMonthSeasonality(top_n=1, min_years=3).generate_targets(data)
    dec_ends = w.index[(w.index.month == 12) & (w.index.year >= 2014)]
    assert (w.loc[dec_ends, "JAN"] == 1.0).all()
    jun_ends = w.index[(w.index.month == 6) & (w.index.year >= 2014)]
    assert (w.loc[jun_ends, "JAN"] == 0.0).all()


def _minute_day(day: str, path: np.ndarray) -> pd.DataFrame:
    t = pd.date_range(f"{day} 09:30", periods=390, freq="1min")
    return pd.DataFrame({"open": path, "high": path, "low": path, "close": path, "volume": 1000.0}, index=t)


def test_minute_grid_labels_bars_by_end_minute():
    bars = _minute_day("2024-03-01", np.linspace(100, 101, 390))
    g = intraday.minute_grid(bars)
    assert g["close"].columns[0] == 1 and g["close"].columns[-1] == 390
    assert g["close"].iloc[0, 29] == bars["close"].iloc[29]  # 10:00 = close of the 09:59 bar


def test_first_half_hour_goes_long_after_a_strong_open():
    days = pd.bdate_range("2024-03-01", periods=3)
    flat = _minute_day(str(days[0].date()), np.full(390, 100.0))
    up = _minute_day(str(days[1].date()), np.r_[np.linspace(100, 101, 30), np.linspace(101, 102, 360)])
    down = _minute_day(str(days[2].date()), np.r_[np.linspace(102, 101, 30), np.linspace(101, 100, 360)])
    out = intraday.first_half_hour(pd.concat([flat, up, down]), long_only=False, cost=FREE)
    assert out["side"].tolist() == [1.0, -1.0]
    assert (out["ret"] > 0).all()


def test_noise_area_trades_a_breakout_and_flattens_by_the_close():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2024-01-01", periods=40)
    frames, px = [], 100.0
    for d in days[:-1]:
        path = px * (1 + np.cumsum(rng.normal(0, 0.0003, 390)))
        frames.append(_minute_day(str(d.date()), path))
        px = path[-1]
    trend = px * (1 + np.linspace(0, 0.03, 390))  # a 3% straight-line up day
    frames.append(_minute_day(str(days[-1].date()), trend))
    out = intraday.noise_area(pd.concat(frames), cost=FREE, target_vol=1.0)
    last = out.iloc[-1]
    assert last["n_trades"] == 2 and last["ret"] > 0
