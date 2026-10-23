import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, CostModel
from backtester.core import MarketData
from backtester.data.edgar_forms import classify_filings, parse_form_index
from backtester.data.synthetic import synthetic_market
from backtester.kalshi_maker import maker_fee, maker_fills, orders, settle, summarize
from backtester.spinoffs import child_events, dead_fate, survivorship_bound
from backtester.strategies import CEFDiscount, PreFOMCOverlay
from backtester.weather_brackets import edge_trades, taker_fee

T0 = pd.Timestamp("2025-06-01 18:00", tz="UTC")


def prints(rows):
    return pd.DataFrame(rows, columns=["ticker", "count", "yes_price", "taker_side", "created_time"])


def test_orders_last_price_follows_sweep_direction():
    p = prints([("M", 5, 40, "yes", T0), ("M", 5, 42, "yes", T0), ("M", 3, 39, "no", T0 + pd.Timedelta("10s")),
                ("M", 3, 37, "no", T0 + pd.Timedelta("10s"))])
    o = orders(p)
    assert o["last"].tolist() == [42, 37]


def test_maker_fills_through_vs_at():
    t = [T0 + pd.Timedelta(seconds=s) for s in (0, 1, 10, 20)]
    p = prints([("M", 10, 50, "yes", t[0]),
                ("M", 4, 50, "yes", t[1]),    # within 2 s of t0: quotes still off t0
                ("M", 6, 49, "no", t[2]),     # at our bid 49
                ("M", 2, 47, "no", t[3])])    # through the bid (ref is now 49 -> bid 48)
    through = maker_fills(p, k=1, through=True)
    assert len(through) == 1 and through.iloc[0]["price"] == 48 and through.iloc[0]["contracts"] == 2
    at = maker_fills(p, k=1, through=False)
    assert sorted(at["price"].tolist()) == [48, 49]
    s = settle(at, pd.DataFrame({"ticker": ["M"], "result": ["no"]}))
    assert (s["gross"] == -s["price"]).all()  # bought YES, settled NO
    assert s["net"].iloc[0] == pytest.approx(s["gross"].iloc[0] - maker_fee(s["price"].iloc[0]))
    summ = summarize(s)
    assert summ["contracts"].iloc[0] == 8


def test_maker_fee():
    assert maker_fee(50) == pytest.approx(0.4375)
    assert maker_fee([1, 99])[0] == pytest.approx(maker_fee([1, 99])[1])


def cef_data(n=400):
    idx = pd.bdate_range("2020-01-01", periods=n)
    price = pd.Series(10.0, index=idx)
    nav = pd.Series(10.0, index=idx)
    disc = np.full(n, -0.05)
    disc[300] = -0.15  # one-day discount blow-out
    raw = nav * (1 + disc)
    bars = {"A": pd.DataFrame({"open": raw, "high": raw, "low": raw, "close": raw, "volume": 1e6}, index=idx),
            "B": pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6}, index=idx)}
    navs = pd.DataFrame({"A": nav, "B": nav})
    return MarketData(bars, {"nav": navs})


def test_cef_discount_enters_on_wide_discount_only_after_signal():
    data = cef_data()
    w = CEFDiscount(lookback=100, entry_z=2.0, exit_z=0.0, max_positions=2).generate_targets(data)
    assert w.shape == (400, 2)
    assert w["A"].iloc[:300].sum() == 0 and w["A"].iloc[300] == 0.5  # signal at close 300
    assert w["A"].iloc[301] == 0.0  # discount back at mean -> exit
    assert w["B"].sum() == 0
    res = Backtester(5000, costs={"A": CostModel(spread_bps=0, slippage_bps=0), "B": CostModel(spread_bps=0, slippage_bps=0)}).run(
        CEFDiscount(lookback=100, entry_z=2.0, max_positions=2), data)
    first = res.trades["date"].min() if len(res.trades) else None
    assert first is None or first > data.index[300]


def test_pre_fomc_overlay_window():
    data = synthetic_market(("UPRO",), n=60) if "n" in synthetic_market.__code__.co_varnames else synthetic_market(("UPRO",))
    idx = data.index
    d = idx[20]
    data.extra["fomc"] = pd.DatetimeIndex([d])
    w = PreFOMCOverlay(add=0.5).generate_targets(data)
    assert w["UPRO"].iloc[18] == 0.5 and w["UPRO"].iloc[19] == 0.5
    assert w["UPRO"].iloc[17] == 0 and w["UPRO"].iloc[20] == 0


def test_weather_edge_trades_fee_and_sides():
    rows = pd.DataFrame({"day": ["2026-08-01"] * 2, "p_model": [0.5, 0.1], "yes_ask": [0.3, 0.31],
                         "yes_bid": [0.28, 0.3], "y": [1.0, 0.0]})
    t = edge_trades(rows, "p_model", 0.10)
    yes = t[t.side == "yes"].iloc[0]
    assert yes["pnl"] == pytest.approx(1 - 0.3 - 0.02)
    no = t[t.side == "no"].iloc[0]
    assert no["price"] == pytest.approx(0.7) and no["pnl"] == pytest.approx(1 - 0.7 - 0.02)
    assert taker_fee(0.5) == pytest.approx(0.02)


def test_form_index_and_filer_classification():
    line = "10-12B      Spinco Inc                                                    1234567     2019-03-04  edgar/data/1234567/0001.txt"
    df = parse_form_index("header\n" + line + "\n8-K         Other Co      99   2019-03-04  x.txt", ("10-12B",))
    assert df.iloc[0]["cik"] == 1234567 and df.iloc[0]["name"] == "Spinco Inc" and len(df) == 1
    f = pd.DataFrame({"form": ["10-12B", "10-Q", "10-K", "8-K", "DEFM14A"], "date": ["2019-01-01", "2019-08-01", "2020-03-01", "2021-01-01", "2021-02-01"],
                      "items": ["", "", "", "1.01,9.01", ""]})
    s = classify_filings(f, "2019-01-01")
    assert s["n_periodic"] == 2 and s["acquired"] and not s["bankrupt"]
    assert dead_fate(True, True) == "bankrupt" and dead_fate(False, False) == "other"


def test_spinoff_events_and_bound():
    idx = pd.bdate_range("2015-01-01", periods=400)
    child = pd.Series(np.nan, index=idx)
    child.iloc[100:] = np.linspace(10, 20, 300)
    bench = pd.Series(100.0, index=idx)
    closes = pd.DataFrame({"KID": child, "IWM": bench})
    ev = child_events(pd.Series({"KID": "2015-04-01"}), closes, closes, lags=(21,), holds=(126,))
    r = ev.iloc[0]
    assert r["entry"] == idx[121] and r["bench"] == 0
    assert r["ret"] == pytest.approx(child.iloc[247] / child.iloc[121] - 1)
    b = survivorship_bound(pd.Series([0.2, 0.4]), pd.Series(["bankrupt", "acquired"]))
    assert b["mean_excess"] == pytest.approx((0.2 + 0.4 - 1.0 + 0.0) / 4)


def test_auction_cycles_and_window():
    from backtester.auction_cycle import cycles, simulate, window_mask, window_returns

    idx = pd.bdate_range("2024-01-01", periods=40)
    cyc = cycles([idx[10], idx[11], idx[30], pd.Timestamp("2024-01-06")], idx)  # a Saturday is dropped
    assert cyc == [(10, 11), (30, 30)]
    m = window_mask(idx, cyc, days=3)
    assert m.iloc[8:12].all() and not m.iloc[7] and not m.iloc[12]
    close = pd.DataFrame({"TLT": np.linspace(100, 139, 40), "TBF": np.linspace(50, 30.5, 40)}, index=idx)
    w = window_returns(close["TLT"], cyc, days=3)
    assert w.index[0] == idx[7] and w.iloc[0] == pytest.approx(close.TLT.iloc[11] / close.TLT.iloc[7] - 1)
    r = simulate(close, idx[[10, 11, 30]], "TLT", days=3, cost_bps=0)
    rets = close.TLT.pct_change().fillna(0)
    assert r.iloc[9] == 0 and r.iloc[12] == pytest.approx(rets.iloc[12])
    rc = simulate(close, idx[[10, 11, 30]], "TLT", days=3, cost_bps=10)
    assert rc.iloc[8] == pytest.approx(-10 / 1e4)  # exit at close 7 shows up on day 8
    rs = simulate(close, idx[[10, 11, 30]], "TLT", days=3, short="TBF", cost_bps=0)
    assert rs.iloc[9] == pytest.approx(close.TBF.pct_change().iloc[9])


def test_calendar_overlay_and_opex():
    from backtester.strategies import CalendarOverlay, VolTarget, opex_days

    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    ox = opex_days(idx)
    assert list(ox.strftime("%Y-%m-%d")) == ["2024-01-19", "2024-02-16", "2024-03-15"]
    data = synthetic_market(("UPRO",))
    base = VolTarget(symbol="UPRO", target_vol=0.2, max_weight=0.5, band=0.0)
    mult = pd.Series(1.0, index=data.index)
    mult.iloc[100] = 2.0
    data.extra["mult"] = mult
    w = CalendarOverlay(base=base).generate_targets(data)
    b = base.generate_targets(data)
    assert w["UPRO"].iloc[99] == pytest.approx(min(2 * b["UPRO"].iloc[99], 1.0))  # set at the close before
    assert w["UPRO"].iloc[100] == pytest.approx(b["UPRO"].iloc[100])
