import numpy as np
import pandas as pd

from backtester.news_sentiment import (cashtag_whitelist, cost_for, fade_book, kalshi_directional, kalshi_fee,
                                       mention_spikes, session_for, stock_events, ticker_mentions)

SESSIONS = pd.bdate_range("2024-01-01", periods=30)


def test_text_trades_at_the_first_open_after_it_is_public():
    ts = ["2024-01-03 13:00Z",  # 08:00 ET -> same day's open
          "2024-01-03 15:00Z",  # 10:00 ET -> next day
          "2024-01-05 22:00Z",  # Friday after the close -> Monday
          "2024-01-06 15:00Z"]  # Saturday -> Monday
    T, pre = session_for(ts, SESSIONS)
    assert list(pd.DatetimeIndex(T).strftime("%m-%d")) == ["01-03", "01-04", "01-08", "01-08"]
    assert list(pre) == [True, False, False, False]


def test_stock_events_returns_and_market_excess():
    idx = SESSIONS
    opens = pd.DataFrame({"SPY": 100.0, "ABC": 10.0}, index=idx)
    closes = pd.DataFrame({"SPY": 101.0, "ABC": 10.5}, index=idx)
    vol = pd.DataFrame({"SPY": 1e9, "ABC": 1e8}, index=idx)
    news = pd.DataFrame({"published": ["2024-02-02 12:00Z"], "symbol": ["ABC"], "s": [0.9]})
    ev = stock_events(news, opens, closes, vol)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert np.isclose(r.oc, 0.05) and np.isclose(r.oc_x, 0.04) and r.pre
    assert r.cost == 7.5e-4  # $1B/day of dollar volume -> large-cap tier


def test_cost_tiers():
    assert list(cost_for([1e8, 2e7, 1e6])) == [7.5e-4, 15e-4, 40e-4]


def test_ticker_mentions_need_a_cashtag_history():
    titles = ["$GME to the moon"] * 30 + ["YOLO on GME calls", "THE DD is in", "$ABC once"]
    wl = cashtag_whitelist(titles, min_count=30)
    assert wl == {"GME"}
    assert ticker_mentions(["YOLO on GME calls", "THE DD"], wl) == [["GME"], []]


def test_mention_spike_uses_only_prior_days():
    c = pd.DataFrame({"GME": [1.0] * 20 + [41.0]})
    ab = mention_spikes(c, window=20)
    assert np.isclose(ab.GME.iloc[-1], 42 / 2)


def test_fade_book_shorts_and_pays_costs():
    ev = pd.DataFrame({"T": pd.to_datetime(["2024-01-02"] * 2), "ab": [12.0, 2.0], "m": [10, 10], "gap": [0.1, 0.1],
                       "oc": [-0.05, 0.02], "o5": [0.0, 0.0], "cost": [0.001, 0.001]})
    tr, day = fade_book(ev, min_ratio=10, borrow_per_day=0.003)
    assert len(tr) == 1
    assert np.isclose(tr.pnl.iloc[0], 0.05 - 0.002 - 0.003)


def test_kalshi_directional_buys_the_signalled_side():
    book = pd.DataFrame({"yes": [0.5, 0.5, 0.5], "result": [1, 1, 0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    sig = pd.Series([2.0, -2.0, 0.0], index=book.index)
    tr = kalshi_directional(book, sig, threshold=1.0, slippage=0.0)
    assert list(tr.side) == ["yes", "no"]
    assert np.isclose(tr.pnl.iloc[0], 0.5 - kalshi_fee(0.5)) and np.isclose(tr.pnl.iloc[1], -0.5 - kalshi_fee(0.5))
    assert kalshi_fee(0.5) == 0.02
