import numpy as np
import pandas as pd

from backtester import Backtester, CostModel, MarketData
from backtester.data.gov_stakes import stake_events
from backtester.data.synthetic import synthetic_bars, synthetic_market
from backtester.data.trump_posts import TWITTER_EPOCH_MS, covered, daily_counts, sessions
from backtester.strategies import BuyAndHold, EventDip, GovStakeEvents, PostRiskOverlay

FREE = CostModel(spread_bps=0, slippage_bps=0)


def posts_at(*times_et):
    t = pd.DatetimeIndex(pd.to_datetime(list(times_et))).tz_localize("America/New_York").tz_convert("UTC")
    return pd.DataFrame({"time": t, "text": "Tariffs are coming!", "url": [str(i) for i in range(len(t))]})


def test_posts_count_for_the_next_close():
    idx = pd.bdate_range("2025-04-07", "2025-04-18")  # Mon 7th .. Fri 18th
    p = posts_at("2025-04-07 15:59", "2025-04-07 16:01", "2025-04-12 10:00", "2025-04-30 09:00")
    pos = sessions(p["time"], idx)
    assert [idx[i] for i in pos[:3]] == [pd.Timestamp(d) for d in ("2025-04-07", "2025-04-08", "2025-04-14")]
    assert pos.iloc[3] == -1  # after the last bar
    counts = daily_counts(p, idx, r"tariff")
    assert counts.sum() == 3 and counts.loc["2025-04-08"] == 1


def test_tweet_id_decodes_to_utc_time():
    # First tweet of the term, posted 07:31 ET on 20 Jan 2017.
    ms = (822421390125043713 >> 22) + TWITTER_EPOCH_MS
    assert pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d %H:%M") == "2017-01-20 12:31"


def test_covered_marks_gaps_in_the_archive():
    idx = pd.bdate_range("2020-12-01", "2022-03-31")
    p = posts_at("2020-12-01 09:00", "2021-01-08 12:00", "2022-02-14 10:00", "2022-03-01 10:00")
    cov = covered(p, idx)
    assert cov.loc["2021-01-08"] and not cov.loc["2021-06-01"] and cov.loc["2022-02-20":"2022-03-01"].all()
    assert not cov.loc["2022-03-02":].any()


def test_overlay_cuts_risky_weight_into_risk_off():
    data = synthetic_market(["UPRO", "SHY"], n=60)
    counts = pd.Series(0, index=data.index)
    counts.iloc[20] = 2
    counts.iloc[22] = 1
    data.extra["trump_counts"] = counts
    s = PostRiskOverlay(base=BuyAndHold(symbols=["UPRO"]), window=5, min_posts=3, hold=2, scale=0.25, risk_off="SHY")
    w = s.generate_targets(data)
    on = s.triggered(data)
    # 3 posts in the trailing 5 sessions on days 22..24 (day 20 drops out on 25),
    # and the cut lasts `hold` sessions from the last burst day: 22..25.
    assert on.iloc[22:26].all() and not on.iloc[21] and not on.iloc[26]
    assert np.allclose(w.loc[on, "UPRO"], 0.25) and np.allclose(w.loc[on, "SHY"], 0.75)
    assert np.allclose(w.loc[~on, "UPRO"], 1.0) and np.allclose(w.loc[~on, "SHY"], 0.0)


def test_event_dip_only_buys_dips_on_event_days():
    bars = synthetic_bars(n=200, sigma=0.4, seed=3)
    data = MarketData({"SPY": bars})
    ret = bars["close"].pct_change()
    dips = ret <= -0.015
    event = pd.Series(False, index=data.index)
    event[dips[dips].index[::2]] = True
    data.extra["ev"] = event
    w_any = EventDip(hold=1).generate_targets(data)["SPY"]
    w_ev = EventDip(hold=1, event_key="ev", mode="require").generate_targets(data)["SPY"]
    w_not = EventDip(hold=1, event_key="ev", mode="exclude").generate_targets(data)["SPY"]
    assert (w_any > 0).equals(dips)
    assert (w_ev > 0).equals(dips & event)
    assert ((w_ev > 0) | (w_not > 0)).equals(dips) and not ((w_ev > 0) & (w_not > 0)).any()


def test_stake_signal_day_is_last_close_before_news_reaches_an_open():
    idx = pd.bdate_range("2025-07-01", "2026-06-30")
    ev = stake_events(idx).set_index(["symbol", "public"])["filed"]
    # Premarket news: signal the day before, so the engine buys that morning's open.
    assert ev.loc[("MP", pd.Timestamp("2025-07-10 07:00"))] == pd.Timestamp("2025-07-09")
    # After-close and weekend news: buy the next open.
    assert ev.loc[("TMQ", pd.Timestamp("2025-10-06 17:00"))] == pd.Timestamp("2025-10-06")
    assert ev.loc[("USAR", pd.Timestamp("2026-01-24 12:00"))] == pd.Timestamp("2026-01-23")


def test_gov_stake_buys_next_open_and_holds():
    data = synthetic_market(["AAA", "BBB"], n=80, start="2025-07-01")
    idx = data.index
    data.extra["gov_stakes"] = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "filed": [idx[10], idx[12]], "outcome": ["deal", "denied"], "public": idx[[10, 12]]}
    )
    res = Backtester(costs=FREE, min_trade_value=0).run(GovStakeEvents(hold_days=5, max_positions=2), data)
    buys = res.trades[res.trades.qty > 0].groupby("symbol")["date"].first()
    assert buys["AAA"] == idx[11] and buys["BBB"] == idx[13]
    assert res.positions["AAA"].iloc[16] == 0 and res.positions["AAA"].iloc[15] > 0
    late = Backtester(costs=FREE).run(GovStakeEvents(hold_days=5, delay=2, outcomes=("deal",)), data)
    assert late.trades.symbol.unique().tolist() == ["AAA"] and late.trades.date.iloc[0] == idx[13]
