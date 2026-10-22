import numpy as np
import pandas as pd
import pytest

from backtester.event_markets import (
    arb_episodes, bankroll, bracket_arb, calibration, cross_arb, exclusive_events, favourite_bets,
    first_entries, is_financial_text, kalshi_category, kalshi_fee, lead_follow, snapshot_quotes, taker_prints,
)

T0 = pd.Timestamp("2025-06-01 18:00", tz="UTC")


def ts(sec):
    return T0 + pd.Timedelta(seconds=sec)


def test_categories():
    assert kalshi_category("KXNBAGAME") == "sports"
    assert kalshi_category("KXHIGHNY") == "weather"
    assert kalshi_category("PRES") == "politics"
    assert kalshi_category("KXTRUMPMENTION") == "mentions"
    assert kalshi_category("KXBTCD") == "finance"
    assert kalshi_category("KXEUROLEAGUEGAME") == "sports"
    assert kalshi_category("KXMVENFLSINGLEGAME") == "parlay"
    assert is_financial_text("Will Bitcoin hit $150k?") and not is_financial_text("Lakers vs. Celtics")


def markets():
    return pd.DataFrame({
        "ticker": ["E-A", "E-B"], "event_ticker": ["E", "E"], "series": ["S", "S"], "cat": ["sports"] * 2,
        "result": ["yes", "no"], "close_time": [T0 + pd.Timedelta("3h")] * 2,
    })


def test_taker_prints_known_answer():
    tr = pd.DataFrame({"ticker": ["E-A", "E-A"], "count": [10, 5], "yes_price": [60, 60],
                       "taker_side": ["yes", "no"], "created_time": [T0, T0]})
    p = taker_prints(tr, markets())
    yes, no = p.iloc[0], p.iloc[1]
    assert yes["price"] == pytest.approx(0.60) and yes["won"] == 1
    assert yes["net"] == pytest.approx(1 - 0.60 - 0.07 * 0.6 * 0.4)
    assert no["price"] == pytest.approx(0.40) and no["won"] == 0
    assert no["net"] == pytest.approx(-0.40 - 0.07 * 0.4 * 0.6)
    assert p["tau_h"].iloc[0] == pytest.approx(3.0)
    c = calibration(p, by=("cat",))
    assert c["contracts"].sum() == 15
    assert set(c.columns) >= {"price", "hit", "taker_net", "maker_gross", "t"}


def test_exclusive_events():
    m = pd.DataFrame({"ticker": ["A1", "A2", "B1", "B2", "C1", "C2"], "event_ticker": ["A", "A", "B", "B", "C", "C"],
                      "series": ["S"] * 4 + ["T"] * 2, "result": ["yes", "no", "no", "yes", "yes", "yes"]})
    ex = exclusive_events(m)
    assert set(ex.index) == {"A", "B"} and ex["A"] == 2


def ladder(prices, secs, side="yes", count=10):
    return pd.DataFrame({"ticker": [f"M{i}" for i in range(len(prices))], "count": count, "yes_price": prices,
                         "taker_side": side, "created_time": [ts(s) for s in secs]})


def test_bracket_arb_needs_fresh_legs_and_beats_fees():
    p = ladder([40, 50], [0, 5])  # 0.90 for a $1 set
    a = bracket_arb(p, 2, "yes", "10s")
    fee = 0.07 * (0.4 * 0.6 + 0.5 * 0.5)
    assert len(a) == 1 and a["edge"].iloc[0] == pytest.approx(1 - 0.9 - fee)
    assert bracket_arb(p, 2, "yes", "2s").empty  # first leg too old
    assert bracket_arb(ladder([40], [0]), 2, "yes").empty  # a leg never printed
    ep = arb_episodes(a, legs=2)
    assert ep["profit"].iloc[0] == pytest.approx(10 * (1 - 0.9 - fee) - 0.01)
    fair = bracket_arb(ladder([50, 51], [0, 1]), 2, "yes", "10s")
    assert arb_episodes(fair).empty


def test_bracket_arb_no_side():
    p = ladder([70, 70, 70], [0, 1, 2], side="no")  # NO at 0.30 each, pays $2
    a = bracket_arb(p, 3, "no", "10s")
    assert a["edge"].iloc[-1] == pytest.approx(2 - 0.9 - 3 * 0.07 * 0.3 * 0.7)


def test_cross_arb_matches_within_window():
    k = pd.DataFrame({"time": [ts(0), ts(30)], "price": [0.55, 0.55], "size": [10, 10]})
    p = pd.DataFrame({"time": [ts(3)], "price": [0.40], "size": [4.0]})
    a = cross_arb(k, p, "10s", 0.07, 0.0)
    assert len(a) == 1
    assert a["edge"].iloc[0] == pytest.approx(1 - 0.95 - 0.07 * 0.55 * 0.45)
    assert a["size"].iloc[0] == 4


def test_snapshot_and_favourite():
    pr = pd.DataFrame({"market": ["m1", "m1", "m1", "m2", "m2"],
                       "time": [ts(0), ts(10), ts(4000), ts(0), ts(10)],
                       "yes": [0.70, 0.68, 0.10, 0.30, 0.28], "taker_side": ["yes", "no", "yes", "yes", "no"]})
    at = pd.Series({"m1": ts(60), "m2": ts(60)})
    q = snapshot_quotes(pr, at, "30min").set_index("market")
    assert q.loc["m1", "ask"] == 0.70 and q.loc["m1", "bid"] == 0.68  # later print ignored
    b = favourite_bets(q.reset_index(), pd.Series({"m1": True, "m2": True})).set_index("market")
    assert b.loc["m1", "side"] == "yes" and b.loc["m1", "price"] == pytest.approx(0.70)
    assert b.loc["m2", "side"] == "no" and b.loc["m2", "price"] == pytest.approx(0.72)
    assert b.loc["m1", "ret"] == pytest.approx(0.3 / 0.7) and b.loc["m2", "ret"] == pytest.approx(-1)


def test_lead_follow_known_answer_and_latency():
    lead = pd.DataFrame({"time": [ts(100)], "yes": [0.60]})
    fol = pd.DataFrame({"time": [ts(90), ts(95), ts(101), ts(105), ts(500)],
                        "yes": [0.50, 0.49, 0.51, 0.52, 0.58],
                        "taker_side": ["yes", "no", "yes", "yes", "no"], "size": 10.0})
    r = lead_follow(lead, fol, 0.05, "2s", "5min", "60s")
    assert len(r) == 1
    assert r["p_in"].iloc[0] == 0.52  # first ask print at or after t + 2s, not the one at t + 1s
    assert r["ret"].iloc[0] == pytest.approx(0.58 - 0.52)
    assert lead_follow(lead, fol, 0.20, "2s", "5min").empty
    short = lead.assign(yes=0.40)  # follower too high: sell at bid, buy back at ask
    fol2 = fol.assign(taker_side=["yes", "no", "no", "no", "yes"])
    r2 = lead_follow(short, fol2, 0.05, "2s", "5min")
    assert r2["ret"].iloc[0] == pytest.approx(0.52 - 0.58)


def test_first_entries_and_bankroll():
    tr = pd.DataFrame({"ticker": ["E-A"] * 3, "count": [100, 100, 100], "yes_price": [50, 70, 75],
                       "taker_side": ["yes"] * 3, "created_time": [ts(0), ts(60), ts(120)]})
    p = taker_prints(tr, markets())
    e = first_entries(p, 0.65, 0.8)
    assert len(e) == 1 and e["price"].iloc[0] == pytest.approx(0.70)
    led = bankroll(e, start=1000, stake=0.1)
    n = min(int(100 // (0.7 + 0.07 * 0.7 * 0.3)), 100)  # capped at the print size
    assert led["contracts"].iloc[0] == n
    assert led["equity"].iloc[-1] == pytest.approx(1000 + n * (1 - 0.7 - 0.07 * 0.7 * 0.3))
    assert np.isfinite(kalshi_fee(0.5))


def test_name_matching():
    from backtester.event_markets import match_by_name, name_tokens, poly_subject
    assert name_tokens("Club Atlético de Madrid") == frozenset({"atletico", "madrid"})
    assert poly_subject("Will Chelsea FC win on 2026-01-07?", "Yes") == "Chelsea FC"
    assert poly_subject("Will Chelsea FC vs. Fulham end in a draw?", "Yes") == "tie"
    assert poly_subject("Bucks vs. Hornets", "Bucks") == "Bucks"
    d = pd.Timestamp("2026-01-07")
    k = pd.DataFrame({"ticker": ["G-CHE", "G-FUL", "G-TIE", "H-CHE"], "event_ticker": ["G", "G", "G", "H"],
                      "subject": ["Chelsea", "Fulham", "Tie", "Chelsea"],
                      "date": [d, d, d, d + pd.Timedelta("10D")]})
    p = pd.DataFrame({"id": ["1", "2", "3"], "subject": ["Chelsea FC", "tie", "Arsenal"], "date": [d] * 3})
    m = match_by_name(k, p, "2D").set_index("id")
    assert m.loc["1", "ticker"] == "G-CHE" and m.loc["2", "ticker"] == "G-TIE" and "3" not in m.index


def test_live_book_parsers_and_keys():
    from backtester.data.live_books import kalshi_game_key, kalshi_top, poly_game_key, poly_us_top

    assert kalshi_top({"orderbook": {"yes": [[40, 10], [42, 5]], "no": [[55, 3]]}}) == (0.42, 0.45)
    assert kalshi_top({"orderbook": {"yes_dollars": [["0.4200", 5]], "no_dollars": None}}) == (0.42, None)
    assert poly_us_top({"marketData": {"bestBid": {"value": "0.41", "currency": "USD"}, "bestAsk": {"value": "0.44"}}}) == (0.41, 0.44)
    assert kalshi_game_key("KXNBAGAME-26JAN30DETGSW") == ("2026-01-30", "DETGSW")
    assert poly_game_key("nba-det-gsw-2026-01-30") == ("2026-01-30", ("DET", "GSW"))


def test_pair_games_side_and_flip():
    from backtester.event_markets import pair_games

    k = [{"ticker": "KXNBAGAME-26JAN30DETGSW-DET", "event_ticker": "KXNBAGAME-26JAN30DETGSW", "yes_sub_title": "Detroit"},
         {"ticker": "KXNBAGAME-26JAN30DETGSW-GSW", "event_ticker": "KXNBAGAME-26JAN30DETGSW", "yes_sub_title": "Golden State"}]
    one = [{"slug": "nba-gsw-det-2026-01-30", "team": {"abbreviation": "gsw", "name": "Warriors"}}]
    got = {p["kalshi_ticker"]: (p["poly_slug"], p["flip"]) for p in pair_games(k, one)}
    assert got == {k[1]["ticker"]: ("nba-gsw-det-2026-01-30", False), k[0]["ticker"]: ("nba-gsw-det-2026-01-30", True)}
    assert pair_games(k, [{"slug": "nba-det-gsw-2026-01-31", "team": {"abbreviation": "DET"}}]) == []


def test_gap_followthrough_fee_and_direction():
    from backtester.event_markets import gap_followthrough

    t = pd.date_range("2026-01-01 00:00", periods=6, freq="30s", tz="UTC")
    q = pd.DataFrame({"time": t, "kalshi_ticker": "A",
                      "k_bid": [.49, .59, .59, .59, .59, .59], "k_ask": [.51, .61, .61, .61, .61, .61],
                      "p_bid": [.49, .49, .52, .55, .58, .58], "p_ask": [.51, .51, .54, .57, .60, .60]})
    r = gap_followthrough(q, 0.05, "60s", "60s", rate=0.05)
    assert len(r) == 1
    row = r.iloc[0]
    assert row["entry"] == t[3] and row["exit"] == t[5]  # signal at t1, enter 60s later, exit 60s after that
    assert row["p_in"] == 0.57 and row["p_out"] == 0.58
    assert abs(row["ret"] - (0.01 - 0.05 * (0.57 * 0.43 + 0.58 * 0.42))) < 1e-12
    assert abs(row["toward"] - 0.09) < 1e-12
