"""Offline tests for the live Kalshi / Polymarket US book helpers."""
import pandas as pd


def test_live_book_parsers_and_keys():
    from backtester.data.live_books import kalshi_game_key, kalshi_top, poly_game_key, poly_us_top

    assert kalshi_top({"orderbook": {"yes": [[40, 10], [42, 5]], "no": [[55, 3]]}}) == (0.42, 0.45)
    assert kalshi_top({"orderbook": {"yes_dollars": [["0.4200", 5]], "no_dollars": None}}) == (0.42, None)
    assert poly_us_top({"marketData": {"bestBid": {"value": "0.41", "currency": "USD"}, "bestAsk": {"value": "0.44"}}}) == (0.41, 0.44)
    assert kalshi_game_key("KXNBAGAME-26JAN30DETGSW") == ("2026-01-30", "DETGSW")
    assert poly_game_key("nba-det-gsw-2026-01-30") == ("2026-01-30", ("DET", "GSW"))


def test_pair_games_side_and_flip():
    from backtester.data.live_books import pair_games

    k = [{"ticker": "KXNBAGAME-26JAN30DETGSW-DET", "event_ticker": "KXNBAGAME-26JAN30DETGSW", "yes_sub_title": "Detroit"},
         {"ticker": "KXNBAGAME-26JAN30DETGSW-GSW", "event_ticker": "KXNBAGAME-26JAN30DETGSW", "yes_sub_title": "Golden State"}]
    one = [{"slug": "nba-gsw-det-2026-01-30", "team": {"abbreviation": "gsw", "name": "Warriors"}}]
    got = {p["kalshi_ticker"]: (p["poly_slug"], p["flip"]) for p in pair_games(k, one)}
    assert got == {k[1]["ticker"]: ("nba-gsw-det-2026-01-30", False), k[0]["ticker"]: ("nba-gsw-det-2026-01-30", True)}
    assert pair_games(k, [{"slug": "nba-det-gsw-2026-01-31", "team": {"abbreviation": "DET"}}]) == []


def test_gap_followthrough_fee_and_direction():
    from backtester.data.live_books import gap_followthrough

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


def test_near_miss_rules_and_paper_pnl():
    from backtester.data.live_books import market_top, paper_entry, paper_pnl

    assert market_top({"yes_bid": 30, "yes_ask": 33}) == (0.30, 0.33)
    assert market_top({"yes_bid_dollars": "0.8500", "yes_ask_dollars": "0.8700"}) == (0.85, 0.87)
    assert market_top({"yes_bid": 0, "yes_ask": 100}) == (None, None)  # empty book
    assert paper_entry("mentions_no", 0.30, 0.33) == ("no", 0.70)  # NO costs 1 - YES bid
    assert paper_entry("mentions_no", 0.85, 0.88) is None  # NO at 15c: outside 20-80c
    assert paper_entry("atp_yes", 0.85, 0.87) == ("yes", 0.87)
    assert paper_entry("atp_yes", 0.94, 0.96) is None
    # 10 contracts at 70c: fee 0.07*10*0.7*0.3 = $0.147 -> $0.15, i.e. 1.5c per contract
    assert abs(paper_pnl("no", 0.70, "no", 10) - (1 - 0.70 - 0.015)) < 1e-12
    assert abs(paper_pnl("no", 0.70, "yes", 10) - (-0.70 - 0.015)) < 1e-12
