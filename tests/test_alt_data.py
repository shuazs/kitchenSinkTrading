import pandas as pd

from backtester.data import alt_data as ad


def test_norm_company_matches_wikipedia_and_sec_spellings():
    assert ad.norm_company("Tesla,_Inc.") == ad.norm_company("TESLA, INC.") == "tesla"
    assert ad.norm_company("Amazon_(company)") == ad.norm_company("AMAZON COM INC".replace(" COM", "")) == "amazon"
    assert ad.norm_company("The Walt Disney Company") == "walt disney"
    assert ad.norm_company("Johnson & Johnson") == "johnson and johnson"


def test_first_in_keeps_only_returns_after_the_gap():
    d = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-03-01", "2020-01-02"])
    e = pd.DataFrame({"d": d, "k": ["a", "a", "a", "b"]})
    out = ad.first_in(e, "k", 30)
    assert list(zip(out.k, out.d.dt.strftime("%m-%d"))) == [("a", "01-01"), ("b", "01-02"), ("a", "03-01")]


def test_nfl_cover_and_over_flags(monkeypatch):
    games = pd.DataFrame({"gameday": ["2024-09-08"] * 4, "result": [7, -3, 3, 10], "spread_line": [3, 3, 3, 0],
                          "total": [40, 41, 30, 50], "total_line": [41, 41, 41, 45]})
    monkeypatch.setattr(ad, "_csv", lambda name, refresh=False, **kw: games.assign(
        gameday=pd.to_datetime(games.gameday)))
    g = ad.nfl_games()
    assert g.fav_cover.tolist()[:2] == [1.0, 0.0]
    assert g.fav_cover.isna().tolist()[2:] == [True, True]  # push, pick'em
    assert g.over.tolist()[0] == 0.0 and pd.isna(g.over.tolist()[1]) and g.over.tolist()[3] == 1.0
