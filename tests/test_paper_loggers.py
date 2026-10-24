"""Offline checks of the paper loggers' rule logic (no network)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_loggers"))

import _common  # noqa: E402
from backtester.data.tender_offers import parse_offer  # noqa: E402
from backtester.letf import pressure  # noqa: E402


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KS_CACHE_DIR", str(tmp_path))


def test_load_keys_skips_blanks_and_keeps_shell(tmp_path, monkeypatch):
    f = tmp_path / "keys.env"
    f.write_text("# comment\nA_KEY='x y'\nB_KEY=\nC_KEY=new\n")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "shell")
    assert _common.load_keys(f) == {"A_KEY": "x y", "C_KEY": "new"}
    import os
    assert os.environ["A_KEY"] == "x y" and os.environ["C_KEY"] == "shell" and "B_KEY" not in os.environ


def test_keys_template_lists_every_needed_key():
    text = (Path(_common.HERE) / "keys.example.env").read_text()
    for k in ("SEC_USER_AGENT", "KS_CACHE_DIR", "APCA_API_KEY_ID", "REDDIT_CLIENT_ID"):
        assert f"\n{k}=\n" in text


def test_nyse_calendar():
    d = _common.trading_days("2026-11-23", "2026-11-30")
    assert pd.Timestamp("2026-11-26") not in d  # Thanksgiving
    assert list(_common.trading_days("2027-01-01", "2027-01-31")[[4, 9]]) == [pd.Timestamp("2027-01-08"),
                                                                            pd.Timestamp("2027-01-15")]
    assert pd.Timestamp("2026-04-03") not in _common.trading_days("2026-03-30", "2026-04-06")  # Good Friday


def test_write_log_dates(tmp_path):
    p = tmp_path / "x.csv"
    _common.write_log(pd.DataFrame({"d": [pd.Timestamp("2026-01-02"), None], "v": [1, 2]}, dtype=object), p)
    assert p.read_text().splitlines()[1] == "2026-01-02,1"


def test_auction_cycle_schedule(monkeypatch):
    import treasury_auction_cycle as m

    monkeypatch.setattr(m, "treasurydirect_dates", lambda: pd.DatetimeIndex(["2026-11-12", "2026-11-13", "2026-12-09"]))
    monkeypatch.setattr(m, "long_bond_auction_dates", lambda: pd.DatetimeIndex([]))
    s = m.schedule(pd.Timestamp("2026-11-01"))
    assert list(s.exit_date) == [pd.Timestamp("2026-11-05"), pd.Timestamp("2026-12-02")]
    assert list(s.back_date) == [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-12-09")]


def test_letf_pressure():
    idx = pd.bdate_range("2026-01-01", periods=80)
    c = pd.DataFrame({"TSLA": 100.0, "TSLL": 10.0}, index=idx)
    v = pd.DataFrame({"TSLA": 1000.0, "TSLL": 500.0}, index=idx)
    p = pressure(c, v, "TSLA", {"TSLL": 2})
    assert p.iloc[-1] == pytest.approx(2 * 10 * 500 / (100 * 1000))


def test_kalshi_best_quote():
    import kalshi_sp_catchup as k

    ms = [{"ticker": f"KXINXU-X-T{K}", "K": float(K), "yes_bid": b, "yes_ask": a}
          for K, b, a in [(6700, 60, 62), (6750, 30, 33), (6800, 8, 10)]]
    up = k.best_quote(ms, 6760, 0.0006, 20, True)
    assert up["ticker"].endswith("T6750") and up["side"] == "yes" and up["edge"] > 0.3
    assert k.best_quote(ms, 6740, 0.0006, 20, False)["side"] == "no"


def test_parse_fixed_odd_lot_offer():
    t = ("the company is offering to purchase up to 1,000,000 shares at a purchase price of $12.50 per share. "
         "shares tendered by any odd lot holder will not be subject to proration. "
         "the offer will expire at 12:00 midnight, new york city time, on december 4, 2026.")
    d = parse_offer(t)
    assert d["fixed"] == 12.5 and d["odd"] and not d["dutch"] and d["exp"] == "december 4 2026"
