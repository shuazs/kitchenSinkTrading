import io
import zipfile

import pandas as pd
import pytest

from backtester import Backtester, CostModel
from backtester.data.insider_bulk import _parse_zip
from backtester.data.synthetic import synthetic_market
from backtester.strategies.insider import InsiderCluster, cluster_signals

FREE = CostModel(spread_bps=0, slippage_bps=0)


def buys(rows):
    """rows of (symbol, filed, owner, value, officer)."""
    df = pd.DataFrame(rows, columns=["symbol", "filed", "owner_cik", "value", "officer"])
    return df.assign(filed=pd.to_datetime(df["filed"]), code="P", director=True)


def test_cluster_needs_distinct_insiders_in_window_and_value():
    t = buys([
        ("A", "2015-03-02", 1, 60_000, True),
        ("A", "2015-03-02", 1, 60_000, True),  # same insider twice is still one
        ("A", "2015-03-20", 2, 50_000, False),  # second insider, inside 30 days
        ("B", "2015-03-02", 3, 60_000, True),
        ("B", "2015-05-01", 4, 60_000, True),  # too far apart
        ("C", "2015-03-02", 5, 10_000, True),
        ("C", "2015-03-03", 6, 10_000, True),  # too small
    ])
    ev = cluster_signals(t, min_insiders=2, window=30, min_value=100_000)
    assert ev["symbol"].tolist() == ["A"]
    assert ev["filed"].iloc[0] == pd.Timestamp("2015-03-20")
    assert ev["n_insiders"].iloc[0] == 2 and ev["n_officers"].iloc[0] == 1
    assert ev["value"].iloc[0] == pytest.approx(170_000)
    assert cluster_signals(t, min_value=0, window=10)["symbol"].tolist() == ["C"]
    assert len(cluster_signals(t.assign(officer=False), min_value=0, require_officer=True)) == 0


def test_sales_and_pure_ten_percent_holders_ignored():
    t = buys([("A", "2015-03-02", 1, 1e6, False), ("A", "2015-03-03", 2, 1e6, False)])
    t.loc[0, "code"] = "S"
    assert cluster_signals(t, min_value=0).empty
    t = buys([("A", "2015-03-02", 1, 1e6, False), ("A", "2015-03-03", 2, 1e6, False)]).assign(director=False)
    assert cluster_signals(t, min_value=0).empty
    assert len(cluster_signals(t, min_value=0, insiders_only=False)) == 1


def events(rows):
    df = pd.DataFrame(rows, columns=["symbol", "filed", "n_insiders", "value"])
    return df.assign(filed=pd.to_datetime(df["filed"]), n_officers=1)


def test_enters_day_after_filing_and_exits_after_hold():
    data = synthetic_market(["AAA", "BBB", "CCC"], n=200)
    filed = data.index[20]
    data.extra["insider_events"] = events([("BBB", filed, 2, 200_000)])
    res = Backtester(10_000, costs=FREE).run(InsiderCluster(hold_days=10, max_positions=4), data)
    buys_, sells = res.trades[res.trades["qty"] > 0], res.trades[res.trades["qty"] < 0]
    assert buys_["date"].tolist() == [data.index[21]]  # next open, never the filing day
    assert buys_["symbol"].iloc[0] == "BBB"
    assert buys_["fill_price"].iloc[0] == pytest.approx(data.bars["BBB"]["open"].iloc[21])
    assert sells["date"].tolist() == [data.index[31]]
    assert res.weights["BBB"].iloc[25] == pytest.approx(0.25, abs=0.03)


def test_weekend_filing_waits_for_next_trading_day():
    data = synthetic_market(["AAA"], n=60)  # business days only
    sat = data.index[9] + pd.Timedelta(days=(5 - data.index[9].weekday()) % 7)
    data.extra["insider_events"] = events([("AAA", sat, 2, 200_000)])
    res = Backtester(costs=FREE).run(InsiderCluster(hold_days=5), data)
    signal_day = data.index[data.index.searchsorted(sat)]
    assert res.trades["date"].iloc[0] == data.index[data.index.get_loc(signal_day) + 1]


def test_slot_limit_ranking_and_liquidity_filter():
    data = synthetic_market(["AAA", "BBB", "CCC", "DDD"], n=120)
    data.bars["DDD"]["volume"] = 10  # illiquid: filtered out
    d = data.index[30]
    data.extra["insider_events"] = events([
        ("AAA", d, 2, 150_000),
        ("BBB", d, 3, 120_000),
        ("CCC", d, 2, 900_000),
        ("DDD", d, 5, 9_000_000),
    ])
    strat = InsiderCluster(hold_days=20, max_positions=2)
    res = Backtester(costs=FREE).run(strat, data)
    assert strat.taken["symbol"].tolist() == ["BBB", "CCC"]  # most insiders, then most dollars
    assert set(res.trades["symbol"]) == {"BBB", "CCC"}
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_held_positions_are_not_rebalanced_between_events():
    data = synthetic_market(["AAA", "BBB"], n=150)
    data.extra["insider_events"] = events([("AAA", data.index[10], 2, 2e5), ("BBB", data.index[40], 2, 2e5)])
    res = Backtester(costs=FREE, min_trade_value=25).run(InsiderCluster(hold_days=60, max_positions=2), data)
    # One buy each and one sell each; no drift trades on AAA when BBB enters.
    assert len(res.trades) == 4


def test_parse_form345_zip():
    sub = (
        "ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\n"
        "a1\t02-MAR-2023\t4\t0000000123\tACME\tACME\n"
        "a2\t03-MAR-2023\t4/A\t0000000123\tACME\tACME\n"
    )
    tx = (
        "ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
        "a1\t01-MAR-2023\tP\t1000\t10.5\tA\n"
        "a1\t01-MAR-2023\tA\t500\t0\tA\n"
        "a2\t01-MAR-2023\tP\t1000\t10.5\tA\n"
    )
    own = (
        "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
        "a1\t0000000999\tDoe Jane\tDirector,Officer\tCEO\n"
        "a2\t0000000999\tDoe Jane\tDirector,Officer\tCEO\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("NONDERIV_TRANS.tsv", tx)
        z.writestr("REPORTINGOWNER.tsv", own)
    df = _parse_zip(buf.getvalue())
    assert len(df) == 1  # grants (A) and amendments (4/A) dropped
    row = df.iloc[0]
    assert row["code"] == "P" and row["value"] == pytest.approx(10_500)
    assert row["filed"] == pd.Timestamp("2023-03-02") and row["trans_date"] == pd.Timestamp("2023-03-01")
    assert row["officer"] and row["director"] and not row["ten_pct"]
    assert row["issuer_cik"] == 123 and row["owner_cik"] == 999
