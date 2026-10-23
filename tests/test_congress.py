import pandas as pd

from backtester.data import congress


def trades():
    rows = [
        # member, bioguide, party, ticker, side, amount, trade, filing
        ("A", "A1", "Democrat", "XYZ", 1, 1001, "2020-01-01", "2020-01-20"),
        ("B", "B1", "Republican", "XYZ", 1, 50001, "2020-01-05", "2020-01-20"),
        ("C", "P000197", "Democrat", "XYZ", 1, 15001, "2020-02-01", "2020-02-25"),
        ("A", "A1", "Democrat", "XYZ", 1, 1001, "2020-06-01", "2020-06-10"),
        ("A", "A1", "Democrat", "XYZ", -1, 1001, "2020-03-01", "2020-03-05"),
    ]
    df = pd.DataFrame(rows, columns=["member", "bioguide_id", "party", "ticker", "side", "amount_low", "trade_date", "filing_date"])
    for col in ("trade_date", "filing_date"):
        df[col] = pd.to_datetime(df[col])
    df["lag_days"] = (df["filing_date"] - df["trade_date"]).dt.days
    return df


def test_buy_events_key_on_filing_date_and_count_members():
    ev = congress.buy_events(trades())
    assert len(ev) == 3  # two buyers on the same filing day merge into one event
    first = ev.iloc[0]
    assert first["date"] == pd.Timestamp("2020-01-21")  # day after filing, never the trade date
    assert first["n_members"] == 2 and first["amount_low"] == 51002
    second = ev.iloc[1]
    assert second["members_60d"] == 3 and bool(second["leader"])
    assert ev.iloc[2]["members_60d"] == 1  # earlier buys fell out of the 60-day window
