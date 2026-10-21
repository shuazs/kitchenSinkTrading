"""US government equity stakes in listed companies, 2025-2026, by first public report.

`public` is when the news first reached the market (US Eastern time), checked
against each stock's opening gap in yfinance bars: a report before the open
shows up as a gap at that day's open, one during or after the session shows
up in that day's close or the next open. Sources are news reports collected
on 2026-10-21; the list is hand-built, so it misses deals in private
companies (Westinghouse, Vulcan Elements) and small letters of intent.

outcome: "deal" = a stake was agreed; "denied" = the White House denied the
report; "talks" = reported talks with no deal for that company as of
2026-10-21.
"""
from __future__ import annotations

import pandas as pd

STAKES = [
    # symbol, public (ET), outcome, what was reported
    ("MP", "2025-07-10 07:00", "deal", "Defense Dept buys $400M preferred, about 15%"),
    ("INTC", "2025-08-14 13:00", "deal", "Bloomberg: talks on a stake; 10% via CHIPS grants agreed 22 Aug"),
    ("LAC", "2025-09-23 17:00", "deal", "Reuters: Energy Dept seeks a stake; 5% agreed 30 Sep"),
    ("CRML", "2025-10-06 07:00", "denied", "Reuters: talks on an 8% stake; White House denied it that day"),
    ("TMQ", "2025-10-06 17:00", "deal", "10% stake plus warrants tied to the Ambler Road project"),
    ("IONQ", "2025-10-23 07:00", "talks", "WSJ: quantum firms in talks over stakes for funding"),
    ("RGTI", "2025-10-23 07:00", "deal", "WSJ: quantum firms in talks; deal announced 21 May 2026"),
    ("QBTS", "2025-10-23 07:00", "deal", "WSJ: quantum firms in talks; deal announced 21 May 2026"),
    ("LHX", "2026-01-13 07:00", "deal", "$1B convertible preferred in the missile solutions unit"),
    ("USAR", "2026-01-24 12:00", "deal", "FT (Saturday): 10% stake in a $1.6B package"),
    ("GFS", "2026-05-21 07:00", "deal", "Commerce $2B quantum package; about 1% of GlobalFoundries"),
    ("RGTI", "2026-05-21 07:00", "deal", "Commerce $2B quantum package, about $100M each"),
    ("QBTS", "2026-05-21 07:00", "deal", "Commerce $2B quantum package, about $100M each"),
]


def stake_events(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Events with `public` time and, given a trading `index`, the signal day.

    The signal day is the last session that closed before the news, if it came
    out before the next open (so the engine buys at the first open after it,
    which already has the gap in it), else the session the news came out in.
    """
    ev = pd.DataFrame(STAKES, columns=["symbol", "public", "outcome", "note"])
    ev["public"] = pd.to_datetime(ev["public"])
    if index is not None:
        # First session whose 09:30 open is after the news.
        opens = index + pd.Timedelta(hours=9, minutes=30)
        entry = opens.searchsorted(ev["public"], side="right")
        ev["filed"] = [index[i - 1] if 0 < i <= len(index) else pd.NaT for i in entry]
    return ev
