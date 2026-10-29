"""Paper-trade the short across FDA PDUFA decisions; logs only, never places orders.

    python paper_loggers/pdufa_decision_short.py              # daily after 4pm ET
    python paper_loggers/pdufa_decision_short.py --days 120   # first run: seed the calendar
    python paper_loggers/pdufa_decision_short.py --report     # net return per event so far

Rule (examples/run_nonfin_data.py batch 4, NEAR): for a small or mid-cap
drug maker (not in `alt_data.BIG_PHARMA`, price >= $1) whose 8-K names a
PDUFA target date at least 5 trading days ahead, short the stock at the close
2 trading days before the date (day 0 = first trading day on or after it) and
buy the same dollars of XBI; cover both at the close of day +1. Backtest:
+5.5% per event in-sample (t 5.4), +4.6% Jul 2024+ (t 2.3), net of 15 bp/side
but before borrow fees. The biggest risks are borrow (small biotechs are often
hard to borrow) and approval spikes (up to +96% in the backtest), so log the
broker's borrow rate in `borrow_pct` (annual %) for each trade.

Each run searches EDGAR full-text search for 8-Ks mentioning PDUFA in the last
--days days, adds new (ticker, PDUFA date) pairs to
~/.cache/kitchensink/paper/pdufa_decision_short/events.csv, prints today's
paper shorts and covers, and fills in closes and the net result. Needs
SEC_USER_AGENT in keys.env. Graduates after 30 events at +2.3% or better per
event after borrow.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, need, read_log, today, trading_days, tstat, write_log
from backtester.data import alt_data as ad

LOG = log_dir("pdufa_decision_short") / "events.csv"
FIELDS = ["ticker", "cik", "pdufa", "first", "day0", "entry_date", "exit_date", "status", "entry", "xbi_entry",
          "exit", "xbi_exit", "borrow_pct", "gross", "net"]
COST, BAR, LEAD = 0.003, 0.023, 5


def schedule(pdufa: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(day 0, entry day -2, exit day +1) on the NYSE calendar."""
    days = trading_days(pdufa - pd.Timedelta(days=10), pdufa + pd.Timedelta(days=10))
    k = days.searchsorted(pdufa)
    return days[k], days[k - 2], days[k + 1]


def scan(log: pd.DataFrame, days: int, ua: str) -> pd.DataFrame:
    day = today()
    f = ad.pdufa_filings(f"{day - pd.Timedelta(days=days):%Y-%m-%d}", f"{day:%Y-%m-%d}", ua)
    e = ad.pdufa_events(f, ad.cik_tickers(ua)).dropna(subset=["ticker"])
    known = set(zip(log.ticker, pd.to_datetime(log.pdufa)))
    rows = []
    for r in e.itertuples():
        if (r.ticker, r.pdufa) in known or r.pdufa < day:
            continue
        d0, entry, exit_ = schedule(r.pdufa)
        lead = len(trading_days(r.first + pd.Timedelta(days=1), d0)) - 1
        why = ("big pharma" if r.ticker in ad.BIG_PHARMA else "not a common share" if "-" in r.ticker else
               f"date public only {lead} trading days ahead" if lead < LEAD else "")
        rows.append({"ticker": r.ticker, "cik": r.cik, "pdufa": r.pdufa, "first": r.first, "day0": d0,
                     "entry_date": entry, "exit_date": exit_, "status": f"skip: {why}" if why else "scheduled"})
        print(f"{r.first:%Y-%m-%d} {r.ticker:<6} PDUFA {r.pdufa:%Y-%m-%d} -> {rows[-1]['status']}")
    return pd.concat([log, pd.DataFrame(rows)], ignore_index=True) if rows else log


def update(log: pd.DataFrame) -> pd.DataFrame:
    day = today()
    live = log.index[log.status.isin(["scheduled", "short"])]
    for i in live:
        tk = log.at[i, "ticker"]
        entry, exit_ = pd.Timestamp(log.at[i, "entry_date"]), pd.Timestamp(log.at[i, "exit_date"])
        if day == entry:
            print(f"  TODAY at the close: PAPER SHORT {tk}, BUY the same $ of XBI (PDUFA {log.at[i, 'pdufa']})")
        if day == exit_:
            print(f"  TODAY at the close: COVER {tk}, SELL XBI")
        if day < entry:
            continue
        px = closes([tk, "XBI"], entry - pd.Timedelta(days=7))
        if tk not in px:
            log.at[i, "status"] = "skip: no Yahoo price"
            continue
        if log.at[i, "status"] == "scheduled":
            if entry not in px.index or pd.isna(px.at[entry, tk]):
                if day > entry + pd.Timedelta(days=3):
                    log.at[i, "status"] = "skip: no close on entry day"
                continue
            s0 = float(px.at[entry, tk])
            log.loc[i, ["entry", "xbi_entry", "status"]] = [s0, float(px.at[entry, "XBI"]),
                                                            "short" if s0 >= 1 else "skip: price < $1"]
        if log.at[i, "status"] == "short" and exit_ in px.index and pd.notna(px.at[exit_, tk]):
            s0, b0 = float(log.at[i, "entry"]), float(log.at[i, "xbi_entry"])
            s1, b1 = float(px.at[exit_, tk]), float(px.at[exit_, "XBI"])
            gross = -(s1 / s0 - 1) + (b1 / b0 - 1) - COST
            borrow = pd.to_numeric(log.at[i, "borrow_pct"], errors="coerce")
            held = len(trading_days(entry, exit_)) - 1
            net = gross - (0 if pd.isna(borrow) else borrow / 100 * held / 252)
            log.loc[i, ["exit", "xbi_exit", "gross", "net", "status"]] = [s1, b1, round(gross, 4), round(net, 4), "done"]
            print(f"  DONE {tk}: {s0:.2f} -> {s1:.2f}, XBI {b0:.2f} -> {b1:.2f}: {net:+.2%} net")
    return log


def run(days: int = 10) -> None:
    ua = need("SEC_USER_AGENT")
    write_log(update(scan(read_log(LOG, FIELDS), days, ua)), LOG)


def report() -> None:
    log = read_log(LOG, FIELDS)
    net = pd.to_numeric(log.net, errors="coerce").dropna()
    print(f"{len(log)} PDUFA dates seen, {log.status.eq('scheduled').sum()} scheduled, {log.status.eq('short').sum()} "
          f"open, {len(net)} done: mean {net.mean():+.2%} per event, t {tstat(net):.2f}; "
          f"graduates at >= {BAR:+.1%} over 30+ events")
    print(log[~log.status.str.startswith("skip")].tail(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="look back this many days for new 8-Ks")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run(args.days)


if __name__ == "__main__":
    main()
