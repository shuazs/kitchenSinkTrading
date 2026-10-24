"""Paper-trade the closed-end fund rights-offering short; logs only.

    python paper_loggers/cef_rights_short.py            # daily: new rights prospectuses + update open shorts
    python paper_loggers/cef_rights_short.py --report   # net relative return per event so far

Rule (examples/run_alpha_batch8.py `rights_offerings`, batch 8): when a
listed closed-end fund files a rights-offering prospectus (497 / 424B*),
short it at the next close against a CEF basket and cover 25 trading days
later. The backtest's basket was every listed CEF; live, PCEF (Invesco CEF
Income Composite ETF) stands in for it. One event per fund per 120 days.

Each run searches EDGAR full-text search for new prospectuses, keeps funds
that file N-2 / N-CSR reports (registered closed-end funds) and have a ticker, and logs them
to ~/.cache/kitchensink/paper/cef_rights_short/events.csv. Fill in
`borrow_pct` (the broker's annual borrow rate on the day you would short;
3 is assumed when blank). Net = -(fund - PCEF) - 0.5% spreads - borrow x
25/252. Needs SEC_USER_AGENT. Graduates after 10 events at +1% or better.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, read_log, sec_get, today, trading_days, tstat, write_log
from odd_lot_tenders import search, ticker_map

LOG = log_dir("cef_rights_short") / "events.csv"
FIELDS = ["file_date", "cik", "company", "ticker", "status", "entry_date", "fund_entry", "bench_entry", "exit_date",
          "fund_exit", "bench_exit", "rel_short", "borrow_pct", "net"]
BENCH, HOLD, SPREAD, BORROW, BAR = "PCEF", 25, 0.005, 3.0, 0.01
QUERY = '"rights offering" "subscription period" "net asset value"'
FORMS = "497,424B1,424B2,424B3,424B5"


def is_cef(cik: int) -> bool:
    """A registered closed-end fund files N-2 registrations and N-CSR shareholder reports."""
    forms = set(sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]["form"])
    return bool(forms & {"N-CSR", "N-CSRS", "N-2", "N-2/A", "N-2ASR"})


def scan(log: pd.DataFrame, days: int) -> pd.DataFrame:
    day = today()
    hits = search(day - pd.Timedelta(days=days), day, QUERY, FORMS)
    tk = ticker_map()
    last = pd.to_datetime(log.groupby("ticker").file_date.max()).to_dict() if len(log) else {}
    rows = []
    for h in sorted(hits, key=lambda h: h["_source"]["file_date"]):
        src = h["_source"]
        cik, d = int(src["ciks"][0]), pd.Timestamp(src["file_date"])
        tick = tk.get(cik)
        if not tick or (tick in last and (d - last[tick]).days < 120):
            continue
        last[tick] = d
        status = "open" if is_cef(cik) else "skip: not a listed CEF"
        rows.append({"file_date": d, "cik": cik, "company": src["display_names"][0][:60], "ticker": tick, "status": status})
        print(f"{d:%Y-%m-%d} {tick:<6} {status:<24} {src['display_names'][0][:50]}")
    return pd.concat([log, pd.DataFrame(rows)], ignore_index=True) if rows else log


def update(log: pd.DataFrame) -> pd.DataFrame:
    live = log.index[log.status.isin(["open", "short"])]
    if not len(live):
        return log
    start = pd.to_datetime(log.loc[live, "file_date"]).min() - pd.Timedelta(days=5)
    px = closes(sorted(set(log.loc[live, "ticker"])) + [BENCH], start)
    for i in live:
        tk, filed = log.at[i, "ticker"], pd.Timestamp(log.at[i, "file_date"])
        if tk not in px:
            continue
        p = px[[tk, BENCH]].dropna()
        if log.at[i, "status"] == "open":
            j = p.index.searchsorted(filed, side="right")
            if j < len(p):
                exit_d = trading_days(p.index[j], p.index[j] + pd.Timedelta(days=60))[HOLD]
                log.loc[i, ["entry_date", "fund_entry", "bench_entry", "exit_date", "status"]] = \
                    [p.index[j], round(p.iat[j, 0], 4), round(p.iat[j, 1], 4), exit_d, "short"]
                print(f"  PAPER SHORT {tk} at {p.iat[j, 0]:.2f} vs {BENCH} {p.iat[j, 1]:.2f}; cover at close {exit_d:%Y-%m-%d}")
            else:
                continue
        k = p.index.searchsorted(pd.Timestamp(log.at[i, "exit_date"]))
        if k < len(p):
            exit_d = p.index[k]
            f = p.at[exit_d, tk] / float(log.at[i, "fund_entry"]) - 1
            b = p.at[exit_d, BENCH] / float(log.at[i, "bench_entry"]) - 1
            borrow = float(log.at[i, "borrow_pct"]) if pd.notna(log.at[i, "borrow_pct"]) else BORROW
            net = -(f - b) - SPREAD - borrow / 100 * HOLD / 252
            log.loc[i, ["fund_exit", "bench_exit", "rel_short", "net", "status"]] = \
                [round(p.at[exit_d, tk], 4), round(p.at[exit_d, BENCH], 4), round(-(f - b), 4), round(net, 4), "done"]
            print(f"  COVER {tk}: short vs {BENCH} {-(f - b):+.2%}, net {net:+.2%}")
    return log


def run(days: int = 14) -> None:
    write_log(update(scan(read_log(LOG, FIELDS), days)), LOG)


def report() -> None:
    log = read_log(LOG, FIELDS)
    net = pd.to_numeric(log.net, errors="coerce").dropna()
    print(f"{log.status.eq('short').sum()} open shorts, {len(net)} done: mean {net.mean():+.2%} net per event, "
          f"t {tstat(net):.2f}; graduates at >= {BAR:+.0%} over 10+ events")
    print(log[~log.status.str.startswith("skip")].tail(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="look back this many days for new prospectuses")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run(args.days)


if __name__ == "__main__":
    main()
