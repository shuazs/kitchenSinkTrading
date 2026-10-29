"""Paper-trade the natural-gas storage-report-day short; logs only, never places orders.

    python paper_loggers/natgas_storage_day.py            # today's MOC action + update the log (run after 4pm ET)
    python paper_loggers/natgas_storage_day.py --report   # score finished weeks

Rule (backtester/commodities.py, examples/run_commodity_futures.py): short
UNG market-on-close the trading day before each EIA weekly storage report
(Thursday 10:30 ET, or Wednesday when Thursday is a holiday) and cover
market-on-close on the report day. Futures version: short one QG e-mini
Henry Hub over the same closes. Backtest 2007-Jun 2024 +30 bp per report net
of 6 bp (t 2.8), holdout Jul 2024+ +26 bp. One row per report goes to
~/.cache/kitchensink/paper/natgas_storage/weeks.csv with UNG closes at
entry and exit. Graduates after 26 reports if the short averages +15 bp or
more net of 6 bp (half the backtest edge).
"""
import argparse

import numpy as np
import pandas as pd

from _common import closes, log_dir, read_log, today, trading_days, tstat, write_log
from backtester.commodities import storage_report_days

LOG = log_dir("natgas_storage") / "weeks.csv"
FIELDS = ["entry_date", "report_date", "ung_entry", "ung_exit", "net_bp"]
COST_BP = 6.0
BAR_BP = 15.0


def schedule(day: pd.Timestamp) -> pd.DataFrame:
    idx = trading_days(day - pd.Timedelta(days=60), day + pd.Timedelta(days=21))
    rep = storage_report_days(idx, start=str((day - pd.Timedelta(days=60)).date()))
    rows = [{"entry_date": idx[idx.get_loc(d) - 1], "report_date": d} for d in rep if idx.get_loc(d) > 0]
    return pd.DataFrame(rows)


def run() -> None:
    day = today()
    sched = schedule(day)
    log = read_log(LOG, FIELDS, ("entry_date", "report_date"))
    start = log.report_date.max() if len(log) else day - pd.Timedelta(days=1)
    log = pd.concat([log, sched[sched.report_date > start]], ignore_index=True).sort_values("report_date")
    log = log.reset_index(drop=True)
    px = closes(["UNG"], log.entry_date.min() - pd.Timedelta(days=5))["UNG"]
    for i, r in log.iterrows():
        for col, d in (("ung_entry", r.entry_date), ("ung_exit", r.report_date)):
            if pd.isna(r[col]) and d in px.index:
                log.at[i, col] = round(float(px[d]), 4)
        if pd.notna(log.at[i, "ung_entry"]) and pd.notna(log.at[i, "ung_exit"]) and pd.isna(r.net_bp):
            log.at[i, "net_bp"] = round(-np.log(float(log.at[i, "ung_exit"]) / float(log.at[i, "ung_entry"])) * 1e4 - COST_BP, 1)
    write_log(log, LOG)
    nxt = log[log.report_date >= day].head(1)
    if nxt.empty:
        print("no report day found in the next three weeks; check the EIA schedule")
        return
    c = nxt.iloc[0]
    if day == c.entry_date:
        act = "SHORT UNG market-on-close today"
    elif day == c.report_date:
        act = "COVER UNG market-on-close today (storage report day)"
    else:
        act = "flat"
    print(f"{day:%Y-%m-%d}: {act}. Next: short at close {c.entry_date:%Y-%m-%d}, cover at close {c.report_date:%Y-%m-%d}")


def report() -> None:
    done = pd.to_numeric(read_log(LOG, FIELDS).net_bp, errors="coerce").dropna()
    print(f"{len(done)} finished reports: short UNG {done.mean():+.1f} bp net, t {tstat(done):.2f}; "
          f"graduates if >= {BAR_BP:+.0f} bp over 26+ reports")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run()


if __name__ == "__main__":
    main()
