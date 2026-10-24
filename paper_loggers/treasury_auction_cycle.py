"""Paper-trade the Treasury auction cycle; logs only, never places orders.

    python paper_loggers/treasury_auction_cycle.py            # today's action + update the log (run after 4pm ET)
    python paper_loggers/treasury_auction_cycle.py --report   # score finished cycles

Rule (backtester/auction_cycle.py, PR #33/#36): be out of TLT (or short TBF)
from the close 5 trading days before each month's first 10y/30y auction to
the close of the cycle's last auction day; both legs are market-on-close.

Auction dates come from TreasuryDirect's public API (blocked from the cloud,
fine at home), with the Hugging Face history as a fallback for past dates.
Each run prints what to do at today's close and writes one row per cycle to
~/.cache/kitchensink/paper/auction_cycle/cycles.csv with TLT and BIL closes
at the exit and the re-entry, filled in as they happen. The rule graduates
after 6 cycles if TLT minus BIL over the window averages -18 bp or less
(half the backtest edge).
"""
import argparse

import pandas as pd
import requests

from _common import closes, log_dir, read_log, today, trading_days, tstat, write_log
from backtester.auction_cycle import cycles
from backtester.data.treasury_auctions import LONG_TERMS, long_bond_auction_dates

TD = "https://www.treasurydirect.gov/TA_WS/securities/"
LOG = log_dir("auction_cycle") / "cycles.csv"
FIELDS = ["exit_date", "back_date", "auctions", "tlt_exit", "tlt_back", "bil_exit", "bil_back", "tlt_minus_bil_bp"]
BAR_BP = -18.0
DAYS = 5


def treasurydirect_dates() -> pd.DatetimeIndex:
    """10y/30y auction days (new issues and reopenings), past ~400 days and announced upcoming ones."""
    out = []
    for path, params in (("upcoming", {"format": "json"}), ("auctioned", {"format": "json", "days": 400})):
        try:
            rows = requests.get(TD + path, params=params, timeout=30).json()
        except (requests.RequestException, ValueError) as e:
            print(f"  TreasuryDirect {path}: {e}")
            continue
        for r in rows:
            if r.get("securityType") in ("Note", "Bond") and pd.Series([r.get("securityTerm", "")]).str.match(LONG_TERMS)[0]:
                out.append(pd.Timestamp(r["auctionDate"][:10]))
    return pd.DatetimeIndex(sorted(set(out)))


def schedule(day: pd.Timestamp) -> pd.DataFrame:
    dates = treasurydirect_dates()
    try:
        hist = long_bond_auction_dates()
        dates = dates.union(hist[hist >= day - pd.Timedelta(days=400)])
    except Exception as e:  # the HF mirror only matters if TreasuryDirect is down
        print(f"  HF auction history: {e}")
    idx = trading_days(day - pd.Timedelta(days=400), day + pd.Timedelta(days=120))
    rows = []
    for first, last in cycles(dates, idx, gap=DAYS):
        if first - DAYS < 0:
            continue
        auc = [d for d in dates if idx[first] <= d <= idx[last]]
        rows.append({"exit_date": idx[first - DAYS], "back_date": idx[last],
                     "auctions": " ".join(f"{d:%m-%d}" for d in auc)})
    return pd.DataFrame(rows)


def run() -> None:
    day = today()
    sched = schedule(day)
    if sched.empty:
        print("no auction dates found; check TreasuryDirect by hand")
        return
    log = read_log(LOG, FIELDS, ("exit_date", "back_date"))
    if log.empty:  # start the log at the current or next cycle, not a year back
        sched = sched[sched.back_date >= day]
    open_ = log.tlt_back.isna()
    for _, s in sched[sched.exit_date.isin(log.exit_date[open_])].iterrows():  # a later announcement can extend a cycle
        log.loc[open_ & log.exit_date.eq(s.exit_date), ["back_date", "auctions"]] = [s.back_date, s.auctions]
    new = sched[~sched.exit_date.isin(log.exit_date) & (sched.exit_date > log.exit_date.max() if len(log) else True)]
    log = pd.concat([log, new], ignore_index=True).sort_values("exit_date").reset_index(drop=True)
    if log.empty:
        print("no upcoming cycle announced yet")
        return
    px = closes(["TLT", "BIL"], log.exit_date.min() - pd.Timedelta(days=5))
    for i, r in log.iterrows():
        for leg, d in (("exit", r.exit_date), ("back", r.back_date)):
            if pd.isna(r[f"tlt_{leg}"]) and d in px.index:
                log.at[i, f"tlt_{leg}"], log.at[i, f"bil_{leg}"] = round(px.at[d, "TLT"], 4), round(px.at[d, "BIL"], 4)
        if pd.notna(log.at[i, "tlt_back"]) and pd.notna(log.at[i, "tlt_exit"]):
            t, b = (float(log.at[i, f"{s}_back"]) / float(log.at[i, f"{s}_exit"]) - 1 for s in ("tlt", "bil"))
            log.at[i, "tlt_minus_bil_bp"] = round((t - b) * 1e4, 1)
    write_log(log, LOG)

    nxt = log[log.back_date >= day].head(1)
    if nxt.empty:
        print("no upcoming cycle announced yet")
        return
    c = nxt.iloc[0]
    if day == c.exit_date:
        act = "SELL TLT (or buy TBF) market-on-close today"
    elif day == c.back_date:
        act = "BUY TLT back (or sell TBF) market-on-close today"
    elif c.exit_date < day < c.back_date:
        act = "stay out of TLT"
    else:
        act = "hold TLT"
    print(f"{day:%Y-%m-%d}: {act}. Next cycle: out at close {c.exit_date:%Y-%m-%d}, "
          f"back at close {c.back_date:%Y-%m-%d} (auctions {c.auctions})")


def report() -> None:
    log = read_log(LOG, FIELDS)
    done = pd.to_numeric(log.tlt_minus_bil_bp, errors="coerce").dropna()
    print(f"{len(done)} finished cycles: TLT minus BIL {done.mean():+.1f} bp per window, t {tstat(done):.2f}; "
          f"graduates if <= {BAR_BP:+.0f} bp over 6+ cycles")
    print(log.tail(12).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run()


if __name__ == "__main__":
    main()
