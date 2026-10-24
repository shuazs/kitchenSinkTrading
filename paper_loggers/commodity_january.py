"""Paper-trade the January commodity winners-minus-losers window; logs only.

    python paper_loggers/commodity_january.py            # any day: shows the plan; in January logs entry/exit closes
    python paper_loggers/commodity_january.py --report   # net return per year so far

Rule (examples/run_alpha_batch6.py `commodity_january`, PR #38/#39): rank
commodities on last calendar year's futures return; from the close of
January business day 5 to the close of day 10, long the top third and short
the bottom third, equal weight. Traded with ETFs: USO UNG GLD SLV CPER CORN
WEAT SOYB CANE (the ranking uses their futures, as the backtest did).

Run it daily in January (run_daily.py does) after 4pm ET: on day 5 it logs
the entry closes, on day 10 the exit closes and the spread return net of
20 bp. Log: ~/.cache/kitchensink/paper/commodity_january/legs.csv. NEAR and
post-hoc: graduates if Jan 2027 and Jan 2028 each net +1.2% or more.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, read_log, today, trading_days, write_log

ETF = {"CL=F": "USO", "NG=F": "UNG", "GC=F": "GLD", "SI=F": "SLV", "HG=F": "CPER", "ZC=F": "CORN",
       "ZW=F": "WEAT", "ZS=F": "SOYB", "SB=F": "CANE"}
LOG = log_dir("commodity_january") / "legs.csv"
FIELDS = ["year", "future", "etf", "prior_ret", "side", "entry_date", "entry", "exit_date", "exit", "ret"]
COST = 0.002
BAR = 0.012


def prior_year_returns(year: int) -> pd.Series:
    px = closes(list(ETF), f"{year - 1}-01-01")
    r = px[px.index.dayofweek < 5].pct_change(fill_method=None)
    r = r.where(r.abs() < 0.10)  # roll jumps and bad prints, as in the backtest
    seg = r[str(year - 1)]
    return ((1 + seg).prod() - 1)[seg.count() > 200]


def window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = trading_days(f"{year}-01-01", f"{year}-01-31")
    return d[4], d[9]


def run() -> None:
    day = today()
    year = day.year if day.month == 1 else day.year + 1
    d5, d10 = window(year)
    log = read_log(LOG, FIELDS, ("entry_date", "exit_date"))
    have = log[log.year.astype(str) == str(year)]
    if day < d5:
        print(f"Next window: long top third / short bottom third of {year - 1}'s commodity winners, "
              f"close {d5:%Y-%m-%d} to close {d10:%Y-%m-%d}. Ranking is fixed on {year}-01-01.")
        return
    if have.empty:
        prev = prior_year_returns(year)
        q = prev.rank(pct=True)
        side = pd.Series("", index=prev.index).mask(q >= 2 / 3, "long").mask(q <= 1 / 3, "short")
        have = pd.DataFrame({"year": year, "future": prev.index, "etf": [ETF[f] for f in prev.index],
                             "prior_ret": prev.round(4).values, "side": side.values, "entry_date": d5, "exit_date": d10})
        have = have[have.side != ""]
        log = pd.concat([log, have], ignore_index=True)
    px = closes(sorted(set(ETF.values())), d5 - pd.Timedelta(days=7))
    for i in log.index[log.year.astype(str) == str(year)]:
        e = log.at[i, "etf"]
        for leg, d in (("entry", d5), ("exit", d10)):
            if pd.isna(log.at[i, leg]) and d in px.index:
                log.at[i, leg] = round(float(px.at[d, e]), 4)
        if pd.notna(log.at[i, "exit"]) and pd.notna(log.at[i, "entry"]):
            r = float(log.at[i, "exit"]) / float(log.at[i, "entry"]) - 1
            log.at[i, "ret"] = round(r if log.at[i, "side"] == "long" else -r, 5)
    write_log(log, LOG)
    cur = log[log.year.astype(str) == str(year)]
    print(cur[["etf", "side", "prior_ret", "entry", "exit", "ret"]].to_string(index=False))
    if day == d5:
        print(f"Today is day 5: paper BUY the longs and SHORT the shorts at the close; exit at the close {d10:%Y-%m-%d}.")


def report() -> None:
    log = read_log(LOG, FIELDS)
    log["ret"] = pd.to_numeric(log.ret, errors="coerce")
    for year, g in log.dropna(subset=["ret"]).groupby("year"):
        net = g[g.side == "long"].ret.mean() + g[g.side == "short"].ret.mean() - COST
        print(f"{year}: long-short {net:+.2%} net of {COST:.1%} (bar {BAR:+.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()


if __name__ == "__main__":
    main()
