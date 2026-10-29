"""Paper-trade two volatility candidates; logs only.

    python paper_loggers/vol_candidates.py            # after 4pm ET: tonight's signals + fill earlier ones
    python paper_loggers/vol_candidates.py --report   # scorecards

1. VIX spike -> buy SPY (examples/run_vol_batch1.py `spikes`, `price_vs_vol`): when the VIX close is more
   than 3 standard deviations above its 20-day mean, buy SPY at the next open and sell 5 (and 7)
   trading days later at the open. Backtest 2005-26: +0.99% per event after 6 bp (t 2.9, 56 events,
   about 2.7 per year; Jul 2024+ +1.97%, 10 events). Log: vix_spike.csv.
2. SVXY Mondays: buy at Monday's open, sell at the close. Backtest 2011-26: +31 bp per Monday (t 3.0,
   710 Mondays, other weekdays ~+1 bp; Jul 2024+ +38 bp); the 2018 Volmageddon Monday was -85%, so
   size this at 5% of the account at most. Log: svxy_monday.csv.

Graduation (after 6 bp / 10 bp costs): VIX spike 10 events averaging >= +0.5%; SVXY Monday 52 Mondays
averaging >= +15 bp net. The first is also scored against SPY's own average 5-day return.
"""
import argparse

import numpy as np
import pandas as pd

from _common import closes, log_dir, read_log, today, tstat, write_log

SPIKE_LOG = log_dir("vol_candidates") / "vix_spike.csv"
MON_LOG = log_dir("vol_candidates") / "svxy_monday.csv"
SPIKE_FIELDS = ["signal_date", "vix", "z20", "entry_date", "spy_entry", "spy_exit5", "spy_exit7", "net5", "net7"]
MON_FIELDS = ["date", "svxy_open", "svxy_close", "net"]
Z, COST_SPY, COST_SVXY = 3.0, 0.0006, 0.0010
BAR_SPIKE, BAR_MON = 0.005, 0.0015


def run() -> None:
    start = today() - pd.Timedelta(days=500)
    vix = closes(["^VIX"], start)["^VIX"].dropna()
    z = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
    spy_o = closes(["SPY"], start, "Open")["SPY"].dropna()
    # 1. VIX spike
    log = read_log(SPIKE_LOG, SPIKE_FIELDS, ("signal_date", "entry_date"))
    seen = set(log.signal_date)
    new = []
    for d in z.index[-5:]:  # last 5 sessions, so a missed day is caught up
        if z[d] > Z and d not in seen:
            new.append({"signal_date": d, "vix": round(vix[d], 2), "z20": round(z[d], 2)})
            print(f"{d:%Y-%m-%d} SIGNAL VIX {vix[d]:.1f} is {z[d]:.1f} sd above its 20d mean -> paper BUY SPY at the next open, "
                  "sell 5 trading days later at the open")
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    for i in log.index:
        d = pd.Timestamp(log.at[i, "signal_date"])
        k = spy_o.index.searchsorted(d, side="right")
        if k >= len(spy_o):
            continue
        log.loc[i, ["entry_date", "spy_entry"]] = [spy_o.index[k], round(spy_o.iloc[k], 4)]
        for h in (5, 7):
            if pd.isna(log.at[i, f"net{h}"]) and k + h < len(spy_o):
                x = spy_o.iloc[k + h] / spy_o.iloc[k] - 1 - 2 * COST_SPY
                log.loc[i, [f"spy_exit{h}", f"net{h}"]] = [round(spy_o.iloc[k + h], 4), round(x, 5)]
                print(f"  filled VIX-spike {d:%Y-%m-%d} h{h}: {x:+.2%} net")
    if not new:
        print(f"{z.index[-1]:%Y-%m-%d}: VIX {vix.iloc[-1]:.1f}, z20 {z.iloc[-1]:.2f}: no spike signal")
    write_log(log, SPIKE_LOG)
    # 2. SVXY Mondays
    o, c = closes(["SVXY"], start, "Open")["SVXY"].dropna(), closes(["SVXY"], start)["SVXY"].dropna()
    mon = read_log(MON_LOG, MON_FIELDS, ("date",))
    have = set(mon.date)
    rows = [{"date": d, "svxy_open": round(o[d], 4), "svxy_close": round(c[d], 4), "net": round(c[d] / o[d] - 1 - COST_SVXY, 5)}
            for d in c.index[c.index.dayofweek == 0] if d in o.index and d >= pd.Timestamp(today()) - pd.Timedelta(days=30) and d not in have]
    if rows:
        mon = pd.concat([mon, pd.DataFrame(rows)], ignore_index=True)
        print("  SVXY Mondays filled:", ", ".join(f"{r['date']:%m-%d} {r['net']:+.2%}" for r in rows))
    write_log(mon, MON_LOG)
    if today().dayofweek == 4 or today().dayofweek == 6:
        print("  next Monday: paper BUY SVXY at the open, sell at the close (<= 5% of the account)")


def report() -> None:
    log = read_log(SPIKE_LOG, SPIKE_FIELDS)
    for h in (5, 7):
        x = pd.to_numeric(log[f"net{h}"], errors="coerce").dropna()
        print(f"VIX spike h{h}: {len(x)} events, mean {x.mean():+.2%} net, t {tstat(x):.1f} (graduates after 10 at >= {BAR_SPIKE:+.1%})")
    mon = read_log(MON_LOG, MON_FIELDS)
    x = pd.to_numeric(mon.net, errors="coerce").dropna()
    print(f"SVXY Monday: {len(x)} Mondays, mean {x.mean() * 1e4:+.0f} bp net, t {tstat(x):.1f}, worst {x.min():+.1%} "
          f"(graduates after 52 at >= {BAR_MON * 1e4:.0f} bp)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()
