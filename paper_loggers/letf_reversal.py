"""Paper-trade the single-stock LETF rebalancing reversal; logs only.

    python paper_loggers/letf_reversal.py            # after 4pm ET: tonight's signals + fill yesterday's trades
    python paper_loggers/letf_reversal.py --report   # net return vs SPY so far

Rule (examples/run_alpha_batch10.py `letf_single_stock`, batch 10): after a
day 2 sd or more down (vs 60-day vol) in a stock with 2x single-stock ETFs
(backtester/letf.py) whose LETF pressure is above 0.02, buy at the next
open and sell at that day's close; P&L is measured against SPY over the same
open-to-close, less 5 bp.

Log: ~/.cache/kitchensink/paper/letf_reversal/trades.csv (signal day,
z-score, pressure; the next day's open/close and SPY's filled in on the next
run). Graduates after 40 trades (~1 year) at +0.5% per trade vs SPY.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, read_log, today, tstat, write_log
from backtester.letf import LETF, down_z, pressure

LOG = log_dir("letf_reversal") / "trades.csv"
FIELDS = ["signal_date", "symbol", "z", "pressure", "trade_date", "open", "close", "spy_open", "spy_close", "net_vs_spy"]
Z, MIN_PRESSURE, COST, BAR = -2.0, 0.02, 0.0005, 0.005


def run() -> None:
    tick = sorted(set(LETF) | {s for v in LETF.values() for s in v} | {"SPY"})
    start = today() - pd.Timedelta(days=200)
    c, o, v = closes(tick, start), closes(tick, start, "Open"), closes(tick, start, "Volume")
    last = c.index[-1]
    log = read_log(LOG, FIELDS, ("signal_date", "trade_date"))
    seen = set(zip(log.signal_date, log.symbol))
    rows = []
    for u in LETF:
        if u not in c:
            continue
        z, p = down_z(c[u]), pressure(c, v, u)
        for d in c.index[-5:]:  # last 5 sessions, so a missed day is caught up
            if z.get(d, 0) < Z and p.get(d, 0) > MIN_PRESSURE and (d, u) not in seen:
                rows.append({"signal_date": d, "symbol": u, "z": round(z[d], 2), "pressure": round(p[d], 4)})
                when = "at tomorrow's open" if d == last else "(missed day)"
                print(f"{d:%Y-%m-%d} SIGNAL {u}: z {z[d]:.1f}, pressure {p[d]:.3f} -> paper BUY {when}, sell at that close")
    if rows:
        log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    for i in log.index[log.net_vs_spy.isna()]:
        d, u = pd.Timestamp(log.at[i, "signal_date"]), log.at[i, "symbol"]
        k = c.index.searchsorted(d, side="right")
        if k >= len(c):
            continue
        t = c.index[k]
        r = c.at[t, u] / o.at[t, u] - 1 - (c.at[t, "SPY"] / o.at[t, "SPY"] - 1) - COST
        log.loc[i, ["trade_date", "open", "close", "spy_open", "spy_close", "net_vs_spy"]] = \
            [t, round(o.at[t, u], 4), round(c.at[t, u], 4), round(o.at[t, "SPY"], 4), round(c.at[t, "SPY"], 4), round(r, 5)]
        print(f"  filled {u} on {t:%Y-%m-%d}: {r:+.2%} vs SPY net")
    if not rows:
        print(f"{last:%Y-%m-%d}: no signals")
    write_log(log, LOG)


def report() -> None:
    log = read_log(LOG, FIELDS)
    x = pd.to_numeric(log.net_vs_spy, errors="coerce").dropna()
    print(f"{len(x)} trades: mean {x.mean():+.2%} vs SPY net, t {tstat(x):.2f}, hit {(x > 0).mean():.0%}; "
          f"graduates at >= {BAR:+.1%} over 40+ trades")
    print(log.tail(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()


if __name__ == "__main__":
    main()
