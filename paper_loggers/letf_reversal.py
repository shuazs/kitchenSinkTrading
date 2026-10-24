"""Paper-trade the single-stock LETF rebalancing reversal; logs only.

    python paper_loggers/letf_reversal.py            # after 4pm ET: tonight's signals + fill yesterday's trades
    python paper_loggers/letf_reversal.py --report   # net return vs SPY so far

Rule (examples/run_alpha_batch10.py `letf_single_stock`, batch 10): after a
day 2 sd or more down in a stock with 2x single-stock ETFs (every name in
backtester/letf.py `ALL_LETF`: the batch 10 list plus the 40 newer names
batch 12 checked) whose LETF pressure is above 0.02, buy at the next open and
sell at that day's close; P&L is measured against SPY over the same
open-to-close, less 5 bp. A signal fires on either vol yardstick (60-day as
in batch 10, 20-day as in batch 12); both z-scores are logged.

Log: ~/.cache/kitchensink/paper/letf_reversal/trades.csv (signal day,
both z-scores, pressure; the next day's open/close and SPY's filled in on the
next run). `--report` scores pressure > 0.02 and the top decile (>= 0.154)
separately. Graduates after 40 trades (~1 year) at +0.5% per trade vs SPY.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, read_log, today, tstat, write_log
from backtester.letf import ALL_LETF, down_z, pressure

LOG = log_dir("letf_reversal") / "trades.csv"
FIELDS = ["signal_date", "symbol", "z60", "z20", "pressure", "trade_date", "open", "close", "spy_open", "spy_close", "net_vs_spy"]
Z, MIN_PRESSURE, TOP_DECILE, COST, BAR = -2.0, 0.02, 0.154, 0.0005, 0.005


def run() -> None:
    tick = sorted(set(ALL_LETF) | {s for v in ALL_LETF.values() for s in v} | {"SPY"})
    start = today() - pd.Timedelta(days=400)
    c, o, v = closes(tick, start), closes(tick, start, "Open"), closes(tick, start, "Volume")
    last = c.index[-1]
    log = read_log(LOG, FIELDS, ("signal_date", "trade_date"))
    seen = set(zip(log.signal_date, log.symbol))
    rows = []
    for u in ALL_LETF:
        if u not in c or c[u].notna().sum() < 100:
            continue
        z60, z20, p = down_z(c[u]), down_z(c[u], 20), pressure(c, v, u, min_obs=126)
        for d in c.index[-5:]:  # last 5 sessions, so a missed day is caught up
            zs = (z60.get(d, 0), z20.get(d, 0))
            if min(zs) < Z and p.get(d, 0) > MIN_PRESSURE and (d, u) not in seen:
                rows.append({"signal_date": d, "symbol": u, "z60": round(zs[0], 2), "z20": round(zs[1], 2),
                             "pressure": round(p[d], 4)})
                when = "at tomorrow's open" if d == last else "(missed day)"
                print(f"{d:%Y-%m-%d} SIGNAL {u}: z60 {zs[0]:.1f}, z20 {zs[1]:.1f}, pressure {p[d]:.3f} -> paper BUY {when}, "
                      "sell at that close")
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
    for c in ("net_vs_spy", "pressure", "z60", "z20"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    done = log.dropna(subset=["net_vs_spy"])
    for name, g in ((f"pressure > {MIN_PRESSURE}", done), (f"top decile >= {TOP_DECILE}", done[done.pressure >= TOP_DECILE]),
                    ("60-day z < -2", done[done.z60 < Z]), ("20-day z < -2", done[done.z20 < Z])):
        x = g.net_vs_spy
        print(f"{name:<22} {len(x):>4} trades: mean {x.mean():+.2%} vs SPY net, t {tstat(x):.2f}, hit {(x > 0).mean():.0%}")
    print(f"Graduates at >= {BAR:+.1%} per trade over 40+ trades")
    print(log.tail(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()


if __name__ == "__main__":
    main()
