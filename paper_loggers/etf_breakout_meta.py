"""Paper-trade the ETF breakout + meta-label filter; logs only, never places orders.

    python paper_loggers/etf_breakout_meta.py            # after the close: score today's new breakouts, update exits
    python paper_loggers/etf_breakout_meta.py --report   # kept vs skipped trades so far

Primary rule: 20-day high breakout / 10-day low exit on 32 ETFs (Breakout
entry=20 exit=10), entries at the next open. The filter (LightGBM, PR #16)
is refit on every finished trade before today and skips trades it ranks in
the worse half of its training predictions (`decide(..., "skip_q50")`).

Every new breakout signal is logged with the model's probability and its
call (take / skip), so the plain breakout and the filtered book can be
compared on the same live trades. Exits and net returns (8 bp round trip)
are filled in when the primary exits. Log:
~/.cache/kitchensink/paper/etf_breakout_meta/signals.csv. Needs the `ml`
extra (pip install -e ".[ml]"). Graduates after 3 months if the filter's
Sharpe gain over the plain breakout is still >= 0.08 (half the backtest).
"""
import argparse

import numpy as np
import pandas as pd

from _common import log_dir, read_log, tstat, write_log
from backtester.data.yahoo import load_market
from backtester.ml.metalabel import GBM, event_table
from backtester.strategies import Breakout

ETFS = ("SPY QQQ IWM EFA EEM TLT IEF GLD SLV DBC USO UUP FXE FXY XLK XLF XLE XLV XLI XLY XLP XLU XLB VNQ "
        "EWJ EWZ EWG FXI HYG LQD TIP DBA").split()  # same as examples/run_ml_research.py
MARKET = ["SPY", "TLT", "^VIX", "^VIX3M", "^TNX", "^IRX"]
COST = 0.0008
LOG = log_dir("etf_breakout_meta") / "signals.csv"
FIELDS = ["signal_date", "symbol", "p", "p_q50", "call", "exit_date", "ret"]
BAR_SHARPE = 0.08


def run() -> None:
    data = load_market(ETFS + MARKET, start="2004-01-01")
    curve = data.bars["^TNX"]["close"] - data.bars["^IRX"]["close"]
    trades = event_table(data, Breakout(entry=20, exit=10), ETFS, cost=COST, curve=curve)
    last = data.index[-1]
    log = read_log(LOG, FIELDS, ("signal_date", "exit_date"))

    # new signals: entries in the last 5 sessions (catches missed days) not logged yet
    seen = set(zip(log.signal_date, log.symbol))
    recent = trades[trades.entry >= data.index[-5]]
    new = recent[[k not in seen for k in zip(recent.entry, recent.symbol)]]
    if len(new):
        train = trades[(trades.end < last - pd.Timedelta(days=5)) & trades.label.notna()]
        m = GBM().fit(train, train.label)
        q50 = float(np.quantile(m.predict_proba(train), 0.5))
        p = m.predict_proba(new)
        rows = pd.DataFrame({"signal_date": new.entry.values, "symbol": new.symbol.values, "p": p.round(4), "p_q50": round(q50, 4),
                             "call": np.where(p >= q50, "take", "skip")})
        log = pd.concat([log, rows], ignore_index=True)
        for r in rows.itertuples():
            print(f"{r.signal_date:%Y-%m-%d} breakout {r.symbol:<5} p {r.p:.3f} vs median {r.p_q50:.3f} -> {r.call} "
                  f"(buy at next open if take)")
    else:
        print(f"{last:%Y-%m-%d}: no new breakouts")

    # exits: match open log rows to the primary's closed trades
    done = trades[~trades.open].set_index(["entry", "symbol"])
    for i in log.index[log.exit_date.isna()]:
        k = (log.at[i, "signal_date"], log.at[i, "symbol"])
        if k in done.index:
            t = done.loc[k]
            log.at[i, "exit_date"], log.at[i, "ret"] = t["exit"], round(float(t["ret"]), 5)
            print(f"  exit {k[1]} (signal {k[0]:%Y-%m-%d}): {float(t['ret']):+.2%} net, filter said {log.at[i, 'call']}")
    write_log(log, LOG)


def report() -> None:
    log = read_log(LOG, FIELDS, ("signal_date",))
    log["ret"] = pd.to_numeric(log.ret, errors="coerce")
    done = log.dropna(subset=["ret"])
    print(f"{len(log)} signals since {log.signal_date.min():%Y-%m-%d}, {len(done)} closed")
    for name, g in (("all (plain breakout)", done), ("kept by filter", done[done.call == "take"]),
                    ("skipped by filter", done[done.call == "skip"])):
        print(f"  {name:<22} {len(g):>4} trades, mean {g.ret.mean():+.2%}, t {tstat(g.ret):.2f}")
    print(f"Graduates after 3 months if the filtered book's Sharpe beats the plain one by >= {BAR_SHARPE} "
          "(rerun examples/run_ml_research.py --only meta on the live period for the book Sharpe).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()


if __name__ == "__main__":
    main()
