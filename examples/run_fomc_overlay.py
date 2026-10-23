"""Pre-FOMC drift overlay on UPRO vol target (alpha hunt #8).

Event study: SPY open(t-1) -> open(t+1) around scheduled statements vs an
average two-day return, then the overlay (+0.5 / +1.0 UPRO, cap 1.0) on
VolTarget. Kill rule: Sharpe gain under 0.05 in either 2009-18 or 2019+.
"""
import numpy as np
import pandas as pd

from backtester import Backtester, COSTS
from backtester.data.fomc import statement_dates
from backtester.data.yahoo import load_market
from backtester.strategies import PreFOMCOverlay, VolTarget


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def main():
    dates = statement_dates()
    data = load_market(["UPRO", "SPY"], start="2009-07-01")
    data.extra["fomc"] = dates
    o = data.bars["SPY"]["open"]
    two_day = o.shift(-2) / o - 1  # open t -> open t+2
    ev = two_day.shift(1).reindex(dates).dropna()  # open t-1 -> open t+1
    for lo, hi in (("2009", "2018"), ("2019", "2026")):
        e = ev[lo:hi]
        print(f"SPY {lo}-{hi}: {len(e)} statements, {e.mean()*1e4:.1f} bp vs {two_day[lo:hi].mean()*1e4:.1f} bp any 2 days (t {e.mean()/e.std()*np.sqrt(len(e)):.2f})")
    bt = Backtester(5000, costs=COSTS["etf"])
    base = VolTarget(symbol="UPRO", target_vol=0.2, lookback=20, band=0.1)
    runs = {"UPRO VT": base, "+0.5 pre-FOMC": PreFOMCOverlay(base=base), "+1.0 pre-FOMC": PreFOMCOverlay(base=base, add=1.0)}
    for name, s in runs.items():
        r = bt.run(s, data).equity.pct_change().dropna()
        print(name, " ".join(f"{lo}-{hi} Sharpe {sharpe(r[lo:hi]):.3f}" for lo, hi in (("2009", "2018"), ("2019", "2026"), ("2024-07", "2026"))))


if __name__ == "__main__":
    main()
