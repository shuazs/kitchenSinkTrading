"""Treasury auction cycle (alpha hunt B6, the one candidate that passed).

Per-cycle TLT/EDV return (minus SHY) from the close `days` before a month's
first 10y/30y auction to the close of its last auction day, for 3-7 days,
2008-16 / 2017-Jun 2024 / Jul 2024+; a placebo with windows shifted off the
auctions; then sleeves (TLT outside windows, cash or 1x inverse TBF inside)
and a 20% sleeve inside the Balanced fund plan if its equity file exists.
Kill rule: under 15 bp/cycle or t under 2.

Today's trade: `next_windows()` lists the upcoming exit dates.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.auction_cycle import cycles, simulate, window_mask, window_returns
from backtester.data.treasury_auctions import long_bond_auction_dates

PERIODS = [("2008", "2016"), ("2017", "2024-06"), ("2024-07", "2026"), ("2008", "2026")]
FUND = "/mnt/project-files/fund-plan/fund_equity.csv"


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return f"{((1 + r).prod() ** (252 / len(r)) - 1) * 100:5.1f}% Sh {r.mean() / r.std() * np.sqrt(252):.2f} DD {(eq / eq.cummax() - 1).min() * 100:.0f}%"


def main():
    dates = long_bond_auction_dates()
    close = yf.download(["TLT", "EDV", "TBF", "SHY", "SPY"], start="2007-10-01", auto_adjust=True, progress=False)["Close"]["2008":]
    idx = close.index
    cyc = cycles(dates, idx)
    print(f"{len(dates)} auctions, {len(cyc)} cycles, {dates.min().date()} to {dates.max().date()}")
    for sym in ("TLT", "EDV"):
        for days in (3, 4, 5, 6, 7):
            w = window_returns(close[sym], cyc, days, close["SHY"])
            print(sym, days, " | ".join(f"{lo[:4]}-{hi[:7]} {w[lo:hi].mean()*1e4:6.1f}bp t {w[lo:hi].mean()/w[lo:hi].std()*np.sqrt(len(w[lo:hi])):5.2f}" for lo, hi in PERIODS))
    ex = (close["TLT"].pct_change() - close["SHY"].pct_change()).dropna()
    for shift in (-15, -10, -8, 0, 8, 10, 15):
        shifted = [(a + shift, b + shift) for a, b in cyc if 0 <= a + shift and b + shift < len(idx)]
        m = window_mask(idx, shifted, 5).reindex(ex.index)
        print(f"placebo shift {shift:+3d} days: TLT-SHY {ex[m].mean()*1e4:+.2f} bp/day in window")
    hold = close["TLT"].pct_change()
    runs = {"TLT hold": hold, "TLT ex-window (cash)": simulate(close, dates, "TLT", 5),
            "EDV ex-window (cash)": simulate(close, dates, "EDV", 5), "TLT + TBF in window": simulate(close, dates, "TLT", 5, short="TBF")}
    for name, r in runs.items():
        print(f"{name:22s}", " | ".join(f"{lo[:4]}-{hi[:7]} {stats(r[lo:hi])}" for lo, hi in PERIODS))
    if os.path.exists(FUND):
        bal = pd.read_csv(FUND, index_col=0, parse_dates=True)["balanced"].pct_change().dropna()
        bal = bal[bal.index.dayofweek < 5]
        sleeve = runs["TLT + TBF in window"].reindex(bal.index).fillna(0)
        mix = 0.8 * bal + 0.2 * sleeve
        for lo in ("2018-11", "2024-07"):
            print(f"Balanced fund {lo}+: {stats(bal[lo:])} -> with 20% auction sleeve {stats(mix[lo:])}")
    nxt = next_windows(dates)
    print("upcoming exits (sell TLT at the close):", nxt or
          f"none in the dataset (ends {dates.max().date()}); take the next 10y/30y dates from treasurydirect.gov/auctions/upcoming")


def next_windows(dates, days: int = 5, today=None) -> list[str]:
    """Exit dates (close `days` trading days before each upcoming cycle) and
    the re-entry date (close of the last auction day)."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    future = pd.bdate_range(today - pd.Timedelta(days=15), today + pd.Timedelta(days=120))
    out = []
    for a, b in cycles(dates[dates >= future[0]], future):
        if a - days >= 0 and future[b] >= today:
            out.append(f"out {future[a - days].date()} -> back {future[b].date()}")
    return out


if __name__ == "__main__":
    main()
