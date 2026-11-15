"""Event book: stack the repo's calendar candidates on one capital pool.

Each calendar candidate is in the market only a few days a month, so as a
separate 15-20% sleeve its capital sits idle most of the time. This test puts
them on ONE pool that holds BIL when nothing is on and gives each active
event an equal share of the pool when several overlap. Events (all known in
advance or from the prior close, close-to-close with market-on-close orders):

  auction  short TLT (or long TBF) from the close 5 days before each month's
           10y/30y auctions to the last auction close (`auction_cycle.py`)
  natgas   short UNG from the close before each EIA storage report (Thu, Wed
           in holiday weeks) to the report-day close (PR #49 candidate)
  pension  long SSO over the last 4 days of a month when bonds led stocks
           month to date (`PensionRebalance`)
  vixspike long SPY for 5 days after a close with VIX > 3 sd over its 20-day
           mean, entered one close late (PR #51 candidate used the next open)

Costs: 3 bp per side on TLT/TBF/SPY/SSO/BIL value traded, UNG 6 bp round trip.
Reported for 2008-16 / 2017-Jun 2024 / Jul 2024+ next to each event alone and
the same events as 20% sleeves on a BIL pool (how the fund plan would hold
them). Excess-over-BIL t-stats and leave-one-out runs are in the docs row.
Results ride on in-sample-chosen candidates (natgas and vixspike are
post-hoc), so this measures packaging, not new edge. Also: the book as a 20% sleeve of the Balanced fund plan, if
/mnt/project-files/fund-plan/fund_equity.csv exists.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.auction_cycle import cycles, window_mask
from backtester.data.treasury_auctions import long_bond_auction_dates
from backtester.strategies.turn_of_month import month_position

PERIODS = [("2008", "2016"), ("2017", "2024-06"), ("2024-07", "2026"), ("2008", "2026")]
FUND = "/mnt/project-files/fund-plan/fund_equity.csv"
COST = {"TLT": 3, "TBF": 3, "SPY": 3, "SSO": 3, "BIL": 3, "UNG": 3}


def storage_days(idx):
    out = []
    for _, d in pd.Series(idx, index=idx).groupby(idx.to_period("W-SUN")):
        dow = d.index.dayofweek
        pick = d.index[dow == 3] if (dow == 3).any() else d.index[dow == 2]
        if len(pick):
            out.append(pick[0])
    return pd.DatetimeIndex(out)


def event_weights(close, vix):
    """Daily weights (held over day t's close-to-close return) per event."""
    idx = close.index
    ev = {}
    # auction: inside the window short TLT via TBF where it exists, else -TLT
    inside = window_mask(idx, cycles(long_bond_auction_dates(), idx), 5)
    tbf = close["TBF"].notna()
    ev["auction"] = pd.DataFrame({"TBF": (inside & tbf).astype(float), "TLT": -(inside & ~tbf).astype(float)})
    # natgas: short UNG over each report day's return
    sd = storage_days(idx)
    ev["natgas"] = pd.DataFrame({"UNG": -pd.Series(idx.isin(sd), index=idx).astype(float)})
    # pension: SSO in the last 4 days after bonds led (same rule as the strategy)
    _, from_end = month_position(idx)
    sig = idx[from_end == 5]
    month = idx.to_period("M")
    base = close.groupby(month).last().shift().reindex(month).set_axis(idx)
    gap = ((close["SPY"] / base["SPY"]) - (close["TLT"] / base["TLT"])).loc[sig]
    lo = gap.expanding(36).quantile(1 / 3).shift()
    on = pd.Series(np.nan, index=idx)
    on.loc[sig] = (gap < lo).astype(float)
    window = pd.Series(from_end <= 4, index=idx)  # returns of the last 4 days
    on = on.groupby(month).ffill().shift().where(window).fillna(0.0)
    ev["pension"] = pd.DataFrame({"SSO": on})
    # vixspike: z20 > 3 at close t; buy at close t+1, hold 5 days
    v = vix.reindex(idx).ffill()
    z = (v - v.rolling(20).mean()) / v.rolling(20).std()
    trig = (z > 3).astype(float)
    hold = sum(trig.shift(k) for k in range(2, 7)).clip(upper=1).fillna(0.0)
    ev["vixspike"] = pd.DataFrame({"SPY": hold})
    return ev


def book(close, evs, alloc=None):
    """Pool returns: active events share capital equally (or each gets a fixed
    `alloc`); the rest sits in BIL. Costs on weight changes."""
    rets = close.pct_change(fill_method=None).fillna(0.0)
    idx = close.index
    active = pd.DataFrame({k: (e.abs().sum(axis=1) > 0) for k, e in evs.items()}).astype(float)
    share = active.div(active.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) if alloc is None else active * alloc
    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    for k, e in evs.items():
        for c in e.columns:
            w[c] += e[c].fillna(0.0) * share[k]
    gross = pd.concat([e.abs().sum(axis=1) * share[k] for k, e in evs.items()], axis=1).sum(axis=1)
    w["BIL"] += (1 - gross.clip(upper=1)).where(close["BIL"].notna(), 0.0)
    cost = sum(w[c].diff().abs().fillna(0.0) * b for c, b in COST.items()) / 1e4
    return (w * rets).sum(axis=1) - cost, gross


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = (eq.iloc[-1] ** (252 / len(r)) - 1) * 100
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return f"{cagr:5.1f}% Sh {sh:4.2f} DD {(eq / eq.cummax() - 1).min() * 100:4.0f}%"


def main():
    syms = ["TLT", "TBF", "UNG", "SPY", "SSO", "BIL", "^VIX"]
    px = yf.download(syms, start="2007-06-01", auto_adjust=True, progress=False)["Close"]
    vix = px.pop("^VIX")
    close = px["2007-06":].dropna(subset=["TLT", "UNG", "SPY", "SSO", "BIL"])
    evs = event_weights(close, vix)
    close = close["2008":]
    evs = {k: e.loc["2008":] for k, e in evs.items()}
    act = pd.DataFrame({k: (e.abs().sum(axis=1) > 0) for k, e in evs.items()})
    print("days active per year:", (act.mean() * 252).round(0).to_dict(), "| any:", round(act.any(axis=1).mean() * 252))
    print("overlap days (2+ events):", int((act.sum(axis=1) > 1).sum()), "of", len(act))
    rows = {}
    for k in evs:
        rows[f"{k} alone, full pool"] = book(close, {k: evs[k]})[0]
    for k in evs:
        rows[f"{k} as 20% sleeve"] = book(close, {k: evs[k]}, alloc=0.2)[0]
    rows["4 x 20% sleeves (fund-plan style)"] = book(close, evs, alloc=0.2)[0]
    rows["EVENT BOOK (shared pool)"], gross = book(close, evs)
    no_vix = {k: v for k, v in evs.items() if k != "vixspike"}
    rows["event book without vixspike"] = book(close, no_vix)[0]
    half = dict(evs, natgas=evs["natgas"] * 0.5)
    rows["event book, natgas at half size"] = book(close, half)[0]
    rows["BIL"] = close["BIL"].pct_change(fill_method=None).fillna(0.0)
    rows["SPY"] = close["SPY"].pct_change(fill_method=None).fillna(0.0)
    print(f"{'':36s}" + "".join(f"{lo[:7]}-{hi[:7]:>24s}" for lo, hi in PERIODS))
    for name, r in rows.items():
        print(f"{name:36s}" + "".join(f"{stats(r[lo:hi]):>32s}" for lo, hi in PERIODS))
    eb = rows["EVENT BOOK (shared pool)"]
    print("corr of event book with SPY daily:", round(eb.corr(rows["SPY"]), 2))
    per = pd.DataFrame({k: book(close, {k: e})[0] - rows["BIL"] for k, e in evs.items()})
    print("pairwise corr of event excess returns on active days:\n", per[act.any(axis=1)].corr().round(2))
    yr = (1 + eb).groupby(eb.index.year).prod() - 1
    print("event book by year %:", (yr * 100).round(1).to_dict())
    if os.path.exists(FUND):
        f = pd.read_csv(FUND, index_col=0, parse_dates=True).iloc[:, 0].pct_change().dropna()
        j = pd.concat([f, eb], axis=1, join="inner").dropna()
        mix = 0.8 * j.iloc[:, 0] + 0.2 * j.iloc[:, 1]
        print("Balanced fund:", stats(j.iloc[:, 0]), "| 80% fund + 20% event book:", stats(mix), f"({j.index[0].date()}+)")


if __name__ == "__main__":
    main()
