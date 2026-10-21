"""Earnings announcement premium on large caps, on real data.

    SEC_USER_AGENT="name email" python examples/run_earnings_premium.py

Announcement dates are 8-K Item 2.02 filings from EDGAR (from late 2004),
and the strategy trades the date predicted from a year earlier. Universe:
60 of today's largest US stocks with long histories, which is survivorship
biased toward winners; the comparison that matters is against holding the
same names all the time (equal-weight buy and hold), not against SPY.
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.edgar import earnings_dates
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.strategies import BuyAndHold, EarningsPremium
from backtester.strategies.earnings import announcement_windows

UNIVERSE = ("AAPL MSFT AMZN GOOGL NVDA JPM JNJ V PG UNH HD MA XOM CVX KO PEP MRK ABBV COST WMT BAC "
            "CSCO ORCL INTC ADBE CRM NFLX AMD QCOM TXN IBM HON CAT DE MMM GE BA LMT RTX UPS "
            "NKE MCD SBUX LOW TGT DIS CMCSA T VZ PFE BMY AMGN GILD MDT ABT TMO LLY GS MS C").split()
START = "2006-01-01"

dates = {}
for s in UNIVERSE:
    d = earnings_dates(s)
    if len(d) > 8:
        dates[s] = d["date"]
print(f"earnings dates found for {len(dates)} of {len(UNIVERSE)} names, "
      f"median {int(np.median([len(v) for v in dates.values()]))} announcements each")

data = load_market(list(dates) + ["SPY"], start="2004-06-01", earnings_dates=dates)
spy = data.bars["SPY"]["close"]
bt = Backtester(5_000, costs=COSTS["large_cap"], cash_rate=tbill_rate(), min_trade_value=25)

# How often is the one-year-earlier date right? Distance to the actual date.
miss = []
for s, ds in dates.items():
    ds = pd.DatetimeIndex(ds)
    for d in ds:
        gap = (ds - (d + pd.Timedelta(days=364))).days
        gap = gap[np.abs(gap) < 45]
        if len(gap):
            miss.append(gap[np.argmin(np.abs(gap))])
miss = pd.Series(miss)
print(f"predicted vs actual date: within 3 days {(miss.abs() <= 3).mean():.0%}, within 7 days {(miss.abs() <= 7).mean():.0%}")

# Event study: open-to-open excess return over SPY across each window.
opens, idx = data.field("open"), data.index
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
for pre, post in [(5, 1), (10, 1), (5, 3)]:
    w = announcement_windows(dates, idx, pre, post)
    rows = []
    for e in w.itertuples():
        a, b = e.entry + 1, e.exit + 1
        if b < len(idx) and idx[a] >= pd.Timestamp(START):
            r = opens[e.symbol].iloc[b] / opens[e.symbol].iloc[a] - 1
            m = opens["SPY"].iloc[b] / opens["SPY"].iloc[a] - 1
            if np.isfinite(r):
                rows.append((idx[a].year, r - m))
    es = pd.DataFrame(rows, columns=["year", "excess"])
    by = es.assign(period=pd.cut(es["year"], [2005, 2012, 2019, 2030], labels=["2006-12", "2013-19", "2020+"]))
    print(f"\n== Window {pre} days before to {post} after: excess return vs SPY per trade, gross")
    print(by.groupby("period", observed=True)["excess"].agg(n="size", mean="mean", median="median",
                                                             hit=lambda x: (x > 0).mean()).to_string())

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    f"EW buy and hold, {len(dates)} names": BuyAndHold(symbols=list(dates)),
    "Premium, 5 slots, 5d/1d, idle SPY": EarningsPremium(idle="SPY"),
    "Premium, 5 slots, 10d/1d, idle SPY": EarningsPremium(pre_days=10, idle="SPY"),
    "Premium, 10 slots, 5d/1d, idle SPY": EarningsPremium(max_positions=10, idle="SPY"),
    "Premium, 5 slots, 5d/1d, idle cash": EarningsPremium(),
}
pd.set_option("display.float_format", "{:.3f}".format)
print("\n", pd.DataFrame({n: bt.run(s, data, start=START).stats(benchmark=spy) for n, s in runs.items()})
      .drop(["start"]).to_string())
