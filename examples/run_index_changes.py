"""S&P 500 additions and deletions: the index-rebalancing trade event desks
run, and what is left of it for a small account.

    python examples/run_index_changes.py

Change dates are effective dates (the first day the new list counts; index
funds trade at the close before). Announcement dates are not in the free
data, so "announcement to effective" returns use the open 5 trading days
before, which assumes you knew about the change then: S&P usually announces
2 to 5 trading days ahead, so this is an upper bound on what was capturable.
Deletions are tradeable without that assumption: buy at the open of the
effective date and hold.
"""
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data import sp500
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch
from backtester.research import event_study, summarize_events
from backtester.strategies import BuyAndHold, EventHold

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
START = "2005-01-01"

ch = sp500.changes(START)
symbols = prefetch(sorted(ch["symbol"].unique()) + ["SPY", "MDY", "IJR"], verbose=True)
bars = {s: df for s in symbols if len(df := load_bars(s, start="2004-01-01")) > 20}
data = MarketData(bars)
ch = ch[ch["symbol"].isin(bars)]
print("changes with Yahoo prices (tickers as of the change date, so renamed names drop out):")
print(ch["change"].value_counts())

adds, dels = ch[ch["change"] == "add"], ch[ch["change"] == "remove"]
print("\nAdditions: open 5 days before effective to the close before effective (upper bound, see top)")
print(summarize_events(event_study(data, adds, horizons=(5,), benchmark="SPY", shift=-5)))
print("\nAdditions: from the effective date's open")
print(summarize_events(event_study(data, adds, horizons=(5, 21, 63), benchmark="SPY", shift=0)))
print("\nDeletions: from the effective date's open (vs MDY, the mid-cap index they usually move to)")
d = event_study(data, dels, horizons=(5, 21, 63, 126), benchmark="MDY", shift=0)
print(summarize_events(d))
d["period"] = pd.cut(pd.to_datetime(d["date"]).dt.year, [2004, 2014, 2030], labels=["2005-14", "2015-26"])
print(d.groupby("period", observed=True)[["x21", "x63"]].agg(["mean", "count"]))

# Portfolio: buy deletions at the effective-date open. The signal has to be
# formed the trading day before, which is fine because the change was public.
idx = data.index
prev = dels.assign(date=idx[(idx.searchsorted(dels["date"]) - 1).clip(0)])
data = MarketData(bars, extra={"deletions": prev})
mdy = data.bars["MDY"]["close"]
bt = Backtester(5_000, costs=COSTS["small_cap"], min_trade_value=25)
runs = {
    "MDY buy and hold": BuyAndHold(symbols=["MDY"]),
    "Deletions, 5 slots, hold 1 month": EventHold(events_key="deletions", hold_days=21),
    "Deletions, 5 slots, hold 3 months": EventHold(events_key="deletions", hold_days=63),
    "Deletions, 5 slots, hold 6 months": EventHold(events_key="deletions", hold_days=126),
}
print("\nPortfolios since 2006")
print(pd.DataFrame({n: bt.run(s, data, start="2006-01-01").stats(benchmark=mdy) for n, s in runs.items()}))
