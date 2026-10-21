"""Following activist hedge funds into their targets on the day a new
Schedule 13D is filed.

Needs SEC EDGAR and Yahoo access and SEC_USER_AGENT set:
    SEC_USER_AGENT="name email" python examples/run_activists.py

First run downloads the EDGAR form index for each quarter since 2010
(~5 MB each) and daily bars for every target.
"""
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data import activists
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch
from backtester.research import event_study, summarize_events
from backtester.strategies import BuyAndHold, EventHold

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
START = "2010-01-01"

ev = activists.events(2010)
act = ev[ev["activist"].notna()]
other = ev[ev["activist"].isna()]
print(f"original 13Ds on companies with a current ticker: {len(ev)}; by listed activists: {len(act)}")

symbols = prefetch(sorted(set(act["symbol"]) | set(other["symbol"])) + ["IWM", "SPY"], verbose=True)
bars = {s: df for s in symbols if len(df := load_bars(s, start=START)) > 20}
data = MarketData(bars)
print(f"activist events with Yahoo prices: {act['symbol'].isin(bars).sum()} of {len(act)}")

print("\nEvent study, activists (buy next open after filing; x = minus IWM)")
study = event_study(data, act, benchmark="IWM")
print(summarize_events(study))
print("\nEvent study, activists, by period")
study["period"] = pd.cut(pd.to_datetime(study["date"]).dt.year, [2009, 2017, 2030], labels=["2010-17", "2018-26"])
print(study.groupby("period", observed=True)[["x21", "x63", "x126"]].agg(["mean", "count"]))

# Other 13D filers: mostly insiders, strategic buyers and small funds.
liquid_other = other[other["symbol"].isin(bars)]
print(f"\nEvent study, all other original 13Ds ({len(liquid_other)} events)")
print(summarize_events(event_study(data, liquid_other, benchmark="IWM")))

data = MarketData(bars, extra={"activist": act[act["symbol"].isin(bars)]})
iwm = data.bars["IWM"]["close"]
bt = Backtester(5_000, costs=COSTS["small_cap"], min_trade_value=25)
runs = {
    "IWM buy and hold": BuyAndHold(symbols=["IWM"]),
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "Activists, 5 slots, hold 1 month": EventHold(events_key="activist", hold_days=21),
    "Activists, 5 slots, hold 3 months": EventHold(events_key="activist", hold_days=63),
    "Activists, 5 slots, hold 6 months": EventHold(events_key="activist", hold_days=126),
    "Activists, 3 slots, hold 6 months": EventHold(events_key="activist", hold_days=126, max_positions=3),
}
print("\nPortfolios since", START)
print(pd.DataFrame({n: bt.run(s, data, start=START).stats(benchmark=iwm) for n, s in runs.items()}))
