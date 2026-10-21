"""Buy companies after the first report of a US government equity stake.

    python examples/run_gov_stakes.py

13 events in 11 stocks, July 2025 to May 2026. That is far too few to trust
any average; read the per-event table, not just the summary row.
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.gov_stakes import stake_events
from backtester.data.yahoo import load_market
from backtester.strategies import BuyAndHold, GovStakeEvents

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

ev0 = stake_events()
syms = sorted(set(ev0["symbol"]))
data = load_market(syms + ["SPY"], start="2025-01-01")
idx = data.index
ev = stake_events(idx)
data.extra["gov_stakes"] = ev
close, open_ = data.field("close"), data.field("open")

# 1. Per event: the gap you cannot get (last close before the news to the first
# open after it; intraday news also moved before that close), then what is left.
rows = []
for e in ev.itertuples():
    i = idx.get_loc(e.filed)  # last close before the news reached the market
    c, o = close[e.symbol], open_[e.symbol]
    row = {"symbol": e.symbol, "public": e.public, "outcome": e.outcome,
           "gap_to_entry": o.iloc[i + 1] / c.iloc[i] - 1 if i + 1 < len(idx) else None}
    for h in (5, 20, 60):
        j = i + h
        row[f"open_to_{h}d"] = c.iloc[j] / o.iloc[i + 1] - 1 if j < len(idx) else None
        row[f"spy_{h}d"] = close["SPY"].iloc[j] / open_["SPY"].iloc[i + 1] - 1 if j < len(idx) else None
    rows.append(row)
table = pd.DataFrame(rows)
print(table)
print("\nmedian / mean across events")
print(table.drop(columns=["symbol", "public", "outcome"]).agg(["median", "mean"]))

# 2. As a $5,000 strategy: up to 4 equal slots, small-cap costs.
bt = Backtester(initial_capital=5_000, costs=COSTS["small_cap"], cash_rate=tbill_rate())
start = ev["filed"].min()
runs = {"SPY buy and hold": BuyAndHold(symbols=["SPY"])}
for hold in (5, 20, 60):
    for delay in (0, 1, 5):
        runs[f"hold {hold}d, enter {delay}d late"] = GovStakeEvents(hold_days=hold, delay=delay)
runs["hold 20d, deals only (hindsight)"] = GovStakeEvents(hold_days=20, outcomes=("deal",))
out = pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=close["SPY"]) for n, s in runs.items()}).T
print("\n", out[["total_return", "cagr", "sharpe", "max_drawdown", "exposure", "n_trades"]])
