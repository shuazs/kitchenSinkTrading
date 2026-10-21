"""Crypto N-day-high breakout vs buy and hold, on real data.

    python examples/run_breakout.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import Breakout, BuyAndHold

data = load_market(["BTC-USD", "ETH-USD"], start="2017-11-09")
bt = Backtester(initial_capital=5_000, costs=COSTS["crypto"], cash_rate=tbill_rate())
btc = data.bars["BTC-USD"]["close"]

runs = {
    "BTC buy and hold": BuyAndHold(symbols=["BTC-USD"]),
    "BTC+ETH 50/50 hold": BuyAndHold(symbols=["BTC-USD", "ETH-USD"]),
    "BTC breakout 20/10": Breakout(entry=20, exit=10),
    "BTC+ETH breakout 20/10": Breakout(entry=20, exit=10),
}
data_btc = load_market(["BTC-USD"], start="2017-11-09")
table = pd.DataFrame(
    {name: bt.run(s, data_btc if name.startswith("BTC ") else data).stats(benchmark=btc) for name, s in runs.items()}
)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(table)

# Neighbouring parameters: a real edge should not depend on one exact pair.
grid = sweep(Breakout, {"entry": [10, 20, 30, 50], "exit": [5, 10, 20]}, data, bt)
print(grid[["entry", "exit", "cagr", "sharpe", "max_drawdown", "n_trades", "cost_drag_pct_per_year"]])
