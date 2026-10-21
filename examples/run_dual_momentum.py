"""Dual momentum (Antonacci GEM and a wider ETF version), on real data.

    python examples/run_dual_momentum.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, DualMomentum

data = load_market(["SPY", "EFA", "AGG", "QQQ", "IWM", "EEM", "GLD", "TLT", "SHY"], start="2005-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "50/50 SPY/AGG hold": BuyAndHold(symbols=["SPY", "AGG"]),
    "GEM: SPY/EFA else AGG": DualMomentum(universe=["SPY", "EFA"], safe="AGG"),
    "Top 2 of 7 ETFs else SHY": DualMomentum(universe=["SPY", "EFA", "QQQ", "IWM", "EEM", "GLD", "TLT"], safe="SHY", top_n=2),
}
table = pd.DataFrame({name: bt.run(s, data).stats(benchmark=spy) for name, s in runs.items()})
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(table)

grid = sweep(DualMomentum, {"lookback": [63, 126, 189, 252]}, data, bt, universe=["SPY", "EFA"], safe="AGG")
print(grid[["lookback", "cagr", "sharpe", "max_drawdown", "n_trades"]])
