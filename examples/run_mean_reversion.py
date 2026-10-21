"""RSI(2) pullback buying on index ETFs, on real data.

    python examples/run_mean_reversion.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep, walk_forward
from backtester.strategies import BuyAndHold, RSIMeanReversion

etfs = ["SPY", "QQQ", "IWM", "DIA"]
data = load_market(etfs, start="2003-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "4 ETFs equal-weight hold": BuyAndHold(symbols=etfs),
    "RSI(2)<10, exit >70": RSIMeanReversion(),
    "RSI(2)<5, exit >70": RSIMeanReversion(entry=5),
}
table = pd.DataFrame({name: bt.run(s, data).stats(benchmark=spy) for name, s in runs.items()})
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(table)

grid = sweep(RSIMeanReversion, {"entry": [5, 10, 15, 20], "exit": [50, 70, 90]}, data, bt)
print(grid[["entry", "exit", "cagr", "sharpe", "max_drawdown", "exposure", "n_trades"]])

oos, picks = walk_forward(RSIMeanReversion, {"entry": [5, 10, 15], "exit": [50, 70]}, data, bt, train_years=5)
print(picks)
print("walk-forward out-of-sample CAGR:", (oos.iloc[-1] / oos.iloc[0]) ** (252 / len(oos)) - 1)
