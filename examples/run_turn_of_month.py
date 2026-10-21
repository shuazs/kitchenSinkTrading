"""Turn-of-month on SPY, QQQ and 3x ETFs vs buy and hold, on real data.

    python examples/run_turn_of_month.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, TrendFilter, TurnOfMonth

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_market(["SPY", "QQQ", "UPRO", "TQQQ", "SHY"], start="1994-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]

long_runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "SPY turn-of-month": TurnOfMonth(symbol="SPY"),
}
print("1994 onward")
print(pd.DataFrame({n: bt.run(s, data, start="1994-01-01").stats(benchmark=spy) for n, s in long_runs.items()}))

# The 3x ETFs start in 2009-2010.
start = "2010-03-01"
runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "UPRO buy and hold": BuyAndHold(symbols=["UPRO"]),
    "TQQQ buy and hold": BuyAndHold(symbols=["TQQQ"]),
    "UPRO above SPY 200d, else SHY": TrendFilter(risk_on="UPRO", signal="SPY", risk_off="SHY", ma=200),
    "QQQ turn-of-month": TurnOfMonth(symbol="QQQ"),
    "UPRO turn-of-month": TurnOfMonth(symbol="UPRO"),
    "TQQQ turn-of-month": TurnOfMonth(symbol="TQQQ"),
    "TQQQ turn-of-month, QQQ > 200d": TurnOfMonth(symbol="TQQQ", signal="QQQ", ma=200),
    "QQQ buy and hold": BuyAndHold(symbols=["QQQ"]),
    "TQQQ at turn-of-month, else QQQ": TurnOfMonth(symbol="TQQQ", risk_off="QQQ"),
}
print("\n2010 onward")
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}))

grid = sweep(TurnOfMonth, {"enter_days": [1, 2, 3], "exit_day": [2, 3, 4, 5]}, data, bt, start=start, symbol="TQQQ")
print(grid[["enter_days", "exit_day", "cagr", "sharpe", "max_drawdown", "exposure"]])
