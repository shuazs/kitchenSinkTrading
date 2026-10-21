"""Volatility-targeted UPRO vs buy and hold and the 200-day trend filter.

    python examples/run_vol_target.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, TrendFilter, VolTarget

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_market(["SPY", "UPRO", "SHY"], start="2009-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]
start = "2009-07-01"  # UPRO launched June 2009

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "UPRO buy and hold": BuyAndHold(symbols=["UPRO"]),
    "UPRO above SPY 200d, else SHY": TrendFilter(risk_on="UPRO", signal="SPY", risk_off="SHY", ma=200),
    "UPRO vol target 20%": VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20),
    "UPRO vol target 30%": VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.30),
    "UPRO vol target 40%, rest SHY": VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.40, risk_off="SHY"),
}
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}))

grid = sweep(
    VolTarget, {"target_vol": [0.2, 0.3, 0.4], "lookback": [10, 20, 60]}, data, bt, start=start,
    symbol="UPRO", signal="SPY", leverage=3,
)
print(grid[["target_vol", "lookback", "cagr", "sharpe", "max_drawdown", "annual_turnover"]])
