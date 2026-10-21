"""Sector SPDR momentum rotation vs SPY, on real data.

    python examples/run_sector_rotation.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, SectorRotation
from backtester.strategies.sector_rotation import SECTOR_SPDRS

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_market(SECTOR_SPDRS + ["SPY", "SHY"], start="1999-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]
start = "2000-01-03"  # a year of history for the 252-day lookback

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "Top 3 sectors": SectorRotation(),
    "Top 3 sectors, else SHY": SectorRotation(safe="SHY"),
    "Top 3 sectors, SPY > 200d, else SHY": SectorRotation(safe="SHY", market="SPY", ma=200),
    "Top 2 sectors, 6-month, else SHY": SectorRotation(safe="SHY", top_n=2, lookbacks=(126,)),
}
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}))

grid = sweep(SectorRotation, {"top_n": [1, 2, 3, 4], "lookbacks": [(63,), (126,), (252,), (63, 126, 252)]}, data, bt, start=start, safe="SHY")
print(grid[["top_n", "lookbacks", "cagr", "sharpe", "max_drawdown"]])
