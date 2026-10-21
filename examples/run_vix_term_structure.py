"""Short volatility (SVXY) only while VIX is below 3-month VIX, on real data.

    python examples/run_vix_term_structure.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, VixTermStructure

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_market(["SPY", "SVXY", "SHY", "^VIX", "^VIX3M"], start="2011-10-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]
start = "2011-10-06"  # SVXY's first full day

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "SVXY buy and hold": BuyAndHold(symbols=["SVXY"]),
    "SVXY when VIX/VIX3M < 1": VixTermStructure(),
    "SVXY when 5d VIX/VIX3M < 0.95": VixTermStructure(threshold=0.95, smooth=5),
    "SVXY when VIX/VIX3M < 0.9, else SHY": VixTermStructure(threshold=0.9, risk_off="SHY"),
}
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}))

for name, period in {"Feb 2018 (Volmageddon)": ("2018-01-25", "2018-02-28"), "Mar 2020": ("2020-02-14", "2020-04-15")}.items():
    eq = bt.run(VixTermStructure(), data, start=start).equity.loc[period[0] : period[1]]
    print(f"{name}: strategy {eq.iloc[-1] / eq.iloc[0] - 1:.1%}")

grid = sweep(VixTermStructure, {"threshold": [0.85, 0.9, 0.95, 1.0], "smooth": [1, 3, 5]}, data, bt, start=start)
print(grid[["threshold", "smooth", "cagr", "sharpe", "max_drawdown", "exposure"]])
