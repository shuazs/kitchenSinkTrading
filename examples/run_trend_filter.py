"""Leveraged trend filter vs buy and hold, on real data.

Needs network access to Yahoo Finance and FRED:
    python examples/run_trend_filter.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.strategies import BuyAndHold, TrendFilter

data = load_market(["SPY", "SSO", "SHY"], start="2007-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "SSO buy and hold": BuyAndHold(symbols=["SSO"]),
    "SSO above SPY 200d, else SHY": TrendFilter(risk_on="SSO", signal="SPY", risk_off="SHY", ma=200),
}
table = pd.DataFrame({name: bt.run(s, data).stats(benchmark=spy) for name, s in runs.items()})
pd.set_option("display.float_format", "{:.3f}".format)
print(table)
