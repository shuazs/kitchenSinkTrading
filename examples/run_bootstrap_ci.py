"""Bootstrap confidence intervals on backtest statistics, offline.

Two strategies run on random-walk prices, where the true edge is zero by
construction. The point estimates still differ from zero and from each other;
the intervals show how much of that is noise.
"""
import pandas as pd

from backtester import Backtester
from backtester.bootstrap import bootstrap_ci, bootstrap_compare
from backtester.data.synthetic import synthetic_market
from backtester.strategies import BuyAndHold, TrendFilter

data = synthetic_market(["AAA", "SAFE"], n=252 * 10)
bt = Backtester(initial_capital=5_000)
hold = bt.run(BuyAndHold(symbols=["AAA"]), data).equity.pct_change().dropna()
trend = bt.run(TrendFilter(risk_on="AAA", risk_off="SAFE"), data).equity.pct_change().dropna()

pd.set_option("display.width", 200)
print("Buy and hold AAA, 90% intervals (stationary bootstrap, mean block 20 days)")
print(bootstrap_ci(hold).round(3))
print("\nTrend filter, 90% intervals")
print(bootstrap_ci(trend).round(3))
print("\nTrend filter minus buy and hold, paired on the same days")
print(bootstrap_compare(trend, hold).round(3))
print("\nBlock length check: Sharpe band of the trend filter by mean block")
print(pd.DataFrame({b: bootstrap_ci(trend, block=b).loc["sharpe"] for b in (1, 5, 20, 60)}).T.round(3))
