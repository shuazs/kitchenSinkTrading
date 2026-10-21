"""Runs every strategy on random-walk prices to show the workflow offline.

Random walks have no edge, so any profit here is luck or the cost model.
"""
import pandas as pd

from backtester import Backtester, walk_forward
from backtester.data.synthetic import synthetic_market
from backtester.strategies import Breakout, BuyAndHold, DualMomentum, RSIMeanReversion, TrendFilter

data = synthetic_market(["AAA", "BBB", "SAFE"], n=252 * 10)
bt = Backtester(initial_capital=5_000)
strategies = [
    BuyAndHold(symbols=["AAA"]),
    TrendFilter(risk_on="AAA", risk_off="SAFE"),
    Breakout(entry=20, exit=10),
    RSIMeanReversion(),
    DualMomentum(safe="SAFE"),
]
pd.set_option("display.width", 200)
print(pd.DataFrame({type(s).__name__: bt.run(s, data).stats() for s in strategies}).round(3))

equity, picks = walk_forward(Breakout, {"entry": [10, 20, 50], "exit": [5, 10, 20]}, data, bt, train_years=4)
print(picks.round(3))
