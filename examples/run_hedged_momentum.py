"""S&P 100 momentum (top 10/20 by 12-1 month return), with and without an SH hedge.

    python examples/run_hedged_momentum.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.data.yahoo_batch import prefetch
from backtester.strategies import BuyAndHold
from backtester.strategies.stock_momentum import SP100, HedgedMomentum

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

symbols = prefetch(SP100 + ["SPY", "SH"])
data = load_market(symbols, start="2005-01-01")
universe = [s for s in SP100 if s in data.bars]
bt = Backtester(initial_capital=5_000, costs=COSTS["large_cap"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]
start = "2007-01-01"  # SH launched mid-2006; leaves a year of momentum history

runs = {
    "SPY": BuyAndHold(symbols=["SPY"]),
    # Equal weight of today's S&P 100: the survivorship-bias baseline.
    "today's S&P 100, equal weight": BuyAndHold(symbols=universe),
    "top 10": HedgedMomentum(),
    "top 20": HedgedMomentum(top_n=20),
    "top 10, fully hedged with SH": HedgedMomentum(hedge="SH"),
    "top 10, half hedged": HedgedMomentum(hedge="SH", hedge_ratio=0.5),
    "top 10, hedged when SPY < 200d": HedgedMomentum(hedge="SH", ma=200),
    "top 20, hedged when SPY < 200d": HedgedMomentum(top_n=20, hedge="SH", ma=200),
}
rows = ["cagr", "volatility", "sharpe", "max_drawdown", "beta", "annual_turnover", "cost_drag_pct_per_year"]
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy, rf=tbill_rate()) for n, s in runs.items()}).loc[rows].T)
for a, b in [("2007-01-01", "2016-12-31"), ("2017-01-01", None)]:
    print(f"\nCAGR {a[:4]}-{(b or 'now')[:4]}:")
    print(pd.Series({n: bt.run(s, data, start=a, end=b).stats()["cagr"] for n, s in runs.items()}))
