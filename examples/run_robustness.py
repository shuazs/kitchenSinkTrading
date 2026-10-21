"""Robustness checks: crypto breakout across fee levels and coins, and
walk-forward tests for the breakout and the leveraged trend filter.

    python examples/run_robustness.py
"""
import pandas as pd

from backtester import COSTS, Backtester, CostModel
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.metrics import drawdown
from backtester.research import walk_forward
from backtester.strategies import Breakout, BuyAndHold, TrendFilter

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
rf = tbill_rate()
KEY = ["final_equity", "cagr", "sharpe", "max_drawdown", "n_trades", "cost_drag_pct_per_year"]


def curve_stats(eq: pd.Series) -> pd.Series:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return pd.Series({"start": eq.index[0].date(), "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1, "max_drawdown": drawdown(eq).min()})


# 1. Fees. COSTS["crypto"] is a retail taker fee (0.4%); maker fees on
# low-fee exchanges run near 0.1% and below.
fees = {
    "0.40% (retail taker)": COSTS["crypto"],
    "0.25%": CostModel(pct_fee=0.0025, spread_bps=10, slippage_bps=10),
    "0.10% (low-fee taker)": CostModel(pct_fee=0.001, spread_bps=10, slippage_bps=10),
    "0.05% (maker)": CostModel(pct_fee=0.0005, spread_bps=5, slippage_bps=5),
    "no costs": CostModel(spread_bps=0, slippage_bps=0),
}
pair = load_market(["BTC-USD", "ETH-USD"], start="2017-11-09")
print("== BTC+ETH breakout 20/10 by fee level")
print(pd.DataFrame({k: Backtester(5_000, costs=c, cash_rate=rf).run(Breakout(entry=20, exit=10), pair).stats()[KEY] for k, c in fees.items()}).T)

# 2. More coins. These are today's large coins, so the list itself carries
# survivorship bias: the coins that died since 2018 are not in it.
coins = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "LINK-USD", "XLM-USD", "TRX-USD"]
basket = load_market(coins, start="2018-01-01")
low_fee = Backtester(5_000, costs=fees["0.10% (low-fee taker)"], cash_rate=rf)
print("\n== 10-coin basket from 2018, 0.10% fees")
print(pd.DataFrame({
    "equal-weight hold": low_fee.run(BuyAndHold(), basket).stats()[KEY],
    "BTC hold": low_fee.run(BuyAndHold(symbols=["BTC-USD"]), basket).stats()[KEY],
    "breakout 20/10, all 10": low_fee.run(Breakout(entry=20, exit=10), basket).stats()[KEY],
    "breakout 20/10, top 3": low_fee.run(Breakout(entry=20, exit=10, max_positions=3), basket).stats()[KEY],
}).T)

# 3. Walk-forward: pick entry/exit on the prior 3 years, trade the next one.
grid = {"entry": [10, 20, 30, 50], "exit": [5, 10, 20]}
oos, picks = walk_forward(Breakout, grid, pair, low_fee, train_years=3, test_years=1)
print("\n== Breakout walk-forward (BTC+ETH, 0.10% fees, 3y train / 1y test)")
print(picks)
fixed = low_fee.run(Breakout(entry=20, exit=10), pair, start=oos.index[0]).equity
hold = low_fee.run(BuyAndHold(symbols=["BTC-USD"]), pair, start=oos.index[0]).equity
print(pd.DataFrame({"walk-forward": curve_stats(oos), "fixed 20/10": curve_stats(fixed), "BTC hold": curve_stats(hold)}).T)

# 4. Walk-forward for the trend filter: pick the moving average on the
# prior 5 years, trade the next one.
etfs = load_market(["SPY", "SSO", "SHY"], start="2007-01-01")
bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf)
tf = dict(risk_on="SSO", signal="SPY", risk_off="SHY")
oos, picks = walk_forward(TrendFilter, {"ma": [50, 100, 150, 200, 250]}, etfs, bt, train_years=5, test_years=1, **tf)
print("\n== Trend filter walk-forward (SSO/SHY on SPY's moving average, 5y train / 1y test)")
print(picks)
fixed = bt.run(TrendFilter(ma=200, **tf), etfs, start=oos.index[0]).equity
spy = bt.run(BuyAndHold(symbols=["SPY"]), etfs, start=oos.index[0]).equity
sso = bt.run(BuyAndHold(symbols=["SSO"]), etfs, start=oos.index[0]).equity
print(pd.DataFrame({"walk-forward": curve_stats(oos), "fixed 200d": curve_stats(fixed), "SPY hold": curve_stats(spy), "SSO hold": curve_stats(sso)}).T)
