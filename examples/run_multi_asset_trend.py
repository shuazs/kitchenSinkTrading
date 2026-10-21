"""Multi-asset ETF trend following (a retail managed-futures fund) and the
stocks-plus-trend return stack, against DBMF/KMLM and SPY.

    python examples/run_multi_asset_trend.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, MultiAssetTrend, Sleeves
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

etfs = [s for s in TREND_UNIVERSE if s != "BTC-USD"]
syms = TREND_UNIVERSE + list(INVERSE.values()) + ["DBMF", "KMLM", "SSO", "UPRO"]
data = load_market(syms, start="2005-01-01")
costs = {s: COSTS["etf"] for s in syms} | {"BTC-USD": COSTS["crypto"]}
rf = tbill_rate()
bt = Backtester(5_000, costs=costs, cash_rate=rf, max_leverage=1.5)
spy = data.bars["SPY"]["close"]
start = "2007-06-01"  # a year after the commodity and inverse ETFs list


def table(runs, start, bt=bt):
    return pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()})


trend = MultiAssetTrend()
print("Trend variants since", start)
print(table({
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "Trend, shorts via inverse ETFs": trend,
    "Trend, long or cash": MultiAssetTrend(short="cash"),
    "Trend, ETFs only (no BTC)": MultiAssetTrend(universe=etfs),
    "Trend, 12-month signal only": MultiAssetTrend(lookbacks=(252,)),
    "Trend, 25% vol target, 2x cap": MultiAssetTrend(target_vol=0.25, max_gross=2.0),
}, start))

# Same windows as the replicator ETFs, so the comparison is like for like.
for name, since in [("DBMF", "2019-06-01"), ("KMLM", "2021-01-01")]:
    print(f"\nAgainst {name} since {since}")
    print(table({
        name: BuyAndHold(symbols=[name]),
        "Trend, inverse": trend,
        "Trend, ETFs only": MultiAssetTrend(universe=etfs),
        "SPY": BuyAndHold(symbols=["SPY"]),
    }, since))

# Crisis years: trend's job is to make money when stocks fall.
print("\nCalendar-year returns")
years = {}
for n, s in {"SPY": BuyAndHold(symbols=["SPY"]), "Trend": trend, "Trend ETFs only": MultiAssetTrend(universe=etfs)}.items():
    eq = bt.run(s, data, start=start).equity
    years[n] = eq.resample("YE").last().pct_change().fillna(eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1)
print(pd.DataFrame(years).set_index(pd.DataFrame(years).index.year))

# Return stack: 2x S&P on half the money (100% equity exposure) plus trend on the rest.
stack_bt = Backtester(5_000, costs=costs, cash_rate=rf, max_leverage=1.5)
stack_start = "2009-07-01"
print("\nStocks plus trend since", stack_start)
print(table({
    "SPY": BuyAndHold(symbols=["SPY"]),
    "SSO buy and hold": BuyAndHold(symbols=["SSO"]),
    "50% SSO + 50% trend": Sleeves(sleeves=[(BuyAndHold(symbols=["SSO"]), ["SSO"], 0.5), (trend, syms, 0.5)]),
    "50% UPRO + 50% trend": Sleeves(sleeves=[(BuyAndHold(symbols=["UPRO"]), ["UPRO"], 0.5), (trend, syms, 0.5)]),
    "33% UPRO + 67% trend": Sleeves(sleeves=[(BuyAndHold(symbols=["UPRO"]), ["UPRO"], 1 / 3), (trend, syms, 2 / 3)]),
}, stack_start, stack_bt))

grid = sweep(MultiAssetTrend, {"asset_vol": [0.10], "target_vol": [0.10, 0.15, 0.20], "vol_weeks": [13, 26, 52]}, data, bt, start=start)
print("\nSweep (9 variants)")
print(grid[["target_vol", "vol_weeks", "cagr", "sharpe", "max_drawdown", "annual_turnover"]])
