"""Same-calendar-month seasonality on sector and country ETFs vs SPY and
the momentum sector rotation, on real data.

    python examples/run_seasonality.py

Sector SPDRs start in December 1998 and most iShares country ETFs in 1996,
so with a 5-year minimum history the first trades are in 2001-2004.
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, SameMonthSeasonality, SectorRotation
from backtester.strategies.seasonality import COUNTRY_ETFS
from backtester.strategies.sector_rotation import SECTOR_SPDRS

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
KEY = ["cagr", "volatility", "sharpe", "max_drawdown", "beta", "annual_turnover", "cost_drag_pct_per_year"]

data = load_market(SECTOR_SPDRS + COUNTRY_ETFS + ["SPY", "SHY"], start="1996-01-01")
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate(), min_trade_value=25)
spy = data.bars["SPY"]["close"]
start = "2005-01-03"
both = SECTOR_SPDRS + COUNTRY_ETFS

runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "Sector momentum top 3 (existing)": SectorRotation(safe="SHY"),
    "Seasonality, sectors, top 3": SameMonthSeasonality(universe=SECTOR_SPDRS, safe="SHY"),
    "Seasonality, countries, top 3": SameMonthSeasonality(universe=COUNTRY_ETFS, safe="SHY"),
    "Seasonality, both, top 3": SameMonthSeasonality(universe=both, safe="SHY"),
    "Seasonality, both, top 5": SameMonthSeasonality(universe=both, top_n=5, safe="SHY"),
    "Seasonality, both, top 3, raw (not demeaned)": SameMonthSeasonality(universe=both, demean=False, safe="SHY"),
    f"EW buy and hold, {len(both)} ETFs": BuyAndHold(symbols=both),
}
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy)[KEY] for n, s in runs.items()}).T)

# Does it beat simply holding the same ETFs? Per-month spread of picks vs the
# equal-weight universe, by decade, from the strategy's own picks.
strat = SameMonthSeasonality(universe=both, top_n=3)
w = strat.generate_targets(data)
close = data.field("close").ffill()
monthly = close.loc[w.index].pct_change(fill_method=None).shift(-1)  # next month's return
picked = (w[both] > 0)
spread = (monthly[both].where(picked).mean(axis=1) - monthly[both].mean(axis=1)).loc[start:].dropna()
print("\n== Picks minus equal-weight universe, next-month return, gross")
print(spread.groupby((spread.index.year // 5) * 5).agg(["count", "mean", "std"]).assign(
    t=lambda x: x["mean"] / x["std"] * x["count"] ** 0.5).to_string())

grid = sweep(SameMonthSeasonality, {"top_n": [2, 3, 5], "years": [10, 20], "demean": [True, False]},
             data, bt, start=start, universe=both, safe="SHY")
print(f"\n== Parameter grid ({grid.attrs['n_variants']} variants)")
print(grid[["top_n", "years", "demean", "cagr", "sharpe", "max_drawdown"]].to_string(index=False))
