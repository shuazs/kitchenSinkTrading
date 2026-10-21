"""The "TACO" trade: buy index ETFs after a tariff-scare selloff.

    python examples/run_taco_dip.py

Compares buying a sharp down day when Trump posted about tariffs that
session against buying any other sharp down day (the control). If the TACO
story holds, the tariff dips should bounce more.
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.trump_posts import TARIFF, daily_counts, load_posts
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, EventDip, RSIMeanReversion

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

etfs = ["SPY", "QQQ"]
data = load_market(etfs, start="2015-01-01")
posts = load_posts()
idx = data.index
data.extra["tariff_day"] = daily_counts(posts, idx, TARIFF) > 0
start = "2017-01-20"
end = posts["time"].max().tz_convert(None).normalize()
spy = data.bars["SPY"]["close"]

# 1. Event table: every SPY drop of 1.5% or more, tariff-post day or not.
r = spy.pct_change()
fwd = {f"next_{h}d": spy.shift(-h) / spy - 1 for h in (1, 3, 5, 10)}
ev = pd.DataFrame({"ret": r, "tariff_day": data.extra["tariff_day"], **fwd}).loc[start:end]
drops = ev[ev["ret"] <= -0.015]
print("SPY drops of 1.5%+ by whether he posted about tariffs that session")
print(drops.groupby("tariff_day")[list(fwd)].agg(["mean", "count"]).T)
a, b = drops.loc[drops["tariff_day"], "next_5d"], drops.loc[~drops["tariff_day"], "next_5d"]
welch_t = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
print(f"5-day bounce difference {a.mean() - b.mean():+.2%}, Welch t = {welch_t:.2f} (|t| < 2: could be luck)")
print("hit rate (next 5d > 0):", drops.groupby("tariff_day")["next_5d"].apply(lambda x: (x > 0).mean()).to_dict())
print("\ntariff-day drops by year (mean next 5d, count)")
td = drops[drops["tariff_day"]]
print(td.groupby(td.index.year)["next_5d"].agg(["mean", "count"]))

# 2. As strategies on $5,000 with costs, cash earning T-bills.
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "any 1.5% dip, hold 5": EventDip(symbols=etfs),
    "tariff-day dip, hold 5 (TACO)": EventDip(symbols=etfs, event_key="tariff_day", mode="require"),
    "non-tariff dip, hold 5": EventDip(symbols=etfs, event_key="tariff_day", mode="exclude"),
    "RSI(2)<10": RSIMeanReversion(),
    "RSI(2)<10 on tariff days only": RSIMeanReversion(event_key="tariff_day", event_mode="require"),
    "RSI(2)<10 skipping tariff days": RSIMeanReversion(event_key="tariff_day", event_mode="exclude"),
}
table = pd.DataFrame({n: bt.run(s, data, start, end).stats(benchmark=spy) for n, s in runs.items()}).T
print("\n", table[["cagr", "sharpe", "max_drawdown", "exposure", "n_trades", "cost_drag_pct_per_year"]])

grid = sweep(
    EventDip, {"dip": [-0.01, -0.015, -0.02, -0.03], "hold": [3, 5, 10], "mode": [None, "require"]},
    data, bt, start=start, end=end, symbols=etfs, event_key="tariff_day",
)
print(f"\nsweep ({grid.attrs['n_variants']} variants); mode None = any dip, require = tariff days only")
print(grid.pivot_table(index=["dip", "hold"], columns="mode", values="sharpe", dropna=False))
