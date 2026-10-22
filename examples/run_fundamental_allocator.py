"""Long-term fundamental portfolio manager vs SPY: point-in-time S&P 500
stock picks on SEC fundamentals, plus international and small-cap ETFs, a
macro regime switch and volatility cap. $5,000, costs, T-bill cash.

Settings were chosen on 2007-2018 only; 2019-2026 is the holdout. Also runs
a walk-forward of the factor mix and the survivorship-bias check (equal
weight of the members Yahoo still has, against RSP, the real equal-weight
index).

    SEC_USER_AGENT="name email" python examples/run_fundamental_allocator.py

First run downloads ~650 price histories and ~700 SEC frames (10-15 min).
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.core import Result
from backtester.data import fundamentals as F
from backtester.data.fred import tbill_rate, unemployment_rate
from backtester.research import walk_forward
from backtester.strategies import FundamentalAllocator as FA

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

data = F.market_data(macro=pd.DataFrame({"UNRATE": unemployment_rate()}))
costs = {s: COSTS["large_cap"] for s in data.bars} | {s: COSTS["etf"] for s in F.ETFS}
bt = Backtester(5_000, costs=costs, cash_rate=tbill_rate(), min_trade_value=10)
spy = data.bars["SPY"]["close"]
cols = ["cagr", "volatility", "sharpe", "max_drawdown", "annual_turnover", "cost_drag_pct_per_year", "beta"]

off = dict(macro=False, sleeve_trend=False, vol_target=None)
# The original stock picker: 60/20/20 with fundamental picks as the US sleeve.
# The class default is now 80/10/10 with QQQ as the US sleeve.
PICKS = dict(stock_etf=None, sleeves={"stocks": 0.6, "intl": 0.2, "small": 0.2})
US = dict(sleeves={"stocks": 1.0})
runs = {
    "SPY": FA(**US, stock_etf="SPY", **off),
    "RSP (equal-weight S&P 500)": FA(**US, stock_etf="RSP", **off),
    "Equal weight of members Yahoo has": FA(**US, stock_etf=None, top_n=None, **off),
    "Static 60/20/20 SPY/EFA+EEM/IJR": FA(stock_etf="SPY", sleeves=PICKS["sleeves"], **off),
    "US picks only (quality + low vol)": FA(**US, stock_etf=None, **off),
    "Stock picker, no risk overlays": FA(**PICKS, **off),
    "Stock picker": FA(**PICKS),
    "Stock picker + credit-spread signal": FA(**PICKS, macro_signals=("unemployment", "credit")),
    "Stock picker, 50% SPY core + 2x ETF when calm": FA(**PICKS, core_frac=0.5, lever_frac=0.5),
    # Big-tech tilts, added after seeing the holdout (so the holdout is not clean for these).
    "QQQ": FA(**US, stock_etf="QQQ", **off),
    "60/20/20, US sleeve = QQQ": FA(sleeves=PICKS["sleeves"]),
    "Manager (defaults: 80/10/10, US sleeve = QQQ)": FA(),
    "100% QQQ + macro switch + vol cap": FA(**US, stock_etf="QQQ"),
    "Mega-cap quality + momentum picks, cap-weighted": FA(**PICKS, factors={"quality": 1, "momentum": 1, "size": 1}, weighting="cap"),
}
periods = {
    "Design, full cycle": ("2007-06-01", "2018-12-31"),
    "Design, stock picks live": ("2010-07-01", "2018-12-31"),
    "Holdout": ("2019-01-01", None),
}
for label, (start, end) in periods.items():
    print(f"\n{label}: {start} to {end or 'today'}")
    # The equal-weight check only means something once stock data is in (2010+).
    todo = {n: s for n, s in runs.items() if start >= "2010" or not n.startswith("Equal")}
    print(pd.DataFrame({n: bt.run(s, data, start, end).stats(benchmark=spy)[cols] for n, s in todo.items()}).T)

print("\nShare of S&P 500 members with Yahoo prices (the rest are mostly acquired or bankrupt):")
print(data.extra["coverage"].resample("YE").mean().round(2).to_string())

# Walk-forward: re-pick the factor mix and portfolio size every year on the
# prior 5 years, trade it the next year.
MIXES = [{"quality": 1, "lowvol": 1}, {"quality": 1, "momentum": 1},
         {"value": 1, "quality": 1, "momentum": 1}, {"quality": 1, "momentum": 1, "lowvol": 1}]


class Mix(FA):
    params = {**FA.params, "mix": 0}

    def __init__(self, **kw):
        super().__init__(**kw)
        self.p["factors"] = MIXES[self.p["mix"]]


recent = data.slice("2009-07-01")
recent.extra = data.extra
eq, picks = walk_forward(Mix, {"mix": [0, 1, 2, 3], "top_n": [15, 25]}, recent, bt, train_years=5, test_years=1, **PICKS)
res = Result(eq, pd.DataFrame(index=eq.index), pd.DataFrame(index=eq.index), pd.DataFrame(columns=["qty", "fill_price"]),
             pd.Series(0.0, index=eq.index), eq.iloc[0], "walk-forward")
print(f"\nWalk-forward ({len(MIXES) * 2} variants per window)")
print(res.stats(benchmark=spy)[["start", "end", "cagr", "max_drawdown", "sharpe", "bench_cagr", "bench_max_drawdown"]].to_string())
print(picks.to_string())
