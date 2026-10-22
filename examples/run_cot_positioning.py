"""CFTC Commitments of Traders positioning across 20 futures-tracking ETFs.

Signals from data/cot.py (Kang-Rouwenhorst-Tang flow reversal, Basu-Miffre
hedging pressure, the practitioner commercial "COT index", and a combo),
traded weekly on the day each report became public. Periods: design
2011-2018, validation 2019 - Jun 2024, holdout Jul 2024+ (looked at once),
plus a walk-forward that re-picks signal and top_n each year.

    python examples/run_cot_positioning.py
"""
import pandas as pd

from backtester import COSTS, Backtester, CostModel
from backtester.data import cot
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import walk_forward
from backtester.strategies import CotPositioning

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

SYMS = list(cot.ETF_MAP.values())
LIQUID = ["GLD", "SLV", "USO", "UNG", "FXE", "FXY", "TLT", "IEF", "SPY", "QQQ"]
PERIODS = {"design 2011-18": ("2011-11-15", "2018-12-31"), "valid 2019-24H1": ("2019-01-01", "2024-06-30"), "holdout 2024H2+": ("2024-07-01", None)}

data = load_market(SYMS, start="2009-06-01")
panel = cot.weekly_panel(cot.load_pit())
signals = {k: cot.to_trading_days(v, panel["known"], data.index) for k, v in cot.positioning_signals(panel).items()}
signals["equal_weight"] = signals["flow1"].notna().astype(float) * 0  # every symbol ties -> hold all
data.extra.update({f"cot_{k}": v for k, v in signals.items()})


def costs(scale=1.0):
    thin = CostModel(spread_bps=20 * scale, slippage_bps=10 * scale)
    liquid = CostModel(spread_bps=2 * scale, slippage_bps=2 * scale)
    return {s: liquid if s in LIQUID else thin for s in SYMS}


cash = tbill_rate()
long_bt = Backtester(5_000, costs=costs(), cash_rate=cash)
ls_bt = Backtester(5_000, costs=costs(), cash_rate=cash, allow_short=True)


def table(runs, bt):
    rows = {}
    for name, strat in runs.items():
        for per, (a, b) in PERIODS.items():
            st = bt.run(strat, data, a, b).stats(rf=cash)
            rows[(name, per)] = st[["cagr", "volatility", "sharpe", "max_drawdown", "annual_turnover", "cost_drag_pct_per_year"]]
    return pd.DataFrame(rows).T


sig_names = ["flow1", "flow4", "hedging_pressure", "commercial_index", "combo"]
print("Long-short, top 5 vs bottom 5 (gross 100%, no borrow fees):")
print(table({s: CotPositioning(signal_key=f"cot_{s}", top_n=5, short=True) for s in sig_names}, ls_bt))

print("\nLong-only top 5 of 20 vs equal weight of all 20:")
runs = {"equal weight": CotPositioning(signal_key="cot_equal_weight", top_n=20)}
runs.update({s: CotPositioning(signal_key=f"cot_{s}", top_n=5) for s in sig_names})
print(table(runs, long_bt))

print("\nWalk-forward (4y train, 1y test), long-only, picks signal and top_n by Sharpe:")
grid = {"signal_key": [f"cot_{s}" for s in sig_names], "top_n": [3, 5, 8]}
wf_data = data.slice("2011-11-15", None)
eq, picks = walk_forward(CotPositioning, grid, wf_data, long_bt, train_years=4, test_years=1)
print(picks)
ew = long_bt.run(runs["equal weight"], data, eq.index[0], eq.index[-1]).equity
for name, curve in {"walk-forward": eq, "equal weight": ew}.items():
    r = curve.pct_change().dropna()
    yrs = len(r) / 252
    print(f"{name:13s} {eq.index[0].date()}..{eq.index[-1].date()}: CAGR {(curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1:.3f}, Sharpe {r.mean() / r.std() * 252 ** 0.5:.2f}")

print("\nCost sensitivity, combo long-short, 2011+:")
for scale in (0, 1, 2):
    bt = Backtester(5_000, costs=costs(scale), cash_rate=cash, allow_short=True)
    st = bt.run(CotPositioning(signal_key="cot_combo", top_n=5, short=True), data, "2011-11-15").stats(rf=cash)
    print(f"costs x{scale}: CAGR {st['cagr']:.3f} Sharpe {st['sharpe']:.2f} cost drag {st['cost_drag_pct_per_year']:.3f}")
