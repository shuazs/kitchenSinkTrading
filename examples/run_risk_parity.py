"""Leveraged risk parity with 3x ETFs (a retail All Weather) against 60/40,
the Reddit HFEA mix (55% UPRO / 45% TMF) and SPY.

    python examples/run_risk_parity.py
"""
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, MultiAssetTrend, RiskParity, Sleeves
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE
from backtester.strategies.risk_parity import LEVERED

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


class FixedMix(BuyAndHold):
    """Fixed weights rebalanced monthly."""

    params = {"weights": {}}

    def generate_targets(self, data):
        from backtester.strategies.indicators import month_ends

        w = pd.DataFrame(0.0, index=month_ends(data.index), columns=data.symbols)
        for s, x in self.p["weights"].items():
            w[s] = x
        return w


syms = ["SPY", "TLT", "GLD", "DBC", "UPRO", "TMF", "UGL", "SHY"]
trend_syms = TREND_UNIVERSE + list(INVERSE.values())
data = load_market(sorted(set(syms + trend_syms)), start="2005-01-01")
rf = tbill_rate()
costs = {s: COSTS["etf"] for s in data.symbols} | {"BTC-USD": COSTS["crypto"]}
bt = Backtester(5_000, costs=costs, cash_rate=rf)
margin = Backtester(5_000, costs=costs, cash_rate=rf, max_leverage=2.0)  # pays the T-bill rate on borrowing
spy = data.bars["SPY"]["close"]
start = "2010-06-01"  # a year after TMF and UPRO list
unlevered = {"SPY": ("SPY", 1), "TLT": ("TLT", 1), "GLD": ("GLD", 1), "DBC": ("DBC", 1)}

runs = {
    "SPY": (bt, BuyAndHold(symbols=["SPY"])),
    "60/40 SPY/TLT": (bt, FixedMix(weights={"SPY": 0.6, "TLT": 0.4})),
    "HFEA 55 UPRO / 45 TMF": (bt, FixedMix(weights={"UPRO": 0.55, "TMF": 0.45})),
    "Risk parity 3x ETFs, 20% vol": (bt, RiskParity()),
    "Risk parity 3x ETFs, 15% vol": (bt, RiskParity(target_vol=0.15)),
    "Risk parity 3x ETFs, 30% vol": (bt, RiskParity(target_vol=0.30)),
    "Risk parity 3x, 200d trend filter": (bt, RiskParity(trend_ma=200)),
    "Risk parity 1x ETFs on 2x margin, 12% vol": (margin, RiskParity(assets=unlevered, target_vol=0.12, max_gross=2.0)),
    "50% risk parity + 50% trend": (bt, Sleeves(sleeves=[(RiskParity(), list(LEVERED) + ["SPY", "TLT", "GLD"], 0.5), (MultiAssetTrend(max_gross=1.0), trend_syms, 0.5)])),
}
print("Since", start)
print(pd.DataFrame({n: b.run(s, data, start=start).stats(benchmark=spy) for n, (b, s) in runs.items()}))

print("\nCalendar-year returns")
yr = {}
for n in ["SPY", "60/40 SPY/TLT", "HFEA 55 UPRO / 45 TMF", "Risk parity 3x ETFs, 20% vol", "Risk parity 3x, 200d trend filter", "50% risk parity + 50% trend"]:
    b, s = runs[n]
    eq = b.run(s, data, start=start).equity
    ye = pd.concat([eq.iloc[:1], eq.resample("YE").last()])
    yr[n] = ye.pct_change().iloc[1:].set_axis(ye.index[1:].year)
print(pd.DataFrame(yr))

grid = sweep(RiskParity, {"target_vol": [0.15, 0.2, 0.25], "vol_days": [21, 63, 126]}, data, bt, start=start)
print("\nSweep (9 variants)")
print(grid[["target_vol", "vol_days", "cagr", "sharpe", "max_drawdown", "annual_turnover"]])
