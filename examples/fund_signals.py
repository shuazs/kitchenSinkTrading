"""Today's dollar targets for the $5,000 fund plan (docs/README.md, "$5,000 fund plan").

    python examples/fund_signals.py [equity] [balanced|growth]

Runs each sleeve's strategy on data up to the last close and prints what each
symbol should hold at the next open: sleeve weight x strategy weight x fund
equity. Compare with what the accounts hold and trade the differences above
~$25 (sleeves rebalance quarterly or when one drifts 10 points). No orders
are sent.
"""
import sys

import pandas as pd

from backtester.data.yahoo import load_market
from backtester.strategies import EnsembleTrend, MultiAssetTrend, PensionRebalance, Sleeves, VolTarget
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE

PLANS = {
    "balanced": {"upro": 0.30, "trend": 0.35, "pension": 0.15, "crypto": 0.20},
    "growth": {"upro": 0.40, "trend": 0.20, "pension": 0.10, "crypto": 0.30},
}
equity = float(sys.argv[1]) if len(sys.argv) > 1 else 5_000.0
plan = PLANS[sys.argv[2] if len(sys.argv) > 2 else "balanced"]
etfs = [s for s in TREND_UNIVERSE if s != "BTC-USD"]

SLEEVES = {  # name -> (strategy, symbols, history start, venue)
    "upro": (VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20), ["SPY", "UPRO"], "2023-01-01", "ETF broker"),
    "trend": (MultiAssetTrend(universe=etfs, max_gross=1.0), etfs + list(INVERSE.values()), "2021-01-01", "ETF broker"),
    "pension": (PensionRebalance(), ["SPY", "TLT", "SSO"], "2002-08-01", "ETF broker"),
    "crypto": (Sleeves(sleeves=[(EnsembleTrend(top_n=2, min_history=60, target_vol=0.5), ["BTC-USD", "ETH-USD"], 1.0)]),
               ["BTC-USD", "ETH-USD"], "2024-01-01", "crypto exchange"),
}

rows = []
for name, (strat, syms, start, venue) in SLEEVES.items():
    data = load_market(syms, start=start)
    w = strat.generate_targets(data).reindex(data.index).ffill().fillna(0.0).iloc[-1]
    sleeve = plan[name] * equity
    for sym, wt in w[w.abs() > 1e-4].items():
        rows.append({"sleeve": name, "venue": venue, "symbol": sym, "weight_in_sleeve": wt,
                     "dollars": wt * sleeve, "as_of": data.index[-1].date()})
    cash = sleeve * (1 - w.clip(lower=0).sum())
    rows.append({"sleeve": name, "venue": venue, "symbol": "cash/T-bills", "weight_in_sleeve": 1 - w.clip(lower=0).sum(),
                 "dollars": cash, "as_of": data.index[-1].date()})

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"Fund equity ${equity:,.0f}, plan {plan}")
print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\nBy symbol:")
print(out.groupby("symbol")["dollars"].sum().round(2).sort_values(ascending=False).to_string())
