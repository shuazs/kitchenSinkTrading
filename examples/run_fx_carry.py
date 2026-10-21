"""G10 + MXN currency carry on a $5,000 account, as micro FX futures would trade it.

    python examples/run_fx_carry.py
"""
import pandas as pd

from backtester import Backtester, CostModel
from backtester.data.fred import tbill_rate
from backtester.data.fx import CONTRACT_UNITS, load_fx_market
from backtester.strategies.carry import FXCarry

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_fx_market(start="2004-01-01")
start = "2005-01-01"
# Micro FX futures: about $1-2 per side in fees plus a tick of spread on ~$10k notional.
fx_costs = CostModel(spread_bps=2, slippage_bps=1)

runs = {
    "2 long x 2 short, 10% vol": FXCarry(),
    "1 x 1, 10% vol": FXCarry(n_long=1, n_short=1),
    "3 x 3, 10% vol": FXCarry(n_long=3, n_short=3),
    "2 long vs USD only": FXCarry(n_short=0),
    "2 x 2 with 3-month trend": FXCarry(trend=63),
    "2 x 2, 20% vol": FXCarry(target_vol=0.20, max_gross=5.0),
}
out = {}
for name, s in runs.items():
    bt = Backtester(5_000, costs=fx_costs, allow_short=True, max_leverage=s.p["max_gross"], cash_rate=tbill_rate())
    res = bt.run(s, data, start=start)
    st = res.stats(rf=tbill_rate())
    st["avg_gross"] = res.weights.abs().sum(axis=1).mean()
    out[name] = st
print(pd.DataFrame(out).loc[["start", "end", "cagr", "volatility", "sharpe", "max_drawdown", "avg_gross",
                             "annual_turnover", "cost_drag_pct_per_year"]])

# What one contract per leg means for a $5,000 account today.
spot = data.extra["spot"].iloc[-1]
notional = pd.Series({c: CONTRACT_UNITS[c] * spot[c] for c in spot.index})
print("\nNotional of one contract (USD) and leverage it implies on $5,000:")
print(pd.DataFrame({"usd": notional.round(0), "x_equity": (notional / 5_000).round(1)}))
print("\nCurrent rate gap to USD:")
print(data.extra["rate_gap"].iloc[-1].sort_values())
