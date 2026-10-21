"""SPY put credit spreads and iron condors on modeled prices, $5,000 account.

Every leg is priced off VIX with a skew fitted to real SPY trades (see
backtester/strategies/spreads.py and examples/fit_spy_skew.py). Each trade is
run at today's SPY price level, so $5 wide and $0.65 a contract mean what
they mean now.

Three questions, each a block of variants:
  1. Does the edge survive costs? (free vs default vs double costs)
  2. Does management help? (50% take profit + 21-day exit vs hold to expiry)
  3. What does more risk do? (10% vs 30% of equity per trade, 3 staggered positions)
plus how much the answer depends on the skew assumption.

Needs network access to Yahoo Finance and FRED:
    python examples/run_spreads.py
"""
import pandas as pd

from backtester.data.fred import tbill_rate
from backtester.strategies.spreads import CreditSpreads, load_spread_inputs

CAPITAL = 5_000
FREE = dict(commission=0.0, spread_pct=0.0, min_spread=0.0)
HOLD = dict(take_profit=1.0, exit_dte=None)
CONDOR = dict(structure="iron_condor")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

VARIANTS = {
    "put spread, managed": {},
    "put spread, managed, no costs": FREE,
    "put spread, managed, 2x costs": dict(spread_pct=0.01, min_spread=0.04),
    "put spread, hold": HOLD,
    "put spread, hold, no costs": {**HOLD, **FREE},
    "put spread, hold, VRP + trend filter": {**HOLD, "vrp_filter": True, "trend_filter": True},
    "put spread, hold, 16 delta": {**HOLD, "put_delta": 0.16},
    "put spread, hold, 30% risk": {**HOLD, "risk_per_trade": 0.30},
    "put spread, hold, 3 staggered": {**HOLD, "max_positions": 3, "entry_spacing": 5},
    "put spread, hold, half skew": {**HOLD, "skew_scale": 0.5},
    "put spread, hold, 1.5x skew": {**HOLD, "skew_scale": 1.5},
    "condor, managed": CONDOR,
    "condor, managed, no costs": {**CONDOR, **FREE},
    "condor, hold": {**CONDOR, **HOLD},
    "condor, hold, no costs": {**CONDOR, **HOLD, **FREE},
    "condor, hold, 2x costs": {**CONDOR, **HOLD, "spread_pct": 0.01, "min_spread": 0.04},
    "condor, hold, 30% risk": {**CONDOR, **HOLD, "risk_per_trade": 0.30},
    "condor, hold, half skew": {**CONDOR, **HOLD, "skew_scale": 0.5},
}


def table(inputs, rf, start):
    rows, curves = {}, {}
    for name, kw in VARIANTS.items():
        strat = CreditSpreads(**kw)
        res = strat.run(inputs, CAPITAL, start=start)
        st, c = res.stats(benchmark=inputs["adj_close"], rf=rf), strat.closed_
        rows[name] = {
            "CAGR": st["cagr"],
            "vol": st["volatility"],
            "Sharpe": st["sharpe"],
            "max DD": st["max_drawdown"],
            "final $": st["final_equity"],
            "trades": len(c),
            "win rate": (c["pnl"] > 0).mean(),
            "avg credit $": (c["credit"] * 100).mean(),
            "worst trade $": c["pnl"].min(),
            "costs %/yr": st["cost_drag_pct_per_year"],
        }
        curves[name] = res.equity
    spy = inputs["adj_close"].loc[start:]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    bills = (1 + rf.reindex(spy.index).ffill().fillna(0) / 252).cumprod()
    for name, s in (("SPY buy and hold", spy), ("T-bills", bills)):
        dd = (s / s.cummax() - 1).min()
        rows[name] = {"CAGR": (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1, "max DD": dd,
                      "final $": CAPITAL * s.iloc[-1] / s.iloc[0]}
        curves[name] = CAPITAL * s / s.iloc[0]
    return pd.DataFrame(rows).T, pd.DataFrame(curves)


if __name__ == "__main__":
    inputs = load_spread_inputs("SPY")
    rf = tbill_rate()
    fmt = {"CAGR": "{:.1%}", "vol": "{:.1%}", "Sharpe": "{:.2f}", "max DD": "{:.0%}", "final $": "{:,.0f}",
           "trades": "{:.0f}", "win rate": "{:.0%}", "avg credit $": "{:.0f}", "worst trade $": "{:,.0f}",
           "costs %/yr": "{:.1%}"}
    for start in ("2000-01-01", "2010-01-01"):
        t, curves = table(inputs, rf, start)
        shown = t.apply(lambda col: col.map(lambda v, f=fmt[col.name]: "" if pd.isna(v) else f.format(v)))
        print(f"\n=== SPY spreads from {start[:4]}, ${CAPITAL:,} start ===")
        print(shown.to_string())
        yearly = curves.resample("YE").last().pct_change()
        yearly.iloc[0] = curves.resample("YE").last().iloc[0] / CAPITAL - 1
        yearly.index = yearly.index.year
        keep = ["put spread, hold", "condor, hold", "put spread, managed", "SPY buy and hold"]
        print("\nWorst SPY years:")
        print(yearly.sort_values("SPY buy and hold")[keep].head(5).map("{:.1%}".format).to_string())
