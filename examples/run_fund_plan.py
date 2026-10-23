"""The $5,000 fund: re-run each keeper on its own, then combine them into one
account with sleeve weights, quarterly / drift-band rebalancing and a
drawdown brake.

    python examples/run_fund_plan.py [out_dir]

Sleeves (each re-run here with retail costs, $5,000, T-bill cash):
  upro     UPRO vol target 20% (VolTarget)
  trend    multi-asset ETF trend (MultiAssetTrend, ETFs only; 1x gross = no margin, also 1.5x)
  pension  SPY with SSO in the last 4 days when bonds led (PensionRebalance)
  qqq      80/10/10 QQQ/intl/small allocator with macro switch (FundamentalAllocator)
  crypto   ensemble trend on BTC+ETH only, 50% vol target, 0.4% fees (EnsembleTrend);
           the top-20-coin version is shown too but makes ~1,900 tiny trades a year
Weights were picked by rough equal risk, not optimised, then scored on 2019-21
and 2022-26 separately. Results: docs/README.md ("$5,000 fund plan").
"""
import sys
from pathlib import Path

import pandas as pd

from backtester import COSTS, Backtester, drawdown, summary
from backtester.data.fred import tbill_rate, unemployment_rate
from backtester.data.yahoo import load_market
from backtester.portfolio import combine, sleeve_returns
from backtester.strategies import BuyAndHold, EnsembleTrend, FundamentalAllocator, MultiAssetTrend, PensionRebalance, Sleeves, VolTarget
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
KEY = ["start", "cagr", "volatility", "sharpe", "max_drawdown", "calmar", "cost_drag_pct_per_year"]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else None

rf = tbill_rate()
etf_bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf, min_trade_value=10)
runs = {}

d = load_market(["SPY", "UPRO", "SHY"], start="2009-01-01")
runs["upro"] = etf_bt.run(VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20), d)
runs["spy"] = etf_bt.run(BuyAndHold(symbols=["SPY"]), d)

etfs = [s for s in TREND_UNIVERSE if s != "BTC-USD"]  # crypto lives in its own sleeve
d = load_market(etfs + list(INVERSE.values()), start="2005-01-01")
runs["trend"] = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf, min_trade_value=10, max_leverage=1.5).run(
    MultiAssetTrend(universe=etfs), d, start="2007-06-01")
# No margin (an IRA): gross capped at 1x.
runs["trend 1x"] = etf_bt.run(MultiAssetTrend(universe=etfs, max_gross=1.0), d, start="2007-06-01")
d = load_market(TREND_UNIVERSE + list(INVERSE.values()), start="2005-01-01")  # the documented version, BTC included
runs["trend+btc"] = Backtester(5_000, costs={s: COSTS["etf"] for s in d.symbols} | {"BTC-USD": COSTS["crypto"]},
                               cash_rate=rf, min_trade_value=10, max_leverage=1.5).run(MultiAssetTrend(), d, start="2007-06-01")

d = load_market(["SPY", "TLT", "SSO"], start="2002-08-01")
runs["pension"] = etf_bt.run(PensionRebalance(), d)
runs["tlt"] = etf_bt.run(BuyAndHold(symbols=["TLT"]), d)

try:
    d = load_market(["QQQ", "EFA", "EEM", "IJR", "SPY", "IEF", "BIL", "HYG", "SSO"], start="2007-06-01",
                    macro=pd.DataFrame({"UNRATE": unemployment_rate()}))
    runs["qqq"] = etf_bt.run(FundamentalAllocator(), d, start="2009-01-01")
except Exception as e:  # FRED / HF mirror down: run without the unemployment leg
    print("qqq allocator without macro data:", e)
    runs["qqq"] = etf_bt.run(FundamentalAllocator(macro=False), d, start="2009-01-01")

COINS = ("BTC ETH XRP LTC BCH ADA DOGE LINK XLM TRX EOS XMR ETC DASH ZEC NEO XEM WAVES QTUM OMG BAT ZRX "
         "LSK ICX VET XTZ ATOM ALGO DOT SOL AVAX AAVE FIL NEAR BNB SHIB HBAR ICP EGLD THETA MKR SNX SAND "
         "MANA AXS KSM DCR SC DGB RVN ONT ZIL ENJ CHZ IOTA").split()
d = load_market([f"{c}-USD" for c in COINS], start="2017-11-09")
crypto_bt = Backtester(5_000, costs=COSTS["crypto"], cash_rate=rf, min_trade_value=10)
runs["crypto"] = crypto_bt.run(
    Sleeves(sleeves=[(EnsembleTrend(top_n=2, min_history=60, target_vol=0.5), ["BTC-USD", "ETH-USD"], 1.0)]), d, start="2018-11-09")
runs["crypto top20"] = crypto_bt.run(EnsembleTrend(target_vol=0.5), d, start="2018-11-09")
runs["btc"] = crypto_bt.run(BuyAndHold(symbols=["BTC-USD"]), d, start="2018-11-09")

print("== Each sleeve on its own (full history)")
print(pd.DataFrame({k: r.stats(rf=rf)[KEY] for k, r in runs.items()}).T)

rets = sleeve_returns(runs)
if out:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: r.equity for k, r in runs.items()}).to_csv(out / "sleeve_equity.csv")

common = rets.dropna().index[0]
print(f"\n== Daily return correlations since {common.date()} (weekly returns)")
weekly = (1 + rets.loc[common:].fillna(0)).resample("W").prod() - 1
print(weekly[["upro", "trend", "trend+btc", "pension", "qqq", "crypto", "crypto top20", "spy"]].corr().round(2))

PLANS = {
    "Core, no crypto": {"upro": 0.35, "trend 1x": 0.45, "pension": 0.20},
    "Balanced (plan)": {"upro": 0.30, "trend 1x": 0.35, "pension": 0.15, "crypto": 0.20},
    "Balanced, trend 1.5x (margin)": {"upro": 0.30, "trend": 0.35, "pension": 0.15, "crypto": 0.20},
    "Balanced, QQQ allocator for pension": {"upro": 0.30, "trend 1x": 0.35, "qqq": 0.15, "crypto": 0.20},
    "Balanced, top-20 crypto": {"upro": 0.30, "trend 1x": 0.35, "pension": 0.15, "crypto top20": 0.20},
    "Growth (plan)": {"upro": 0.40, "trend 1x": 0.20, "pension": 0.10, "crypto": 0.30},
    "SPY": {"spy": 1.0},
    "60/40": {"spy": 0.6, "tlt": 0.4},
}


def fund(w, **kw):
    kw = {"rebalance": "QE", "band": 0.10, "cost_bps": 5, "cash_rate": rf} | kw
    return combine(rets, w, start=kw.pop("start", common), end=kw.pop("end", None), **kw)


def table(plans, **kw):
    rows = {}
    for name, w in plans.items():
        s = summary(fund(w, **kw), rf=rf)
        rows[name] = s[KEY]
    return pd.DataFrame(rows).T


for label, kw in [("Full common period", {}), ("2019-2021", {"end": "2021-12-31"}),
                  ("2022-2026 (out of sample for the weights)", {"start": "2022-01-01"})]:
    print(f"\n== {label}")
    print(table(PLANS, **kw))

base = PLANS["Balanced (plan)"]
print("\n== Balanced: rebalancing and drawdown-brake variants")
variants = {
    "quarterly + 10pt band (plan)": {},
    "monthly": {"rebalance": "ME", "band": None},
    "yearly": {"rebalance": "YE", "band": None},
    "band only (10pt)": {"rebalance": None},
    "brake -15% -> half, resume -7.5%": {"dd_brake": -0.15},
    "brake -20% -> half, resume -10%": {"dd_brake": -0.20},
    "brake -10% -> half, resume -5%": {"dd_brake": -0.10},
}
print(pd.DataFrame({k: summary(fund(base, **v), rf=rf)[KEY] for k, v in variants.items()}).T)

res = fund(base)
eq = res.equity.resample("YE").last()
yr = eq.pct_change()
yr.iloc[0] = eq.iloc[0] / res.initial_capital - 1
spy = fund({"spy": 1.0}).equity.resample("YE").last()
syr = spy.pct_change()
syr.iloc[0] = spy.iloc[0] / 5_000 - 1
print("\n== Calendar years, Balanced and Growth vs SPY")
g = fund(PLANS["Growth (plan)"]).equity.resample("YE").last()
gyr = g.pct_change()
gyr.iloc[0] = g.iloc[0] / 5_000 - 1
print(pd.DataFrame({"balanced": yr, "growth": gyr, "SPY": syr}).set_index(yr.index.year).to_string())
print(f"\nBalanced: ${res.equity.iloc[-1]:,.0f} from $5,000; worst drawdown {drawdown(res.equity).min():.1%}")
if out:
    pd.DataFrame({"balanced": res.equity, "growth": fund(PLANS["Growth (plan)"]).equity, "spy": fund({"spy": 1.0}).equity,
                  "60/40": fund({"spy": 0.6, "tlt": 0.4}).equity}).to_csv(out / "fund_equity.csv")
    res.weights.to_csv(out / "fund_weights.csv")
