"""Ensemble Donchian trend on a rotating top-N crypto universe vs BTC and
the single-lookback breakouts, on real data.

    python examples/run_ensemble_trend.py

Universe: 55 coins that were large at some point since 2017, including
several that have since faded (EOS, XEM, OMG, ICX, SC...), so the liquidity
filter picks coins as they were then rather than today's winners. Coins
that were fully delisted from Yahoo are still missing (survivorship bias),
and Yahoo's crypto history mostly starts on 2017-11-09.
"""
import pandas as pd

from backtester import COSTS, Backtester, CostModel
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import walk_forward
from backtester.strategies import Breakout, BuyAndHold, EnsembleTrend, Sleeves, VolTargetBreakout

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
KEY = ["final_equity", "cagr", "volatility", "sharpe", "max_drawdown", "n_trades", "cost_drag_pct_per_year"]

COINS = ("BTC ETH XRP LTC BCH ADA DOGE LINK XLM TRX EOS XMR ETC DASH ZEC NEO XEM WAVES QTUM OMG BAT ZRX "
         "LSK ICX VET XTZ ATOM ALGO DOT SOL AVAX AAVE FIL NEAR BNB SHIB HBAR ICP EGLD THETA MKR SNX SAND "
         "MANA AXS KSM DCR SC DGB RVN ONT ZIL ENJ CHZ IOTA").split()
data = load_market([f"{c}-USD" for c in COINS], start="2017-11-09")
START = "2018-11-09"  # a year of history before the first coin can qualify
rf = tbill_rate()
low_fee = CostModel(pct_fee=0.001, spread_bps=10, slippage_bps=10)
btc = data.bars["BTC-USD"]["close"]


def table(runs: dict, bt: Backtester) -> pd.DataFrame:
    return pd.DataFrame({k: bt.run(s, data, start=START).stats(benchmark=btc)[KEY + ["beta"]]
                         for k, s in runs.items()}).T


for label, costs in [("0.10% fees", low_fee), ("0.40% retail fees", COSTS["crypto"])]:
    bt = Backtester(5_000, costs=costs, cash_rate=rf, min_trade_value=10)
    print(f"\n== From {START}, {label}")
    print(table({
        "BTC hold": BuyAndHold(symbols=["BTC-USD"]),
        "BTC+ETH breakout 20/10": Sleeves(sleeves=[(Breakout(entry=20, exit=10), ["BTC-USD", "ETH-USD"], 1.0)]),
        "BTC+ETH vol-target breakout": Sleeves(sleeves=[(VolTargetBreakout(), ["BTC-USD", "ETH-USD"], 1.0)]),
        "Ensemble, BTC only": Sleeves(sleeves=[(EnsembleTrend(top_n=1, min_history=60, target_vol=0.5), ["BTC-USD"], 1.0)]),
        "Ensemble, top 20, 25% vol": EnsembleTrend(),
        "Ensemble, top 20, 50% vol": EnsembleTrend(target_vol=0.5),
        "Ensemble, top 10, 50% vol": EnsembleTrend(top_n=10, target_vol=0.5),
        # Coin vols run 60-100%, so a 50% target leaves the book ~20% invested.
        "Ensemble, top 20, 100% vol, 2x slot cap": EnsembleTrend(target_vol=1.0, max_weight=2.0),
    }, bt))

bt = Backtester(5_000, costs=low_fee, cash_rate=rf, min_trade_value=10)
print("\n== Robustness: lookback sets, top 20, 50% vol, 0.10% fees")
print(table({
    "short (5-30)": EnsembleTrend(target_vol=0.5, lookbacks=(5, 10, 20, 30)),
    "long (60-250)": EnsembleTrend(target_vol=0.5, lookbacks=(60, 90, 150, 250)),
    "all (5-250)": EnsembleTrend(target_vol=0.5),
    "no band": EnsembleTrend(target_vol=0.5, band=0.0),
}, bt))

res = bt.run(EnsembleTrend(target_vol=0.5), data, start=START)
eq = res.equity.resample("YE").last()
yr = eq.pct_change()
yr.iloc[0] = eq.iloc[0] / 5_000 - 1
b = btc.loc[START:].resample("YE").last()
byr = b.pct_change()
byr.iloc[0] = b.iloc[0] / btc.loc[START:].iloc[0] - 1
print("\n== Calendar years, ensemble top 20 at 50% vol vs BTC")
print(pd.DataFrame({"ensemble": yr, "BTC": byr, "avg gross exposure": res.weights.abs().sum(axis=1).resample("YE").mean()})
      .set_index(yr.index.year).to_string())

oos, picks = walk_forward(EnsembleTrend, {"target_vol": [0.25, 0.5, 1.0], "top_n": [5, 10, 20]}, data, bt,
                          train_years=2, test_years=1)
years = (oos.index[-1] - oos.index[0]).days / 365.25
print("\n== Walk-forward (2y train / 1y test)")
print(picks)
print(f"walk-forward CAGR {(oos.iloc[-1] / oos.iloc[0]) ** (1 / years) - 1:.3f}, max DD {(oos / oos.cummax() - 1).min():.3f}")
