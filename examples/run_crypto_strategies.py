"""Crypto momentum basket, volatility-targeted breakout and a capped meme
coin sleeve, against the plain BTC+ETH breakout and buy and hold.

    python examples/run_crypto_strategies.py

The coin lists are today's large coins, so they carry survivorship bias:
coins that died since 2018 are missing, which flatters every basket here.
"""
import pandas as pd

from backtester import COSTS, Backtester, CostModel
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.research import sweep, walk_forward
from backtester.strategies import Breakout, BuyAndHold, CrossSectionalMomentum, Sleeves, VolTargetBreakout

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
KEY = ["final_equity", "cagr", "volatility", "sharpe", "max_drawdown", "n_trades", "cost_drag_pct_per_year"]
rf = tbill_rate()
low_fee = CostModel(pct_fee=0.001, spread_bps=10, slippage_bps=10)


def table(runs: dict, data, bt: Backtester, start=None) -> pd.DataFrame:
    return pd.DataFrame({k: bt.run(s, data, start=start).stats()[KEY] for k, s in runs.items()}).T


# 1. Cross-sectional momentum on 10 large coins, weekly.
coins = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "LINK-USD", "XLM-USD", "TRX-USD"]
basket = load_market(coins, start="2018-01-01")
for label, costs in [("0.10% fees", low_fee), ("0.40% retail fees", COSTS["crypto"])]:
    bt = Backtester(5_000, costs=costs, cash_rate=rf)
    print(f"\n== 10-coin basket from 2018, {label}")
    print(table({
        "BTC hold": BuyAndHold(symbols=["BTC-USD"]),
        "equal-weight hold": BuyAndHold(),
        "BTC+ETH breakout 20/10": Sleeves(sleeves=[(Breakout(entry=20, exit=10), ["BTC-USD", "ETH-USD"], 1.0)]),
        "momentum top 3, 21d, weekly": CrossSectionalMomentum(top_n=3, lookback=21),
        "momentum top 3, no BTC filter": CrossSectionalMomentum(top_n=3, lookback=21, trend_symbol=None),
        "momentum top 1, 21d, weekly": CrossSectionalMomentum(top_n=1, lookback=21),
    }, basket, bt))

bt = Backtester(5_000, costs=low_fee, cash_rate=rf)
grid = sweep(CrossSectionalMomentum, {"lookback": [7, 14, 21, 28, 56], "top_n": [2, 3, 5], "rebalance_days": [7, 14]}, basket, bt)
print(f"\n== Momentum parameter grid ({grid.attrs['n_variants']} variants, 0.10% fees)")
print(grid[["lookback", "top_n", "rebalance_days", "cagr", "sharpe", "max_drawdown", "cost_drag_pct_per_year"]].sort_values("sharpe", ascending=False).to_string(index=False))

oos, picks = walk_forward(CrossSectionalMomentum, {"lookback": [7, 14, 21, 28, 56], "top_n": [2, 3, 5]}, basket, bt, train_years=3, test_years=1)
print("\n== Momentum walk-forward (3y train / 1y test)")
print(picks)
years = (oos.index[-1] - oos.index[0]).days / 365.25
hold = bt.run(BuyAndHold(symbols=["BTC-USD"]), basket, start=oos.index[0]).equity
print(f"walk-forward CAGR {(oos.iloc[-1] / oos.iloc[0]) ** (1 / years) - 1:.3f}, max DD {(oos / oos.cummax() - 1).min():.3f}; "
      f"BTC hold CAGR {(hold.iloc[-1] / hold.iloc[0]) ** (1 / years) - 1:.3f}, max DD {(hold / hold.cummax() - 1).min():.3f}")

# 2. Volatility-targeted breakout on BTC+ETH.
pair = load_market(["BTC-USD", "ETH-USD"], start="2017-11-09")
print("\n== BTC+ETH breakout 20/10, plain vs volatility-targeted (0.10% fees)")
print(table({
    "plain": Breakout(entry=20, exit=10),
    "target 40% vol": VolTargetBreakout(entry=20, exit=10, target_vol=0.4),
    "target 60% vol": VolTargetBreakout(entry=20, exit=10, target_vol=0.6),
    "target 80% vol": VolTargetBreakout(entry=20, exit=10, target_vol=0.8),
}, pair, bt))
since_2021 = table({"plain": Breakout(entry=20, exit=10), "target 60% vol": VolTargetBreakout(entry=20, exit=10, target_vol=0.6)}, pair, bt, start="2021-01-01")
print("\n== Same, since 2021")
print(since_2021)

# 3. Meme coin sleeve: 90% core breakout on BTC+ETH, 10% meme coin breakout.
# PEPE trades on Yahoo as PEPE24478-USD. BONK, PEPE and WIF only exist from
# 2023, so the full test starts when all five trade.
memes = ["DOGE-USD", "SHIB-USD", "PEPE24478-USD", "BONK-USD", "WIF-USD"]
book = load_market(["BTC-USD", "ETH-USD", *memes], start="2020-08-01")
core = (Breakout(entry=20, exit=10), ["BTC-USD", "ETH-USD"])
start = "2024-01-01"
meme_costs = CostModel(pct_fee=0.001, spread_bps=20, slippage_bps=20)
meme_bt = Backtester(5_000, costs={"BTC-USD": low_fee, "ETH-USD": low_fee, **{m: meme_costs for m in memes}}, cash_rate=rf)
print(f"\n== Meme coin sleeve from {start} (0.10% fees; double the spread and slippage on meme coins)")
print(table({
    "BTC hold": BuyAndHold(symbols=["BTC-USD"]),
    "meme basket hold": BuyAndHold(symbols=memes),
    "meme breakout 20/10 alone": Sleeves(sleeves=[(Breakout(entry=20, exit=10, max_positions=3), memes, 1.0)]),
    "core breakout alone": Sleeves(sleeves=[(*core, 1.0)]),
    "90% core + 10% meme breakout": Sleeves(sleeves=[(*core, 0.9), (Breakout(entry=20, exit=10, max_positions=3), memes, 0.1)]),
    "90% core + 10% meme hold": Sleeves(sleeves=[(*core, 0.9), (BuyAndHold(), memes, 0.1)]),
}, book, meme_bt, start=start))
doge_shib = ["DOGE-USD", "SHIB-USD"]
print("\n== Longer history: DOGE+SHIB sleeve from 2021-06")
print(table({
    "core breakout alone": Sleeves(sleeves=[(*core, 1.0)]),
    "DOGE+SHIB breakout alone": Sleeves(sleeves=[(Breakout(entry=20, exit=10), doge_shib, 1.0)]),
    "DOGE+SHIB hold": BuyAndHold(symbols=doge_shib),
    "90% core + 10% DOGE+SHIB breakout": Sleeves(sleeves=[(*core, 0.9), (Breakout(entry=20, exit=10), doge_shib, 0.1)]),
}, book, meme_bt, start="2021-06-01"))
