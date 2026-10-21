"""Do Kalshi's hourly S&P / Nasdaq / BTC ladders lead the underlying by minutes,
and is that enough to trade 0DTE options?

    KALSHI_TRADES=sel_trades.parquet KALSHI_MARKETS=sel_markets.parquet \
        python examples/run_kalshi_minute_leadlag.py

The two parquet files are Kalshi trades and markets filtered from the Hugging
Face mirror TrevorJS/kalshi-trades (Kalshi's own API is not reachable from
everywhere). Trades need ticker, count, yes_price, taker_side, created_time;
markets need ticker, yes_sub_title, close_time, series. Minute bars come from
backtester/data/minute_mirrors.py.

Result on 2024-10 to 2026-01: no lead. Taker flow correlates about 0.15 with
the previous minute's move and about 0 with the next 1-10 minutes; trading the
1% tails hits 49-52%, and a 0DTE ATM option needs 59-69% to break even.
"""
import os

import numpy as np
import pandas as pd

from backtester import kalshi_minute as km
from backtester.data.minute_mirrors import load_minute

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)

trades = pd.read_parquet(os.environ["KALSHI_TRADES"])
markets = pd.read_parquet(os.environ["KALSHI_MARKETS"])

CASES = {"S&P hourly (KXINXU) vs SPX": ("KXINXU", "SPX"),
         "S&P hourly (KXINXU) vs MES": ("KXINXU", "MES"),
         "Nasdaq hourly (KXNASDAQ100U) vs NQ": ("KXNASDAQ100U", "NQ"),
         "BTC hourly (KXBTCD) vs Binance": ("KXBTCD", "BTC")}

bars = {}
for name, (series, under) in CASES.items():
    if under not in bars:
        bars[under] = load_minute(under).close.astype(float)
    close = bars[under]
    mk = markets[markets.series == series]
    f = km.trade_implied(trades[trades.ticker.isin(mk.ticker)], mk, close)
    R = km.returns_around(f, close)
    print(f"\n{name}: {len(f):,} minutes with trades")
    print(km.leadlag_table(f, R).to_string())
    tails = {k: km.tail_hit_rate(f.z, R[f"fwd{k}"]) for k in (3, 5, 10)}
    print("z, 1% tails:", pd.DataFrame(tails).T.to_string())
    # 0DTE proxy, out of sample: thresholds from the first half, trades in the second half (index options only)
    if under != "BTC":
        mid = f.index[len(f) // 2]
        lo, hi = f.z[f.index < mid].quantile([0.01, 0.99])
        later = f.z[f.index >= mid]
        trades_oos = km.zero_dte_proxy(later[(later > hi) | (later < lo)], close, hold=5)
        eq = 5000 * np.cumprod(1 + 0.10 * trades_oos.ret.clip(lower=-1))
        print(f"0DTE proxy, 5-min hold, 10% of equity per trade: {len(trades_oos)} trades, "
              f"hit {(trades_oos.ret > 0).mean():.1%}, avg {trades_oos.ret.mean():+.1%}, $5,000 -> ${eq.iloc[-1]:,.0f}")

spx = bars["SPX"]["2025-01":]
for hold in (3, 5, 10, 20):
    print(f"0DTE ATM break-even hit rate, {hold}-min hold: {km.breakeven_hit_rate(spx, hold=hold):.1%}")
