"""Buy Kalshi hourly index/crypto contracts after a sharp move in the underlying.

    python examples/run_kalshi_catchup.py [KALSHI_DIR] [MINUTE_DIR]

KALSHI_DIR holds sel_markets.parquet and sel_trades.parquet (Kalshi prints
from the Hugging Face mirror TrevorJS/kalshi-trades, filtered to finance
series). MINUTE_DIR holds 1-minute bars (UTC index = bar start): SPX, NDX
(already indexed by known time), BTC, ETH. Both default to the project's
shared folders. See backtester/kalshi_catchup.py for the rule and its
fill assumptions.
"""
import sys

import numpy as np
import pandas as pd

from backtester.kalshi_catchup import FEE_RATE, VOL_MULT, catchup_backtest, known_prices, summarize

KD = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project-files/kalshi-leading/data"
MD = sys.argv[2] if len(sys.argv) > 2 else "/mnt/project-files/minute-data"
UNDERLYING = {"KXINXU": "SPX", "KXNASDAQ100U": "NDX", "KXBTCD": "BTC", "KXETHD": "ETH"}
GRID = [(1, 2.5), (1, 3.0), (2, 2.5), (2, 3.0), (3, 3.0), (5, 3.0), (2, 4.0)]
LATENCY = [1.0, 5.0, 30.0]

pd.set_option("display.width", 250)
pd.set_option("display.float_format", "{:.2f}".format)

m = pd.read_parquet(f"{KD}/sel_markets.parquet")
m = m[m.series.isin(UNDERLYING) & m.result.isin(["yes", "no"])].copy()
m["K"] = m.ticker.str.extract(r"-T(\d+(?:\.\d+)?)$")[0].astype(float)  # "above K" ladders only
m = m.dropna(subset=["K"])
t = pd.read_parquet(f"{KD}/sel_trades.parquet", columns=["ticker", "count", "yes_price", "taker_side", "created_time"])
t = t[t.ticker.isin(set(m.ticker))].merge(m[["ticker", "event_ticker", "series", "K", "result", "close_time"]], on="ticker")
for c in ["created_time", "close_time"]:
    t[c] = t[c].astype("datetime64[ns, UTC]")

rows = []
for ser, und in UNDERLYING.items():
    bars = pd.read_parquet(f"{MD}/{und}.parquet")
    px = bars["close"] if und == "NDX" else known_prices(bars)
    x = t[(t.series == ser) & t.created_time.between(px.index.min(), px.index.max())]
    print(f"{ser} vs {und}: {len(x):,} prints", flush=True)
    for k, z in GRID:
        for lat in LATENCY:
            r = catchup_backtest(px, x, k=k, z=z, latency=lat, vol_mult=VOL_MULT[ser], fee_rate=FEE_RATE[ser])
            rows.append(dict(series=ser, k=k, z=z, latency=lat, **summarize(r)))
res = pd.DataFrame(rows)
print("\n$250 risk per trade, fills capped at printed size. pnl = held to expiry; pnl_x15 = sold at bid 15 min later.")
print(res.to_string(index=False))
best = res[(res.series == "KXINXU") & (res.k == 2)]
print("\nS&P 2-minute moves:", best[["z", "latency", "fills", "pnl", "t", "pnl_x15", "t_x15"]].to_string(index=False))
