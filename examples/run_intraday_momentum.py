"""Intraday momentum on SPY and QQQ: first-half-hour rule and noise-area
breakout. Scaffolding until minute history is available.

    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python examples/run_intraday_momentum.py

With Alpaca keys (free account, see backtester/data/alpaca.py) this loads
1-minute bars from 2016. Without them it falls back to Yahoo's last ~60
days of 5-minute bars, which only checks that the pipeline runs: 60 days
says nothing about whether the strategy works.
"""
import os

import pandas as pd

from backtester import intraday
from backtester.data.alpaca import load_minute_bars, load_yahoo_intraday

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

have_keys = bool(os.environ.get("APCA_API_KEY_ID"))
if not have_keys:
    print("No Alpaca keys: using Yahoo's last 60 days of 5-minute bars (pipeline check only).\n")

for sym in ["SPY", "QQQ"]:
    bars = load_minute_bars(sym, start="2016-01-01") if have_keys else load_yahoo_intraday(sym)
    runs = {
        "first half hour, long only": intraday.first_half_hour(bars, long_only=True),
        "first half hour, long/short": intraday.first_half_hour(bars, long_only=False),
        "noise area, long only, 1x": intraday.noise_area(bars, long_only=True),
        "noise area, long/short, 1x": intraday.noise_area(bars, long_only=False),
    }
    if have_keys:
        # The paper's sizing: 2% daily vol target with up to 4x (needs a margin account over $25k).
        runs["noise area, long/short, up to 4x"] = intraday.noise_area(bars, max_leverage=4.0)
    print(f"== {sym}: {bars.index[0].date()} to {bars.index[-1].date()}")
    print(pd.DataFrame({k: intraday.summary(v) for k, v in runs.items()}).T, "\n")
