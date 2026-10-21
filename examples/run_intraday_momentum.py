"""Intraday momentum on SPY and QQQ: first-half-hour rule and noise-area
breakout.

    python examples/run_intraday_momentum.py
    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... python examples/run_intraday_momentum.py

Without Alpaca keys this uses the free 1-minute mirrors in
backtester/data/minute_mirrors.py: a SPY stand-in built from the cash S&P
500 index (Jun 2021 on) and a QQQ stand-in from NQ futures (2024 to Mar
2025) then QQQ (to Sep 2026). With Alpaca keys (free account, see
backtester/data/alpaca.py) it loads real SPY and QQQ bars from 2016.
"""
import os

import pandas as pd

from backtester import intraday
from backtester.data.alpaca import load_minute_bars
from backtester.data.minute_mirrors import rth_bars

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

have_keys = bool(os.environ.get("APCA_API_KEY_ID"))
if not have_keys:
    print("No Alpaca keys: using the free 1-minute mirrors (S&P index for SPY, NQ/QQQ for QQQ).\n")

for sym in ["SPY", "QQQ"]:
    bars = load_minute_bars(sym, start="2016-01-01") if have_keys else rth_bars(sym)
    runs = {
        "first half hour, long only": intraday.first_half_hour(bars, long_only=True),
        "first half hour, long/short": intraday.first_half_hour(bars, long_only=False),
        "noise area, long only, 1x": intraday.noise_area(bars, long_only=True),
        "noise area, long/short, 1x": intraday.noise_area(bars, long_only=False),
    }
    # The paper's sizing: 2% daily vol target with up to 4x intraday margin.
    runs["noise area, long/short, up to 4x"] = intraday.noise_area(bars, max_leverage=4.0)
    print(f"== {sym}: {bars.index[0].date()} to {bars.index[-1].date()}")
    print(pd.DataFrame({k: intraday.summary(v) for k, v in runs.items()}).T, "\n")
