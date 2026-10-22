"""Sell XSP 0DTE iron flies when Kalshi's S&P range is narrow versus VIX1D.

    KALSHI_TRADES=sel_trades.parquet KALSHI_MARKETS=sel_markets.parquet \
        python examples/run_kalshi_vol.py

Kalshi trades and markets are filtered from the Hugging Face mirror
TrevorJS/kalshi-trades (the project copy is /mnt/project-files/kalshi-leading/data/).
SPX 1-minute bars come from backtester/data/minute_mirrors.py. VIX1D hourly
bars come from Yahoo (about 730 days back, so the test runs Jan 2024 to the
end of the Kalshi data in Jan 2026); set VIX1D_FILE to reuse a saved copy.
The built panel is cached at PANEL_FILE.

Option prices are modelled with Black-Scholes at a multiple of VIX1D. The run
reports each rule at the multiple where straddles break even ("fair": timing
alone) and at 10% and 20% above it (sellers collect a variance premium).
"""
import os

import numpy as np
import pandas as pd

from backtester import kalshi_vol as kv
from backtester.data.minute_mirrors import load_minute

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)

PANEL_FILE = os.environ.get("PANEL_FILE", "kalshi_vol_panel.parquet")
VIX1D_FILE = os.environ.get("VIX1D_FILE", "vix1d_1h.parquet")


def vix1d_hourly() -> pd.Series:
    if os.path.exists(VIX1D_FILE):
        s = pd.read_parquet(VIX1D_FILE).iloc[:, 0]
    else:
        import yfinance as yf
        h = yf.download("^VIX1D", period="730d", interval="1h", progress=False, auto_adjust=False)
        s = h["Close"].squeeze().dropna()
        s.to_frame("vix1d").to_parquet(VIX1D_FILE)
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") + pd.Timedelta(hours=1)  # known at bar end
    return s


if os.path.exists(PANEL_FILE):
    P = pd.read_parquet(PANEL_FILE)
else:
    trades = pd.read_parquet(os.environ["KALSHI_TRADES"], columns=["ticker", "yes_price", "created_time"])
    markets = pd.read_parquet(os.environ["KALSHI_MARKETS"])
    vix = vix1d_hourly()
    trades = trades[pd.to_datetime(trades.created_time, utc=True) >= vix.index.min()]
    spx = load_minute("SPX", start=str(vix.index.min().date()))
    P = kv.range_panel(markets, trades, spx, vix)
    P.to_parquet(PANEL_FILE)

P = kv.add_signal(P)
live = P[P.live]
print(f"panel: {len(P)} rows, {P.index.normalize().nunique()} days, {P.index.min().date()} to {P.index.max().date()}")

# 1) does the signal sort realised variance vs VIX1D-implied variance?
live = live.assign(kept=1 - live.rv / ((live.vix1d / 100) ** 2 * live.minutes_left / kv.MIN_PER_YEAR))
print("\nShare of VIX1D-implied variance a seller keeps (1 - realised/implied), by Kalshi bucket")
print(live.groupby(["hm", np.select([live.low, live.high], ["narrow", "wide"], "middle")]).kept.agg(["mean", "count"]).unstack())

# 2) iron fly backtests
fair = {hm: kv.fair_iv_mult(g) for hm, g in live.groupby("hm")}
print("\nfair VIX1D multiple (ATM straddles break even):", {k: round(v, 3) for k, v in fair.items()})
rows = []
for hm in sorted(fair):
    for prem in (0.0, 0.10, 0.20):
        for rule in ("low", "always", "high"):
            for wing in (0.003, 0.005):
                tr = kv.iron_fly_backtest(P, fair[hm] * (1 + prem), rule=rule, times=(hm,), wing_pct=wing)
                s = kv.fly_summary(tr)
                if s["trades"]:
                    h = len(tr) // 2
                    s["first_half"] = tr.ret_on_risk.iloc[:h].mean()
                    s["second_half"] = tr.ret_on_risk.iloc[h:].mean()
                rows.append({"time": hm, "premium": prem, "rule": rule, "wing_pct": wing, **s})
res = pd.DataFrame(rows)
print("\n$5,000, 10% of equity at risk per fly, held to the close")
print(res.to_string(index=False))
res.to_csv(os.environ.get("RESULTS_FILE", "kalshi_vol_results.csv"), index=False)
