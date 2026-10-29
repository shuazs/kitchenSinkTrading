"""On-chain batch 3: crypto equities vs BTC's overnight move (non-trading hours).

BTC trades while US equities are shut. Signal on trading day d: BTC return
from the previous 16:00 ET to 09:30 ET (Binance 1-min), minus beta x the
equity's own overnight gap. If the equity under-reacts at the open, buy it at
the open and sell at the close (long-only; cash account). Tickers: COIN MSTR
MARA RIOT CLSK HUT (+ IBIT, FBTC from 2024). Design 2021-06..2023-12, holdout 2024+.
Cost 10 bp per side (liquid stocks, wide for small caps), 0.2% for miners.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

BTC = "/mnt/project-files/minute-data/BTC.parquet"
TICKERS = ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT"]
COST = {"COIN": 0.001, "MSTR": 0.001, "MARA": 0.0015, "RIOT": 0.0015, "CLSK": 0.002, "HUT": 0.002}
SPLIT = "2024-01-01"


def btc_overnight() -> pd.DataFrame:
    b = pd.read_parquet(BTC, columns=["close"])
    b.index = pd.to_datetime(b.index, utc=True).tz_convert("America/New_York")
    px = b["close"]
    days = pd.bdate_range(px.index[0].tz_localize(None).normalize(), px.index[-1].tz_localize(None).normalize()).tz_localize("America/New_York")
    out = []
    for prev, d in zip(days[:-1], days[1:]):
        t0 = prev + pd.Timedelta(hours=16)
        t1 = d + pd.Timedelta(hours=9, minutes=30)
        a = px.asof(t0)
        c = px.asof(t1)
        if np.isnan(a) or np.isnan(c):
            continue
        out.append((d.tz_localize(None), c / a - 1))
    return pd.DataFrame(out, columns=["date", "btc_on"]).set_index("date")["btc_on"]


def main():
    on = btc_overnight()
    print("overnight BTC days", len(on))
    px = yf.download(TICKERS, start="2021-05-01", auto_adjust=False, progress=False, threads=False)
    rows = []
    for t in TICKERS:
        o, c = px["Open"][t], px["Close"][t]
        gap = o / c.shift(1) - 1
        oc = c / o - 1
        df = pd.concat([on, gap, oc], axis=1, keys=["btc", "gap", "oc"]).dropna()
        beta = df.btc.rolling(120).cov(df.gap) / df.btc.rolling(120).var()
        resid = df.btc - df.gap / beta.shift(1).clip(0.2, 3)  # BTC move not yet in the stock (BTC-equivalent units)
        resid = resid.where(beta.shift(1).notna())
        df["resid"] = resid
        df["beta"] = beta.shift(1)
        for thr in (0.01, 0.02):
            long_ = df.resid > thr
            r = (df.oc[long_] - 2 * COST[t])
            for name, per in (("design", df.loc[:SPLIT]), ("holdout", df.loc[SPLIT:])):
                x = r.reindex(per.index).dropna()
                rows.append(dict(tkr=t, thr=thr, period=name, n=len(x), mean_pct=100 * x.mean(), t=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan,
                                 hit=(x > 0).mean()))
        # control: unconditional open->close mean
        for name, per in (("design", df.loc[:SPLIT]), ("holdout", df.loc[SPLIT:])):
            rows.append(dict(tkr=t, thr="all", period=name, n=len(per), mean_pct=100 * (per.oc.mean() - 2 * COST[t]), t=np.nan, hit=(per.oc > 0).mean()))
        # corr of residual with oc return
        rows.append(dict(tkr=t, thr="corr(resid,oc)", period="all", n=len(df), mean_pct=np.corrcoef(df.resid.dropna(), df.oc[df.resid.notna()])[0, 1], t=np.nan, hit=np.nan))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
