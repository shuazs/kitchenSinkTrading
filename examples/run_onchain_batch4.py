"""On-chain batch 4: spot BTC ETF activity (IBIT FBTC ARKB BITB GBTC dollar volume) as an inflow proxy.

Yahoo has no ETF share history, so volume is the flow proxy. Signal on day t
(US close) = z-score of 5d ETF dollar volume / BTC volume vs 60d; long BTC
from day t's US close (BTC-USD close of day t+1 as entry, 1 day lag) for 1-5 days,
or long/flat while z>thr. 0.5% per side. Sample Jan 2024 - Oct 2026 (split Jul 2025).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

ETFS = ["IBIT", "FBTC", "ARKB", "BITB", "GBTC"]
COST = 0.005


def main():
    px = yf.download(ETFS + ["BTC-USD"], start="2024-01-11", auto_adjust=False, progress=False, threads=False)
    dv = (px["Close"][ETFS] * px["Volume"][ETFS]).sum(axis=1)
    btc = px["Close"]["BTC-USD"].dropna()
    ret = btc.pct_change()
    rows = []
    for w in (1, 3, 5):
        z = (dv.rolling(w).mean() - dv.rolling(60).mean()) / dv.rolling(60).std()
        for thr in (0.5, 1.0):
            for sign in (1, -1):
                sig = (sign * z > thr).reindex(btc.index).fillna(False)
                pos = sig.shift(2).fillna(False).astype(float)  # known after US close, trade next BTC close
                r = pos * ret - pos.diff().abs().fillna(0) * COST
                for name, x in (("early", r.loc[:"2025-06-30"]), ("late", r.loc["2025-07-01":])):
                    rows.append(dict(w=w, thr=thr, sign=sign, per=name, sharpe=x.mean() / x.std() * np.sqrt(365), exp=pos.loc[x.index].mean(),
                                     mean_on_bp=1e4 * ret.loc[x.index][pos.loc[x.index] > 0].mean() if (pos.loc[x.index] > 0).any() else np.nan))
    g = pd.DataFrame(rows)
    for name, x in (("early", ret.loc[:"2025-06-30"]), ("late", ret.loc["2025-07-01":])):
        print("BTC hold", name, round(x.mean() / x.std() * np.sqrt(365), 2), "mean bp/day", round(1e4 * x.mean(), 1))
    print(g.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
