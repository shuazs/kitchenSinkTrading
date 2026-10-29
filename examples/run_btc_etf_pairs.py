"""Spot-bitcoin ETF pair reversion (pre-registered, kalshi-macro/HYPOTHESES.md part 5).

IBIT, FBTC, BITB, ARKB, HODL and BTCO all hold spot bitcoin, so their price
gaps should close. Each ETF's log price minus the others' mean, as a 20-day
z-score; at each close go long the lowest and short the highest when the z
range is >= 2, exit at the next close; 2 bp per side per ETF and 5%/yr
borrow. In-sample Feb-Sep 2024, holdout from Oct 2024.
Result: +0.7 bp/day in-sample (t 0.9), -1.7 bp/day holdout (t -4.5) = FAIL.

    python examples/run_btc_etf_pairs.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

ETFS = ["IBIT", "FBTC", "BITB", "ARKB", "HODL", "BTCO"]
HOLD = "2024-10-01"


def main():
    px = yf.download(ETFS, start="2024-01-15", auto_adjust=True, progress=False)["Close"].dropna()
    lp = np.log(px)
    sp = lp.sub(lp.mean(axis=1), axis=0)
    z = (sp - sp.rolling(20).mean()) / sp.rolling(20).std()
    nxt = px.pct_change().shift(-1)
    pnl = {}
    for d in z.index[20:-1]:
        zz = z.loc[d].dropna()
        if zz.max() - zz.min() >= 2:
            pnl[d] = nxt.loc[d, zz.idxmin()] - nxt.loc[d, zz.idxmax()] - 4 * 0.0002 - 0.05 / 252
    x = pd.Series(pnl)
    for lab, g in [("in-sample", x[x.index < HOLD]), ("holdout", x[x.index >= HOLD])]:
        print(f"{lab}: {len(g)} days, {g.mean() * 1e4:+.2f} bp/day, t {g.mean() / g.std() * np.sqrt(len(g)):.2f}")


if __name__ == "__main__":
    main()
