"""On-chain batch 2: cross-section of 10 coins ranked by on-chain value/activity.

Monthly (or weekly) long-only top-3 of btc eth ltc bch doge xrp ada bnb trx link
(Coin Metrics community data; coins that survived to 2026, so every row is
survivor-biased and the comparison is against the equal-weight basket of the
same coins). Signals: cap per active address (value), active-address growth,
tx growth, MVRV (low = long), 30d price momentum as the control. Signal on
day t trades at day t+1 close, 0.5% per side. Design 2019-2022, holdout 2023+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.data.coinmetrics import load_coin

COINS = ["btc", "eth", "ltc", "bch", "doge", "xrp", "ada", "bnb", "trx", "link"]
COST = 0.005
SPLIT = "2023-01-01"


def panel():
    d = {c: load_coin(c) for c in COINS}
    px = pd.DataFrame({c: d[c].PriceUSD for c in COINS})
    cap = pd.DataFrame({c: d[c].CapMrktEstUSD if "CapMrktEstUSD" in d[c] else np.nan for c in COINS})
    adr = pd.DataFrame({c: d[c].AdrActCnt for c in COINS})
    tx = pd.DataFrame({c: d[c].TxTfrCnt for c in COINS})
    mvrv = pd.DataFrame({c: d[c]["CapMVRVCur"] if "CapMVRVCur" in d[c] else np.nan for c in COINS}, index=px.index)
    return px, cap, adr, tx, mvrv


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan


def stats(r):
    eq = (1 + r).cumprod()
    return dict(sh=sharpe(r), cagr=eq.iloc[-1] ** (365 / len(r)) - 1, dd=(eq / eq.cummax() - 1).min())


def long_top(score, px, k, step):
    """Equal-weight top-k by score, rebalanced every `step` days; returns net daily returns."""
    ret = px.pct_change()
    w = pd.DataFrame(np.nan, index=score.index, columns=px.columns)
    for t in score.index[::step]:
        s = score.loc[t].dropna()
        s = s[px.loc[t, s.index].notna()]
        w.loc[t] = 0.0
        if len(s) >= k + 2:
            w.loc[t, s.nlargest(k).index] = 1.0 / k
    w = w.ffill().fillna(0.0).shift(2)  # signal at t, trade at t+1 close, earn from t+2
    turn = w.diff().abs().sum(axis=1)
    return (w * ret.fillna(0.0)).sum(axis=1) - turn * COST


def main():
    px, cap, adr, tx, mvrv = panel()
    start = "2019-01-01"
    px = px.loc[start:]
    ew = px.pct_change().mean(axis=1)
    sig = {
        "mom30": px.pct_change(30),
        "mom90": px.pct_change(90),
        "low_cap_per_adr": -(cap / adr),
        "adr_growth30": adr.rolling(30).mean() / adr.rolling(90).mean() - 1,
        "tx_growth30": tx.rolling(30).mean() / tx.rolling(90).mean() - 1,
        "adr_g_minus_px30": (adr.rolling(30).mean() / adr.rolling(90).mean() - 1) - px.pct_change(30),
        "low_mvrv": -mvrv,
        "mvrv_mom30": mvrv.pct_change(30),
    }
    print("EW basket design", {k: round(v, 2) for k, v in stats(ew.loc[start:SPLIT].iloc[:-1]).items()},
          "holdout", {k: round(v, 2) for k, v in stats(ew.loc[SPLIT:]).items()})
    rows = []
    for name, s in sig.items():
        s = s.loc[start:]
        for k in (2, 3):
            for step in (7, 30):
                r = long_top(s, px, k, step).loc[start:]
                a, b = stats(r.loc[:SPLIT].iloc[:-1]), stats(r.loc[SPLIT:])
                rows.append(dict(sig=name, k=k, step=step, sh_des=a["sh"], sh_hold=b["sh"], cagr_des=a["cagr"], cagr_hold=b["cagr"], dd_hold=b["dd"]))
    g = pd.DataFrame(rows)
    hb = sharpe(ew.loc[SPLIT:])
    print(f"\n{len(g)} trials; EW holdout Sharpe {hb:.2f}")
    print(g.round(2).sort_values("sh_des", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
