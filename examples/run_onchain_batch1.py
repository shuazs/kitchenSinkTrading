"""On-chain batch 1: single-feature long/flat timing of BTC and ETH.

Features from Coin Metrics community CSVs: exchange net flow, MVRV, active
addresses, tx count, hash ribbons, fees, stablecoin supply growth, stablecoin
supply ratio. Signal on day t trades at the day t+1 close (one extra day of
lag, since a day's metrics are final only at its close), 0.5% per side cost.
Design 2018-2022, holdout 2023-01..2026-05. Prints the grid and writes
/mnt/project-files/onchain/batch1_grid.csv when that folder exists.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.data.coinmetrics import load_coin, stablecoin_supply

COST = 0.005
SPLIT = "2023-01-01"
START = {"btc": "2018-01-01", "eth": "2018-01-01"}


def features(coin: str, stable: pd.Series) -> dict[str, pd.Series]:
    d = load_coin(coin)
    f: dict[str, pd.Series] = {}
    if "FlowInExNtv" in d:
        net = d.FlowInExNtv - d.FlowOutExNtv
        gross = (d.FlowInExNtv + d.FlowOutExNtv).rolling(90).mean()
        for w in (3, 7, 30):
            f[f"exnet{w}"] = (net.rolling(w).sum() / gross / w)  # inflow pressure
        f["exsup30"] = d.SplyExNtv.pct_change(30)  # exchange supply change
    f["mvrv"] = d.CapMVRVCur
    for w in (7, 30, 90):
        f[f"mvrv_mom{w}"] = d.CapMVRVCur.pct_change(w)
        f[f"adr{w}"] = d.AdrActCnt.rolling(w).mean() / d.AdrActCnt.rolling(w * 3).mean() - 1
        f[f"tx{w}"] = d.TxTfrCnt.rolling(w).mean() / d.TxTfrCnt.rolling(w * 3).mean() - 1
        f[f"fee{w}"] = (d.FeeTotNtv.rolling(w).mean() / d.FeeTotNtv.rolling(w * 3).mean()) - 1
    if "HashRate" in d:
        f["hash_ribbon"] = d.HashRate.rolling(30).mean() / d.HashRate.rolling(60).mean() - 1
        f["hash_mom90"] = d.HashRate.pct_change(90)
    cap = d.CapMrktCurUSD
    for w in (7, 30, 90):
        f[f"stable{w}"] = stable.reindex(d.index).ffill().pct_change(w)
    f["ssr"] = cap / stable.reindex(d.index).ffill()  # supply ratio: low = lots of dry powder
    f["ssr_mom30"] = f["ssr"].pct_change(30)
    return f


def zscore(s: pd.Series, win=365) -> pd.Series:
    return (s - s.rolling(win, min_periods=180).mean()) / s.rolling(win, min_periods=180).std()


def run(sig: pd.Series, ret: pd.Series) -> pd.Series:
    pos = sig.astype(float).shift(2).reindex(ret.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos * ret - turn * COST


def stats(r: pd.Series, pos: pd.Series | None = None) -> dict:
    if len(r) < 30 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, dd=np.nan)
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return dict(sharpe=r.mean() / r.std() * np.sqrt(365), cagr=eq.iloc[-1] ** (1 / yrs) - 1,
                dd=(eq / eq.cummax() - 1).min())


def main():
    stable = stablecoin_supply()
    rows = []
    bh = {}
    for coin in ("btc", "eth"):
        d = load_coin(coin)
        px = d.PriceUSD.dropna()
        ret = px.pct_change().loc[START[coin]:]
        bh[coin] = ret
        feats = features(coin, stable)
        for name, f in feats.items():
            z = zscore(f)
            for sign, thr in itertools.product((1, -1), (0.0, 0.5, 1.0)):
                sig = (sign * z > thr)
                sig = sig.loc[ret.index]
                r = run(sig, ret)
                a, b = r.loc[:SPLIT].iloc[:-1], r.loc[SPLIT:]
                sa, sb = stats(a), stats(b)
                exp_a = sig.shift(2).loc[a.index].mean()
                exp_b = sig.shift(2).loc[b.index].mean()
                rows.append(dict(coin=coin, feat=name, sign=sign, thr=thr,
                                 sh_des=sa["sharpe"], sh_hold=sb["sharpe"], cagr_des=sa["cagr"], cagr_hold=sb["cagr"],
                                 dd_hold=sb["dd"], exp_des=exp_a, exp_hold=exp_b,
                                 flips_yr=sig.astype(int).diff().abs().sum() / (len(sig) / 365)))
    g = pd.DataFrame(rows)
    for coin in ("btc", "eth"):
        r = bh[coin]
        print(coin, "buy&hold design", {k: round(v, 2) for k, v in stats(r.loc[:SPLIT].iloc[:-1]).items()},
              "holdout", {k: round(v, 2) for k, v in stats(r.loc[SPLIT:]).items()})
    print(f"{len(g)} trials")
    for coin in ("btc", "eth"):
        c = g[g.coin == coin].copy()
        hb = stats(bh[coin].loc[SPLIT:])["sharpe"]
        print(f"\n{coin}: share of cells with holdout Sharpe > buy&hold ({hb:.2f}): {(c.sh_hold > hb).mean():.2f}; "
              f"share positive both periods: {((c.sh_des > 0) & (c.sh_hold > 0)).mean():.2f}")
        top = c.sort_values("sh_des", ascending=False).head(12)
        print(top.round(2).to_string(index=False))
    out = Path("/mnt/project-files/onchain")
    if out.exists():
        g.to_csv(out / "batch1_grid.csv", index=False)


if __name__ == "__main__":
    main()
