"""Alpha hunt batch 10: Kalshi 15-minute markets, single-stock LETFs, EM carry.

Each test prints sub-periods / holdouts so the verdicts in
docs/guide/strategies.md can be re-checked:

- Kalshi BTC 15-min markets (HF narutowindy/kalshiBTC15min: ~1 s quote
  snapshots + CF BRTI, Jun-Oct 2026): fair value = P(60 s BRTI average at
  close >= strike) from trailing 1 h BRTI vol, normal or t(4); take the first
  quote per market that is >= 3/5/10c below fair, with 0 or 10 s delay;
  near expiry, buy the favourite at 85-97c when spot is far from the strike;
- Kalshi 15-min alt coins (HF papylove/kalshi-15m-data: 1-min quotes with
  results + exchange minute bars, BTC/ETH/SOL/XRP/DOGE/HYPE, Jul-Oct 2026),
  same fair value from 1-min spot, fill at this or the next minute's quote;
- single-stock 2x LETF rebalancing: next-day open->close of the underlying
  vs SPY after a < -2 sd day, by LETF pressure = sum |L(L-1)| x LETF 20-day
  $ volume / underlying 20-day $ volume, and a control group of volatile
  names without single-stock LETFs;
- EMLC carry sleeve (6-month EMLC trend up and UUP trend down, 8% vol
  target, else BIL);
- YieldMax roll weeks (weeks with an option-income ETF ex-date) after a
  10%+ prior-week rally in the underlying.
"""
import glob
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm, t as tdist

from backtester.data.cache import cache_dir
from backtester.letf import LETF, down_z, pressure

NARUTO = "https://huggingface.co/datasets/narutowindy/kalshiBTC15min"
PAPY = "https://huggingface.co/datasets/papylove/kalshi-15m-data"
CONTROL = "AFRM UPST ROKU DKNG SNOW U NET SHOP ZS TWLO RIVN LCID ENPH SQ CVNA RBLX DDOG MDB OKTA SOFI".split()
MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def fmt(x: pd.Series, splits, scale=100, unit="c") -> str:
    return " | ".join(f"{a}-{b} {x[a:b].mean() * scale:6.2f}{unit} t{tstat(x[a:b]):.2f} n{len(x[a:b].dropna())}" for a, b in splits)


def hf_dir(repo: str, sub: str, local: str) -> list:
    """Download every parquet under `sub` of an HF dataset once (small daily files)."""
    out = cache_dir(local)
    tree = requests.get(f"{repo.replace('/datasets/', '/api/datasets/')}/tree/main/{sub}", timeout=60).json()
    for x in tree:
        paths = [x["path"] if x["type"] == "file" else f"{x['path']}/part-000.parquet"]   # narutowindy: one part per date
        for p in paths:
            f = out / p.replace("/", "_").replace("=", "_")
            for k in range(5):                          # HF rate-limits bursts with a 429 page
                if not p.endswith(".parquet") or (f.exists() and f.read_bytes()[:4] == b"PAR1"):
                    break
                r = requests.get(f"{repo}/resolve/main/{p}", timeout=120)
                if r.ok and r.content[:4] == b"PAR1":
                    f.write_bytes(r.content)
                else:
                    time.sleep(30 * (k + 1))
    return sorted(glob.glob(str(out / f"{sub}_*.parquet")))


def first_entry(g: pd.DataFrame, fair: str, edge: float, delay_s: int):
    """First quote in a market that is `edge` below fair; fill at the quote `delay_s` later."""
    rows = []
    for m, x in g.groupby("market"):
        x = x.reset_index(drop=True)
        f, ya, na = x[fair].to_numpy(), x.ya.to_numpy(), x.na.to_numpy()
        sig = np.where((f - ya >= edge) & (ya >= 0.03) & (ya <= 0.97), 1,
                       np.where(((1 - f) - na >= edge) & (na >= 0.03) & (na <= 0.97), -1, 0))
        idx = np.flatnonzero(sig)
        if not len(idx):
            continue
        i = idx[0]
        j = int(x.t.searchsorted(x.t.iloc[i] + pd.Timedelta(seconds=delay_s))) if delay_s else i
        if j >= len(x):
            continue
        p = ya[j] if sig[i] == 1 else na[j]
        win = x.yes.iloc[0] if sig[i] == 1 else 1 - x.yes.iloc[0]
        rows.append((x.t.iloc[i], win - p - 0.07 * p * (1 - p)))
    return pd.Series(dict(rows)).sort_index()


def kalshi_btc():
    K = pd.concat([pd.read_parquet(f, columns=["ts_ns", "market", "yes_ask_cents", "no_ask_cents", "yes_bid_cents", "strike_price"])
                   for f in hf_dir(NARUTO, "kalshi", "naruto")])
    B = pd.concat([pd.read_parquet(f, columns=["ts_ns", "value"]) for f in hf_dir(NARUTO, "brti", "naruto")])
    b1 = B.assign(t=pd.to_datetime(B.ts_ns)).drop_duplicates("t").set_index("t").value.sort_index().resample("1s").last().ffill()
    vol = np.log(b1).diff().rolling(3600, min_periods=1800).std().shift(1)
    K["t"], K["market"] = pd.to_datetime(K.ts_ns), K.market.astype(str)

    def close(mk):  # ticker time is ET; the data window is all EDT
        m = re.match(r"KXBTC15M-(\d\d)([A-Z]{3})(\d\d)(\d\d)(\d\d)", mk)
        return pd.Timestamp(2000 + int(m[1]), MON[m[2]], int(m[3]), int(m[4]), int(m[5])) + pd.Timedelta(hours=4) if m else pd.NaT
    cl = {m: close(m) for m in K.market.unique()}
    final = {m: b1[c - pd.Timedelta(seconds=60):c - pd.Timedelta(seconds=1)].mean() for m, c in cl.items() if pd.notna(c)}
    K["close"], K["final"] = K.market.map(cl), K.market.map(final)
    K = K.dropna(subset=["close", "final", "strike_price"])
    K["yes"] = (K.final >= K.strike_price).astype(float)
    K["tau"] = (K.close - K.t).dt.total_seconds()
    K = K[K.tau.between(5, 840)]
    K = K.assign(sec=K.t.astype("int64") // 15_000_000_000).drop_duplicates(["market", "sec"]).sort_values("t")
    S, V = b1.reindex(K.t.dt.floor("1s")).to_numpy(), vol.reindex(K.t.dt.floor("1s")).to_numpy()
    z = np.log(K.strike_price.to_numpy() / S) / (V * np.sqrt(np.maximum(K.tau.to_numpy() - 30, 1)))
    K["fair_n"], K["fair_t"] = 1 - norm.cdf(z), 1 - tdist.cdf(z / np.sqrt(0.5), 4)
    K["ya"], K["na"], K["yb"] = K.yes_ask_cents / 100, K.no_ask_cents / 100, K.yes_bid_cents / 100
    K = K[np.isfinite(K.fair_t) & K.ya.between(0.01, 0.99) & K.na.between(0.01, 0.99)]
    sp = [("2026-06", "2026-08"), ("2026-09", "2026-10")]
    for fair in ("fair_n", "fair_t"):
        for edge in (0.03, 0.05, 0.10):
            for delay in (0, 10):
                print(f"Kalshi BTC 15m, {fair} edge {edge:.2f}, delay {delay}s: {fmt(first_entry(K, fair, edge, delay), sp)}")
    near = K[(K.tau.between(20, 300)) & (np.abs(norm.ppf(1 - K.fair_n.clip(1e-9, 1 - 1e-9))) >= 1.5)]
    a = near[near.ya.between(0.03, 0.15)].assign(p=lambda d: d.na, win=lambda d: 1 - d.yes)
    b = near[(1 - near.yb).between(0.03, 0.15)].assign(p=lambda d: d.ya, win=lambda d: d.yes)
    e = pd.concat([a, b]).sort_values("t").groupby("market").first()
    print(f"Kalshi BTC 15m, last 5 min, >= 1.5 sd from strike, buy the 85-97c side: "
          f"{fmt((e.win - e.p - 0.07 * e.p * (1 - e.p)).set_axis(e.t).sort_index(), sp)}")


def kalshi_alts():
    q = pd.concat([pd.read_parquet(f) for f in hf_dir(PAPY, "quote_history", "papylove")]).drop_duplicates(["ticker", "minute"])
    s = pd.concat([pd.read_parquet(f) for f in hf_dir(PAPY, "spot_history", "papylove")]).drop_duplicates(["coin", "ts"])
    q = q[q.result.isin(["yes", "no"])]
    out = []
    for coin, g in q.groupby("coin"):
        sp = s[s.coin == coin].set_index("ts").close.sort_index()
        if sp.empty:
            continue                                    # metals, ZEC, NEAR: no spot in the dataset
        vol = np.log(sp).diff().rolling(60, min_periods=40).std()
        g = g.sort_values(["ticker", "minute"]).copy()
        tau = (g.close_ts - g.end_period_ts).to_numpy() / 60
        z = np.log(g.floor_strike.to_numpy() / sp.reindex(g.end_period_ts).to_numpy()) / (
            vol.reindex(g.end_period_ts).to_numpy() * np.sqrt(np.maximum(tau, 0.25)))
        g["fair"], g["tau"], g["yes"] = 1 - tdist.cdf(z / np.sqrt(0.5), 4), tau, (g.result == "yes").astype(float)
        g["ya1"], g["yb1"] = g.groupby("ticker").yes_ask.shift(-1), g.groupby("ticker").yes_bid.shift(-1)
        out.append(g.assign(coin=coin))
    G = pd.concat(out)
    G = G[np.isfinite(G.fair) & (G.tau >= 1)]
    for delay in (0, 1):
        ya, yb = (G.yes_ask, G.yes_bid) if delay == 0 else (G.ya1, G.yb1)
        for edge in (0.05, 0.10):
            by, bn = (G.fair - G.yes_ask >= edge) & ya.between(0.03, 0.97), (G.yes_bid - G.fair >= edge) & (1 - yb).between(0.03, 0.97)
            E = pd.concat([G[by].assign(p=ya[by], win=G.yes[by]), G[bn].assign(p=1 - yb[bn], win=1 - G.yes[bn])])
            E = E.sort_values(["ticker", "minute"]).groupby("ticker").first()
            pnl = (E.win - E.p - 0.07 * E.p * (1 - E.p)).set_axis(pd.to_datetime(E.open_ts, unit="s")).sort_index()
            print(f"Kalshi 15m alts, edge {edge:.2f}, {delay} min delay: {fmt(pnl, [('2026-07', '2026-09-14'), ('2026-09-15', '2026-10')])}")


def letf_single_stock():
    tick = sorted(set(LETF) | {s for v in LETF.values() for s in v} | set(CONTROL) | {"SPY"})
    d = yf.download(tick, start="2015-01-01", auto_adjust=True, progress=False)
    c, o, v = d["Close"], d["Open"], d["Volume"]
    spy_oc = c["SPY"].shift(-1) / o["SPY"].shift(-1) - 1

    def down_days(u):
        oc = c[u].shift(-1) / o[u].shift(-1) - 1 - spy_oc - 0.0005     # buy next open, sell its close, 5 bp
        return down_z(c[u]), oc
    ev = []
    for u in LETF:
        z, oc = down_days(u)
        e = pd.DataFrame({"z": z, "p": pressure(c, v, u), "oc": oc}).dropna()
        ev.append(e[(e.z < -2) & (e.index >= "2022-07-01")])
    E = pd.concat(ev).sort_index()
    for th in (0.0, 0.02, 0.05):
        x = E.oc[E.p > th] if th else E.oc[E.p <= 0.02]
        lab = f"pressure > {th}" if th else "pressure <= 0.02"
        print(f"Single-stock LETF, {lab}: {fmt(x, [('2022-07', '2024-06'), ('2024-07', '2025-06'), ('2025-07', '2026')], 1e4, 'bp')}")
    ctl = pd.concat([down_days(u)[1][down_days(u)[0] < -2] for u in CONTROL if u in c]).dropna().sort_index()
    print(f"  control, volatile names without single-stock LETFs: "
          f"{fmt(ctl, [('2015', '2022-06'), ('2022-07', '2024-06'), ('2024-07', '2026')], 1e4, 'bp')}")


def emlc_sleeve():
    m = yf.download("EMLC UUP BIL".split(), start="2010-08-01", auto_adjust=True, progress=False)["Close"]
    vol = m.EMLC.pct_change().rolling(63).std().resample("ME").last() * np.sqrt(252)
    m = m.resample("ME").last()
    mr = m.pct_change()
    sig = ((m.EMLC / m.EMLC.shift(6) > 1) & (m.UUP / m.UUP.shift(6) < 1)).astype(float)
    w = (sig * (0.08 / vol).clip(upper=1.5)).shift(1).fillna(0)
    sl = w * mr.EMLC + (1 - w).clip(lower=0) * mr.BIL - w.diff().abs().fillna(0) * 0.001
    for a, b in (("2011", "2018"), ("2019", "2024-06"), ("2024-07", "2026")):
        x = sl[a:b].dropna()
        print(f"EMLC carry sleeve {a}-{b}: Sharpe {(x - mr.BIL[a:b]).mean() / x.std() * np.sqrt(12):.2f}, {x.mean() * 12:.1%}/yr")


def yieldmax():
    pairs = {"TSLY": "TSLA", "NVDY": "NVDA", "MSTY": "MSTR", "CONY": "COIN", "PLTY": "PLTR", "AMDY": "AMD", "APLY": "AAPL", "AMZY": "AMZN"}
    x = yf.download(list(pairs) + list(pairs.values()) + ["SPY"], start="2022-11-01", auto_adjust=False, actions=True, progress=False)
    c, dv = x["Adj Close"], x["Dividends"]
    rows = []
    for e, u in pairs.items():
        w = c[[u, "SPY"]].resample("W-FRI").last().pct_change()
        roll = (dv[e].fillna(0).resample("W-FRI").sum() > 0).reindex(w.index, fill_value=False)
        rows.append(pd.DataFrame({"rally": (w[u] - w.SPY).shift(1) > 0.10, "roll": roll, "r": w[u] - w.SPY})
                    .dropna()[c[e].first_valid_index():])
    D = pd.concat(rows)
    for (ra, ro), g in D.groupby(["rally", "roll"]):
        print(f"YieldMax underlying vs SPY, prior-week rally {ra}, roll week {ro}: {g.r.mean() * 1e4:.0f}bp t{tstat(g.r):.2f} n{len(g)}")


def main():
    kalshi_btc()
    kalshi_alts()
    letf_single_stock()
    emlc_sleeve()
    yieldmax()


if __name__ == "__main__":
    main()
