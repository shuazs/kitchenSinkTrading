"""Kalshi EUR/USD and USD/JPY "above / below / between" contracts vs live FX.

Pre-registered (/mnt/project-files/kalshi-fx/HYPOTHESES.md): at each taker
print in the last 60 min before the fixing, fair value is a driftless
lognormal P(above k) = Phi(ln(S/k) / (sigma sqrt(tau))), S = OANDA 1-min close
one full minute before the print, sigma = trailing 60-min realized vol per
minute, tau = minutes to the fixing. Buy the printed side when the model edge
after the taker fee is > 3c (5-95c, first qualifying print per market), hold
to settlement. In-sample before 2025-10-01, holdout after. Placebo: S lagged
30 min. t is clustered by fixing time.

Data: /mnt/project-files/kalshi-fx/ (HF TrevorJS/kalshi-trades markets and
prints, HF bunqi OANDA 1-min bars), Aug 2022 - Jan 2026.

    python examples/run_kalshi_fx_catchup.py
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtester.event_markets import kalshi_fee

D = "/mnt/project-files/kalshi-fx/"
HOLD = "2025-10-01"
SERIES = {"EURUSD": "eurusd", "EURUSDH": "eurusd", "KXEURUSD": "eurusd", "KXEURUSDH": "eurusd",
          "USDJPY": "usdjpy", "USDJPYH": "usdjpy", "KXUSDJPY": "usdjpy", "KXUSDJPYH": "usdjpy"}


def parse(title):
    """-> (kind, lo, hi): 'above' k, 'below' k or 'between' lo..hi."""
    t = title.lower()
    nums = [float(x) for x in re.findall(r"\d+\.\d+", t)]
    if " between " in t and len(nums) >= 2:
        return "between", nums[0], nums[1]
    if " above " in t and nums:
        return "above", nums[0], np.nan
    if " below " in t and nums:
        return "below", nums[0], np.nan
    return None, np.nan, np.nan


def fx_frame(name):
    f = pd.read_parquet(D + f"{name}_1m.parquet")
    f.index = f.index.as_unit("us")
    r = np.log(f["close"]).diff()
    f["sig"] = r.rolling(60, min_periods=40).std()
    return f[["open", "close", "sig"]]


def p_model(kind, lo, hi, s, sig, tau):
    sd = sig * np.sqrt(np.maximum(tau, 1.0))
    above = lambda k: norm.cdf(np.log(s / k) / sd)
    return np.where(kind == "above", above(lo), np.where(kind == "below", 1 - above(lo), above(lo) - above(hi)))


def build(lag_min=1):
    m = pd.read_parquet(D + "markets.parquet")
    m = m[m["series"].isin(SERIES) & m["result"].isin(["yes", "no"])]
    p = pd.read_parquet(D + "prints.parquet")
    m = m[m["ticker"].isin(p["ticker"].unique())].copy()
    m[["kind", "lo", "hi"]] = pd.DataFrame(m["title"].map(parse).tolist(), index=m.index)
    m = m.dropna(subset=["kind"])
    m["y"] = (m["result"] == "yes").astype(int)
    m["pair"] = m["series"].map(SERIES)
    p = p.merge(m[["ticker", "series", "pair", "kind", "lo", "hi", "y", "close_time"]], on="ticker")
    p["mins"] = (p["close_time"] - p["created_time"]).dt.total_seconds() / 60
    out = []
    for pair, g in p.groupby("pair"):
        f = fx_frame(pair)
        at = g["created_time"].dt.floor("min") - pd.Timedelta(minutes=lag_min + 1)  # bar closed >= lag min before
        fx = f.reindex(pd.DatetimeIndex(at))
        g = g.assign(S=fx["close"].values, sig=fx["sig"].values,
                     tau=(g["close_time"] - at - pd.Timedelta(minutes=1)).dt.total_seconds().values / 60)
        settle = f["open"].reindex(pd.DatetimeIndex(g["close_time"])).values
        g["S_settle"] = settle
        out.append(g)
    p = pd.concat(out).dropna(subset=["S", "sig"])
    p = p[p["sig"] > 0]
    p["q"] = p_model(p["kind"].values, p["lo"].values, p["hi"].values, p["S"].values, p["sig"].values, p["tau"].values)
    return m, p


def settle_check(p):
    g = p.drop_duplicates("ticker").dropna(subset=["S_settle"])
    pred = np.where(g["kind"] == "above", g["S_settle"] > g["lo"],
                    np.where(g["kind"] == "below", g["S_settle"] < g["lo"],
                             (g["S_settle"] >= g["lo"]) & (g["S_settle"] <= g["hi"])))
    return (pred.astype(int) == g["y"]).mean(), len(g)


def trades(p, thr=0.03, window=60):
    e = p[(p["mins"] <= window) & (p["mins"] > 0)].copy()
    e["px"] = np.where(e["taker_side"] == "yes", e["yes_price"] / 100, 1 - e["yes_price"] / 100)
    e["qx"] = np.where(e["taker_side"] == "yes", e["q"], 1 - e["q"])
    e["edge"] = e["qx"] - e["px"] - kalshi_fee(e["px"])
    e = e[e["px"].between(0.05, 0.95) & (e["edge"] > thr)].sort_values("created_time")
    e = e.drop_duplicates("ticker")
    win = np.where(e["taker_side"] == "yes", e["y"], 1 - e["y"])
    return e.assign(net=win - e["px"] - kalshi_fee(e["px"]))


def stats(t):
    if len(t) < 3:
        return f"{len(t)} trades"
    e = t.groupby("close_time")["net"].mean()
    tt = e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 and e.std() > 0 else np.nan
    return f"{len(t):>4} trades {len(e):>4} fixings net {t['net'].mean() * 100:+6.2f}c t {tt:+5.2f}"


def show(label, t):
    print(f"{label:<30} IS {stats(t[t['close_time'] < HOLD])}  |  HO {stats(t[t['close_time'] >= HOLD])}")


def main():
    m, p = build()
    acc, n = settle_check(p)
    print(f"{len(m)} settled FX markets with prints; OANDA open at the fixing matches the result on {acc:.1%} of {n}")
    late = p[p["mins"] <= 60]
    print(f"prints in last 60 min: {len(late)} on {late['ticker'].nunique()} markets "
          f"(holdout {int((late['close_time'] >= HOLD).sum())})")
    late = late.assign(net_taker=np.where(late["taker_side"] == "yes", late["y"] - late["yes_price"] / 100,
                                          (1 - late["y"]) - (1 - late["yes_price"] / 100)))
    print(f"model log-loss vs price, last 60 min: model {log_loss(late['y'], late['q']):.4f}, "
          f"price {log_loss(late['y'], late['yes_price'] / 100):.4f}")
    for thr in (0.03, 0.05, 0.10):
        show(f"H1 edge > {thr * 100:.0f}c", trades(p, thr))
    for kind in ("hourly", "daily"):
        t = trades(p)
        h = t["series"].str.endswith("H")
        show(f"  H1 3c {kind} only", t[h if kind == "hourly" else ~h])
    _, pl = build(lag_min=30)
    show("H2 placebo (FX 30 min stale)", trades(pl))


def log_loss(y, q):
    q = np.clip(np.asarray(q, dtype=float), 0.01, 0.99)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


if __name__ == "__main__":
    main()
