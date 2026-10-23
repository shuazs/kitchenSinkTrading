"""Alpha hunt batch 7: cross-sectional effects (all failed).

Each test prints sub-periods and the Jul 2024+ holdout so the verdicts in
docs/guide/strategies.md can be re-checked:

- LETF rebalancing: next-day reversal (vs SPY) of a sector ETF after a
  > 2 sd day, split by terciles of leverage-weighted LETF $ volume over the
  sector ETF's $ volume (17 sectors, 2010+);
- same-calendar-month seasonality (23 commodity futures, 20 country ETFs),
  commodity 12-1 / 6-1 momentum with January's share, and commodity
  skewness, all monthly L/S thirds at 10 bp per unit of turnover;
- the batch 6 January commodity effect by window (late Dec, Jan days 1-5,
  6-10, 11-15) on the original 13 and on all 23 commodities;
- small-cap sector ETFs (PSC*) after their large-cap peer's week;
- G10 FX value from BIS real effective exchange rates (FRED RB**BIS);
- Binance USDT-M perps (HF linxy daily klines + funding, live symbols only,
  so delisted coins are missing): weekly long most-negative funding vs
  BTC+ETH, and the 30 days after a perp's first listing (log CAR vs BTC,
  and a short's arithmetic P&L after funding).
"""
import io
import json

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtester.data.cache import cache_dir
from backtester.data.fred import load_series

LINXY = "https://huggingface.co/datasets/linxy/USDT-M_Perpetual_Futures"
LETF = {"SOXX": {"SOXL": 3, "SOXS": -3}, "XBI": {"LABU": 3, "LABD": -3}, "XLF": {"FAS": 3, "FAZ": -3},
        "GDX": {"NUGT": 2, "DUST": -2}, "IWM": {"TNA": 3, "TZA": -3}, "QQQ": {"TQQQ": 3, "SQQQ": -3},
        "XLE": {"ERX": 2, "ERY": -2}, "GDXJ": {"JNUG": 2, "JDST": -2}, "XLK": {"TECL": 3, "TECS": -3},
        "XHB": {"NAIL": 3}, "XRT": {"RETL": 3}, "KRE": {"DPST": 3}, "IYR": {"DRN": 3, "DRV": -3},
        "XLY": {"WANT": 3}, "XLI": {"DUSL": 3}, "XLU": {"UTSL": 3}, "SMH": {"USD": 2}}
FUT = ("CL=F NG=F GC=F SI=F PL=F PA=F HG=F ZC=F ZW=F KE=F ZS=F ZL=F ZM=F KC=F SB=F CT=F CC=F OJ=F LE=F HE=F "
       "RB=F HO=F BZ=F").split()
FUT13 = "CL=F NG=F GC=F SI=F HG=F ZC=F ZW=F ZS=F KC=F SB=F CT=F LE=F HE=F".split()
CTRY = "EWA EWC EWG EWH EWI EWJ EWK EWL EWM EWN EWO EWP EWQ EWS EWU EWW EWD EWT EWY EWZ".split()
SECT = {"XLK": "PSCT", "XLF": "PSCF", "XLE": "PSCE", "XLV": "PSCH", "XLI": "PSCI", "XLY": "PSCD", "XLP": "PSCC",
        "XLB": "PSCM", "XLU": "PSCU"}
G10 = {"XM": ("EURUSD=X", 1), "JP": ("USDJPY=X", -1), "GB": ("GBPUSD=X", 1), "CH": ("USDCHF=X", -1),
       "CA": ("USDCAD=X", -1), "AU": ("AUDUSD=X", 1), "NZ": ("NZDUSD=X", 1), "SE": ("USDSEK=X", -1),
       "NO": ("USDNOK=X", -1)}


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def periods(x: pd.Series, splits, scale=1e4, unit="bp") -> str:
    return " | ".join(f"{a}-{b} {x[a:b].mean() * scale:6.1f}{unit} t{tstat(x[a:b]):.2f} n{len(x[a:b].dropna())}"
                      for a, b in splits)


def letf_reversal():
    tick = sorted(set(LETF) | {s for v in LETF.values() for s in v} | {"SPY"})
    d = yf.download(tick, start="2009-01-01", auto_adjust=True, progress=False)
    c, v = d["Close"], d["Volume"]
    r = c.pct_change(fill_method=None)
    ev = []
    for u, legs in LETF.items():
        dv = {s: (c[s] * v[s]).rolling(20).mean().shift(1) for s in [u, *legs]}
        pressure = sum(abs(lv * (lv - 1)) * dv[s].fillna(0) for s, lv in legs.items()) / dv[u]
        z = r[u] / r[u].rolling(60).std().shift(1)
        nxt = (r[u] - r["SPY"]).shift(-1)
        ev.append(pd.DataFrame({"z": z, "p": pressure, "rev": -np.sign(z) * nxt - 2e-4}).dropna().query("abs(z) > 2"))
    e = pd.concat(ev).sort_index()
    lo, hi = e.p.expanding(200).quantile(1 / 3).shift(), e.p.expanding(200).quantile(2 / 3).shift()
    e["terc"] = np.select([e.p <= lo, e.p > hi], ["low", "high"], "mid")
    e = e[lo.notna()]
    for k in ("low", "mid", "high"):
        print(f"LETF reversal, {k} pressure: {periods(e.rev[e.terc == k], [('2010', '2016'), ('2017', '2024-06'), ('2024-07', '2026')])}")


def monthly(c: pd.DataFrame, cols, roll_filter: bool):
    x = c[cols].dropna(how="all")
    r = x[x.index.dayofweek < 5].pct_change(fill_method=None)
    if roll_filter:
        r = r.where(r.abs() < 0.10)  # roll jumps and bad prints
    m = ((1 + r.fillna(0)).resample("ME").prod() - 1).where(r.resample("ME").count() >= 10)
    skew = r.rolling(252, min_periods=200).skew().resample("ME").last()
    return r, m, skew


def long_short(sig: pd.DataFrame, ret: pd.DataFrame, frac=1 / 3, cost=10e-4) -> pd.Series:
    """Monthly L/S thirds on `sig` known at month end t, earned in t+1."""
    q = sig.rank(axis=1, pct=True)
    w = (q >= 1 - frac).astype(float).div((q >= 1 - frac).sum(axis=1), axis=0) \
        - (q <= frac).astype(float).div((q <= frac).sum(axis=1), axis=0)
    w = w.where(sig.notna().sum(axis=1) >= 8, 0).fillna(0).reindex(ret.index).fillna(0)
    p = (w.shift(1) * ret).sum(axis=1) - w.diff().abs().sum(axis=1).shift(1).fillna(0) * cost
    return p[w.shift(1).abs().sum(axis=1) > 0]


def cross_sections():
    c = yf.download(FUT + CTRY, start="1996-01-01", auto_adjust=True, progress=False)["Close"]
    sp = [("2001", "2012"), ("2013", "2024-06"), ("2024-07", "2026")]

    def show(name, p):
        jan = p[p.index.month == 1]
        print(f"{name}: {periods(p, sp, 1200, '%/yr')} | Jan {jan.mean():.2%}/mo, other months {p[p.index.month != 1].mean():.2%}/mo")

    for lab, cols, filt in (("commodities", FUT, True), ("country ETFs", CTRY, False)):
        r, m, skew = monthly(c, cols, filt)
        seas = pd.DataFrame(np.nan, index=m.index, columns=m.columns)
        for mo in range(1, 13):
            sub = m[m.index.month == mo]
            seas.loc[sub.index] = sub.shift(1).rolling(20, min_periods=5).mean().to_numpy()
        show(f"Seasonality {lab}", long_short(seas, m))
        if lab == "commodities":
            for k in (12, 6):
                show(f"Momentum {k}-1", long_short((1 + m).rolling(k).apply(np.prod, raw=True).shift(1) - 1, m))
            show("Low minus high skew", long_short(-skew, m))
            january_windows(c)


def january_windows(c):
    for lab, cols in (("13 commodities", FUT13), ("23 commodities", FUT)):
        r = c[cols].dropna(how="all")
        r = r[r.index.dayofweek < 5].pct_change(fill_method=None).where(lambda x: x.abs() < 0.10)
        out = {k: {} for k in ("Dec last 5", "Jan 1-5", "Jan 6-10", "Jan 11-15")}
        for yr in range(2001, 2027):
            seg = r.loc[str(yr - 1)]
            prev = ((1 + seg).prod() - 1)[seg.count() > 150]
            segd = r.loc[pd.Timestamp(yr - 1, 1, 1):pd.Timestamp(yr - 1, 11, 30)]
            prevd = ((1 + segd).prod() - 1)[segd.count() > 140]  # December windows rank on Jan-Nov
            jan = r[(r.index.year == yr) & (r.index.month == 1)]
            dec = r[(r.index.year == yr - 1) & (r.index.month == 12)]
            if len(jan) < 15 or len(prev) < 8:
                continue
            for k, (days, b0, b1, rank_on) in {"Dec last 5": (dec, -5, None, prevd), "Jan 1-5": (jan, 0, 5, prev),
                                               "Jan 6-10": (jan, 5, 10, prev), "Jan 11-15": (jan, 10, 15, prev)}.items():
                q = rank_on.rank(pct=True)
                w = ((1 + days.iloc[b0:b1]).prod() - 1).reindex(q.index)
                out[k][yr] = w[q >= 2 / 3].mean() - w[q <= 1 / 3].mean()
        for k, v in out.items():
            s = pd.Series(v)
            print(f"January {lab} {k}: winners-losers {s.mean() * 1e4:5.0f}bp t {tstat(s):.2f} | "
                  f"2014+ {s[s.index >= 2014].mean() * 1e4:5.0f}bp t {tstat(s[s.index >= 2014]):.2f}")


def small_cap_lag():
    d = yf.download(list(SECT) + list(SECT.values()) + ["SPY", "IWM"], start="2010-01-01", auto_adjust=True,
                    progress=False)
    c = d["Close"].resample("W-FRI").last()
    o = d["Open"].groupby(d["Open"].index.to_period("W-FRI")).first()
    o.index = o.index.to_timestamp(how="end").normalize()
    o = o.reindex(c.index)
    big, small = list(SECT), [SECT[k] for k in SECT]
    lead = c[big].pct_change().sub(c["SPY"].pct_change(), axis=0)
    nxt = (c[small] / o[small] - 1).sub(c["IWM"] / o["IWM"] - 1, axis=0).shift(-1)  # Monday open to Friday close
    nxt.columns = big
    q = lead.rank(axis=1, pct=True)
    w = ((q >= 2 / 3).astype(float) - (q <= 1 / 3).astype(float)) / 3
    p = (w * nxt).sum(axis=1, min_count=1) - w.diff().abs().sum(axis=1) * 5e-4
    print(f"Small-cap sector lag L/S: {periods(p, [('2010', '2016'), ('2017', '2024-06'), ('2024-07', '2026')], 5200, '%/yr')}")


def fx_value():
    us = np.log(load_series("RBUSBIS"))
    rel = pd.DataFrame({k: np.log(load_series(f"RB{k}BIS")) - us for k in G10})
    rel.index = rel.index + pd.offsets.MonthEnd(0)
    fx = yf.download([v[0] for v in G10.values()], start="1996-01-01", auto_adjust=True,
                     progress=False)["Close"].resample("ME").last()
    ret = pd.DataFrame({k: fx[s].pct_change() if sign == 1 else (1 / fx[s]).pct_change() for k, (s, sign) in G10.items()})
    sig = (-(rel - rel.shift(60))).shift(2).reindex(ret.index)  # BIS publishes with a lag
    p = long_short(sig, ret, cost=3e-4)
    print(f"FX value (5y real change): {periods(p, [('1999', '2012'), ('2013', '2024-06'), ('2024-07', '2026')], 1200, '%/yr')}")


def perps():
    path = cache_dir("linxy") / "all_1d.parquet"
    fpath = cache_dir("linxy") / "all_funding_daily.parquet"
    if not path.exists():
        tree = requests.get("https://huggingface.co/api/datasets/linxy/USDT-M_Perpetual_Futures/tree/main", timeout=60)
        syms = [x["path"] for x in json.loads(tree.text) if x["type"] == "directory"]
        k, f = [], []
        for s in syms:
            try:
                g = lambda kind: pd.read_parquet(io.BytesIO(requests.get(
                    f"{LINXY}/resolve/main/{s}/{s}_{kind}.parquet", timeout=60).content))
                d = g("1d")[["open_time", "close", "quote_volume"]].assign(sym=s)
                k.append(d.astype({"close": float, "quote_volume": float}))
                fr = g("fundingRate")
                f.append(fr.assign(day=fr.calc_time.dt.floor("D")).groupby("day").last_funding_rate.sum()
                         .rename("funding").reset_index().assign(sym=s))
            except Exception:
                continue
        pd.concat(k).to_parquet(path)
        pd.concat(f).to_parquet(fpath)
    k, f = pd.read_parquet(path), pd.read_parquet(fpath)
    c = k.pivot_table(index="open_time", columns="sym", values="close")
    qv = k.pivot_table(index="open_time", columns="sym", values="quote_volume")
    fu = f.pivot_table(index="day", columns="sym", values="funding").reindex(c.index)
    r = c.pct_change(fill_method=None).clip(-0.9, 5)
    sp = [("2020", "2022"), ("2023", "2024-06"), ("2024-07", "2026")]

    elig = (c.notna().cumsum() >= 60) & (qv.rolling(20).mean() > 2e7)
    fund7 = fu.rolling(7, min_periods=5).sum()
    bench = (r["BTCUSDT"] + r["ETHUSDT"]) / 2
    mondays = c.index[c.index.dayofweek == 0]
    rows = {}
    for d0, d1 in zip(mondays[:-1], mondays[1:]):
        s = fund7.loc[d0][elig.loc[d0]].dropna()
        if len(s) < 20:
            continue
        hold = (1 + r.loc[d0:d1].iloc[1:]).prod() - 1
        q = s.rank(pct=True)
        rows[d0] = hold[q[q <= 0.2].index].mean() - ((1 + bench.loc[d0:d1].iloc[1:]).prod() - 1) - 0.002
    print(f"Perps long most-negative funding vs BTC+ETH: {periods(pd.Series(rows), sp, 5200, '%/yr')}")

    first = c.apply(lambda s: s.first_valid_index())
    car, short = {}, {}
    for s, d in first[first > pd.Timestamp("2020-01-15")].items():
        p = c.index.get_loc(d)
        if p + 31 >= len(c) or np.isnan(c[s].iloc[p + 1]) or np.isnan(c[s].iloc[p + 31]):
            continue
        car[d] = np.log(c[s].iloc[p + 31] / c[s].iloc[p + 1]) - np.log(c["BTCUSDT"].iloc[p + 31] / c["BTCUSDT"].iloc[p + 1])
        rr = r[s].iloc[p + 2:p + 32].fillna(0)
        short[d] = -((1 + rr).prod() - 1) + fu[s].iloc[p + 2:p + 32].fillna(0).sum() - 0.002
    car, short = pd.Series(car).sort_index(), pd.Series(short).sort_index()
    print(f"Perp listing 30d log CAR vs BTC: {periods(car, sp, 100, '%')} | median {car.median():.1%}")
    print(f"  short it for 30d, after funding: {periods(short, sp, 100, '%')} | worst {short.min():.0%}")


def main():
    letf_reversal()
    cross_sections()
    small_cap_lag()
    fx_value()
    perps()


if __name__ == "__main__":
    main()
