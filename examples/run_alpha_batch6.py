"""Alpha hunt batch 6: forced flows outside long Treasuries (daily data).

Each test prints sub-periods and the Jul 2024+ holdout so the verdicts in
docs/guide/strategies.md can be re-checked:

- FX month-end hedge rebalancing: SPY minus EFA (EWJ) month to date at the
  close 3 days before month end -> short USD vs EUR (JPY) for the last 2
  days; placebo on days 10-12 and the unconditional month-end;
- fallen angels: ANGL/FALN weekly excess return on HYG (JNK) + IEF;
- muni season: MUB/HYD minus rolling-beta IEF by month group;
- carry-crash reversal: AUDJPY or AUD/NZD/MXN-JPY basket 20 days after a
  10-day drop beyond 2-3 sd with VIX > 25 (placebo: any VIX);
- commodity January reweight: prior-year winners minus losers (13 Yahoo
  front-month futures, |daily move| > 10% dropped as roll jumps) on January
  business days 6-10, other months as placebo, neighbouring windows;
- ex-US pension month-end: EFA/EWJ/EWG vs BWX, same rule as the US keeper;
- earnings-season leverage: 1.5x SPY in weeks of the quarter that had the
  most S&P 500 8-K filings over the previous 8 quarters (EDGAR form.idx);
- SVXY bought at the next open after a VIX +20% day, held 3 days;
- dividend reinvestment: SPY on days with top-decile S&P 500 dividend
  payments, pay date = ex-date + 7/14/21/28 days (HF data has no pay dates).
"""
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtester.data.cache import cache_dir
from backtester.data.sp500 import membership

SPLITS = [("1996", "2015"), ("2016", "2024-06"), ("2024-07", "2026")]
FUT = "CL=F NG=F GC=F SI=F HG=F ZC=F ZW=F ZS=F KC=F SB=F CT=F LE=F HE=F".split()
DIV = "https://huggingface.co/datasets/siddharthmb/stocks-dividends/resolve/main/dividend.csv"


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def thin(mask: pd.Series, gap: int) -> list[int]:
    keep, last = [], -gap
    for i in np.flatnonzero(mask.to_numpy()):
        if i - last >= gap:
            keep.append(i)
            last = i
    return keep


def line(e: pd.Series, splits=SPLITS) -> str:
    return " | ".join(f"{a}-{b} {e[a:b].mean() * 1e4:6.1f}bp n{len(e[a:b])} t{tstat(e[a:b]):.2f}" for a, b in splits)


def month_groups(idx: pd.DatetimeIndex):
    return pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M"))


def fx_month_end(c):
    for fx, eq, sign in (("EURUSD=X", "EFA", 1), ("USDJPY=X", "EWJ", -1)):  # sign: + when short USD = long fx
        d = c[["SPY", eq, fx]].dropna()
        for X in (0.0, 0.01, 0.02):
            end, mid = {}, {}
            for _, pos in month_groups(d.index):
                p = pos.to_numpy()
                if len(p) < 15 or p[0] == 0:
                    continue
                for store, s, e in ((end, p[-3], p[-1]), (mid, p[9], p[11])):
                    gap = d.SPY.iloc[s] / d.SPY.iloc[p[0] - 1] - d[eq].iloc[s] / d[eq].iloc[p[0] - 1]
                    if abs(gap) > X:
                        store[d.index[s]] = np.sign(gap) * sign * (d[fx].iloc[e] / d[fx].iloc[s] - 1) - 3e-4
            print(f"FX {fx} gap>{X:.0%} month-end: {line(pd.Series(end))}")
            print(f"FX {fx} gap>{X:.0%} days 10-12 (placebo): {line(pd.Series(mid))}")


def fallen_angels(c):
    w = c[["ANGL", "FALN", "HYG", "JNK", "IEF", "BIL"]].resample("W-FRI").last().pct_change()
    w = w.sub(w["BIL"], axis=0)
    for y, hy in (("ANGL", "HYG"), ("FALN", "HYG")):
        for a, b in (("2012", "2018"), ("2019", "2024-06"), ("2024-07", "2026")):
            d = w[[y, hy, "IEF"]][a:b].dropna()
            X = np.column_stack([np.ones(len(d)), d[[hy, "IEF"]]])
            beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)
            e = d[y].to_numpy() - X @ beta
            se = np.sqrt(e @ e / (len(d) - 3) * np.linalg.inv(X.T @ X)[0, 0])
            print(f"Fallen angels {y}~{hy}+IEF {a}-{b}: alpha {beta[0] * 52:6.2%}/yr t {beta[0] / se:.2f}")


def muni_season(c):
    for m in ("MUB", "HYD"):
        r = c[[m, "IEF"]].dropna().pct_change()
        beta = (r[m].rolling(252).cov(r["IEF"]) / r["IEF"].rolling(252).var()).shift(1)
        mo = (r[m] - beta * r["IEF"]).dropna().resample("MS").sum()
        for a, b in (("2008", "2015"), ("2016", "2024-06"), ("2024-07", "2026")):
            x = mo[a:b]
            on, off = x[x.index.month.isin([6, 7, 8])], x[x.index.month.isin([4, 10, 11])]
            print(f"Muni {m} {a}-{b}: Jun-Aug {on.mean() * 1e4:5.1f}bp/mo t {tstat(on):.2f} | "
                  f"Apr+Oct-Nov {off.mean() * 1e4:5.1f}bp t {tstat(off):.2f}")


def carry_crash(c):
    fx = c[["AUDJPY=X", "NZDJPY=X", "USDJPY=X", "MXN=X", "^VIX"]].dropna()
    fx["MXNJPY"] = fx["USDJPY=X"] / fx["MXN=X"]
    basket = np.exp(np.log(fx[["AUDJPY=X", "NZDJPY=X", "MXNJPY"]]).diff().mean(axis=1).cumsum())
    for name, s in (("AUDJPY", fx["AUDJPY=X"]), ("basket", basket)):
        r10 = s / s.shift(10) - 1
        sd = r10.rolling(252).std()
        fwd = s.shift(-21) / s.shift(-1) - 1 - 4e-4
        for k in (2, 3):
            for lab, v in (("VIX>25", fx["^VIX"] > 25), ("any VIX", True)):
                print(f"Carry {name} drop>{k}sd {lab}: {line(fwd.iloc[thin((r10 < -k * sd) & v, 20)].dropna())}")


def commodity_january(c):
    r = c[FUT].dropna(how="all")
    r = r[r.index.dayofweek < 5].pct_change(fill_method=None)
    r = r.where(r.abs() < 0.10)  # roll jumps and bad prints

    def wml(month, b0, b1, look=12):
        out = {}
        for yr in range(2001, 2027):
            hi = pd.Timestamp(yr, month, 1) - pd.Timedelta(days=1)
            seg = r.loc[hi - pd.DateOffset(months=look) + pd.Timedelta(days=1):hi]
            prev = ((1 + seg).prod() - 1)[seg.count() > look * 12]
            d = r[(r.index.year == yr) & (r.index.month == month)]
            if len(d) < 15 or len(prev) < 8:
                continue
            q = prev.rank(pct=True)
            w = ((1 + d.iloc[b0:b1]).prod() - 1).reindex(q.index)
            out[pd.Timestamp(yr, month, 1)] = w[q >= 2 / 3].mean() - w[q <= 1 / 3].mean() - 0.002
        return pd.Series(out)

    j = wml(1, 5, 10)
    print("Commodity Jan bdays 6-10 winners-losers (net 20bp):",
          line(j, [("2001", "2013"), ("2014", "2024"), ("2025", "2026")]), f"| all t {tstat(j):.2f}")
    print("  other months, same days:", {m: round(wml(m, 5, 10).mean() * 1e4) for m in range(2, 13)})
    for b0, b1 in ((4, 9), (6, 11), (5, 8), (7, 10)):
        print(f"  Jan bdays {b0 + 1}-{b1}: t {tstat(wml(1, b0, b1)):.2f}")
    for look in (3, 6):
        print(f"  {look}m lookback: t {tstat(wml(1, 5, 10, look)):.2f}")


def pension_ex_us(c, days=4):
    tlt = yf.download("TLT", start="2002-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    c = c.assign(TLT=tlt)
    for st, bd in (("SPY", "TLT"), ("EFA", "BWX"), ("EWJ", "BWX"), ("EWG", "BWX")):
        d = c[[st, bd]].dropna()
        rows = {}
        for per, pos in month_groups(d.index):
            p = pos.to_numpy()
            if len(p) < 10 or p[0] == 0:
                continue
            s, b = p[-days - 1], p[0] - 1
            gap = d[st].iloc[s] / d[st].iloc[b] - d[bd].iloc[s] / d[bd].iloc[b]
            rows[per.to_timestamp()] = (gap, d[st].iloc[p[-1]] / d[st].iloc[s] - 1)
        e = pd.DataFrame(rows, index=["gap", "ret"]).T
        lo = e.gap.expanding(36).quantile(1 / 3).shift()
        x = e.ret[e.gap < lo]
        print(f"Pension {st}/{bd} bonds lead: {line(x, [('2008', '2016'), ('2017', '2024-06'), ('2024-07', '2026')])}")


def eight_k_counts(y: int, q: int, ciks: set[int]) -> pd.DataFrame:
    """(date, cik, n) of 8-K filings by `ciks` in one quarter's form.idx,
    streamed and cached as counts only."""
    path = cache_dir("edgar_forms") / f"8K_{y}Q{q}.csv"
    if not path.exists():
        url = f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        rows: dict = {}
        with requests.get(url, headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=300, stream=True) as r:
            r.raise_for_status()
            for ln in r.iter_lines(decode_unicode=True):
                f = ln.split()
                if len(f) > 4 and f[0] == "8-K" and f[-3].isdigit() and int(f[-3]) in ciks:
                    rows[(f[-2], int(f[-3]))] = rows.get((f[-2], int(f[-3])), 0) + 1
        pd.DataFrame([(*k, n) for k, n in rows.items()], columns=["date", "cik", "n"]).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])


def earnings_season(c):
    tick = set().union(*membership()["2004":].values)
    j = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=60).json()
    cik = {v["ticker"].replace("-", "."): v["cik_str"] for v in j.values()}
    ciks = {cik[t] for t in tick if t in cik}
    f = pd.concat([eight_k_counts(y, q, ciks) for y in range(2005, 2027) for q in range(1, 5)
                   if pd.Timestamp(y, 3 * q - 2, 1) < pd.Timestamp.today()])
    wk = f.groupby("date").cik.nunique().resample("W-FRI").sum()
    qtr = wk.index.to_period("Q")
    woq = (wk.index - qtr.start_time).days // 7
    W = pd.DataFrame({"n": wk.to_numpy(), "woq": woq, "q": qtr}, index=wk.index)
    exp = W.pivot_table(index="q", columns="woq", values="n", aggfunc="sum").shift(1).rolling(8, min_periods=4).mean()
    W["exp"] = [exp.at[q, w] if q in exp.index and w in exp.columns else np.nan for q, w in zip(W.q, W.woq)]
    W["heavy"] = W.exp > W.groupby("q").exp.transform(lambda s: s.quantile(2 / 3))  # known before the quarter
    spy = c["SPY"].dropna()
    x = spy.pct_change() - c["BIL"].reindex(spy.index).ffill().pct_change().fillna(0)
    heavy = W.heavy.reindex(spy.index, method="bfill").fillna(False).astype(bool)
    lev = pd.Series(np.where(heavy, 1.5, 1.0), index=spy.index)
    for a, b in (("2006", "2014"), ("2015", "2024-06"), ("2024-07", "2026")):
        xx, lv = x[a:b], lev[a:b]
        y = xx * lv - lv.diff().abs().fillna(0) * 2e-4
        print(f"Earnings season {a}-{b}: Sharpe 1x {xx.mean() / xx.std() * 16:.2f} -> 1.5x heavy weeks "
              f"{y.mean() / y.std() * 16:.2f}")


def svxy_spike(c, o):
    v = c[["^VIX", "SVXY"]].dropna()
    scale = pd.Series(np.where(v.index < "2018-02-28", 0.5, 1.0), index=v.index)  # -1x before Feb 2018
    ret = (v.SVXY.shift(-3) / o["SVXY"].reindex(v.index).shift(-1) - 1) * scale - 6e-4
    for th in (0.15, 0.20, 0.30):
        e = ret.iloc[thin(v["^VIX"].pct_change() > th, 3)].dropna()
        print(f"SVXY after VIX +{th:.0%}: {line(e, [('2011', '2017'), ('2018', '2024-06'), ('2024-07', '2026')])}")


def dividends(c):
    path = cache_dir("hf_dividends") / "dividend.csv"
    if not path.exists():
        path.write_bytes(requests.get(DIV, timeout=300).content)
    dv = pd.read_csv(path, parse_dates=["ex_date"])
    m = membership()["2000":]
    pos = m.index.searchsorted(dv.ex_date, side="right") - 1
    dv = dv[pos >= 0]
    dv = dv[[s in m.iloc[p] for s, p in zip(dv.act_symbol, pos[pos >= 0])]]
    spy = c["SPY"].dropna()
    r, idx = spy.pct_change(), spy.index
    for lag in (7, 14, 21, 28):
        pay = idx[idx.searchsorted(dv.ex_date + pd.Timedelta(days=lag)).clip(max=len(idx) - 1)]
        n = pd.Series(1, index=pay).groupby(level=0).sum().reindex(idx, fill_value=0)["2005":]
        top = n > n.rolling(252).quantile(0.9).shift(1)
        parts = []
        for a, b in (("2006", "2015"), ("2016", "2024-06"), ("2024-07", "2026")):
            x, tp = r[a:b], top[a:b]
            se = np.sqrt(x[tp].var() / tp.sum() + x[~tp].var() / (~tp).sum())
            parts.append(f"{a}-{b} {(x[tp].mean() - x[~tp].mean()) * 1e4:5.1f}bp t {(x[tp].mean() - x[~tp].mean()) / se:.2f}")
        print(f"Dividends pay = ex + {lag}d, top-decile days minus rest: " + " | ".join(parts))


def main():
    tick = ("EURUSD=X USDJPY=X AUDJPY=X NZDJPY=X MXN=X SPY EFA EWJ EWG ANGL FALN HYG JNK IEF MUB HYD BIL BWX "
            "^VIX SVXY").split() + FUT
    px = yf.download(tick, start="1995-01-01", auto_adjust=True, progress=False)
    c = px["Close"]
    fx_month_end(c)
    fallen_angels(c)
    muni_season(c)
    carry_crash(c)
    commodity_january(c)
    pension_ex_us(c)
    earnings_season(c)
    svxy_spike(c, px["Open"])
    dividends(c)


if __name__ == "__main__":
    main()
