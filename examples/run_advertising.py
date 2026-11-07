"""Advertising spend vs stock returns (needs SEC_USER_AGENT; Yahoo for prices).

1. Contemporaneous and 12-month-forward rank correlation of EDGAR AdvertisingExpense
   signals (ad growth, intensity, change in intensity, ad growth minus sales growth)
   with year-demeaned stock returns.
2. Calendar-time quintile portfolios, monthly, holding 0-3 / 3-12 / 0-12 / 12-24 months
   after the 10-K is public (100 days after fiscal year end), IS <= 2019 vs OOS 2020+.
Results: docs/advertising.md. Marketing job postings and Super Bowl side tests were
run inline (see the doc).
"""
import sys
import numpy as np
import pandas as pd
from backtester.data import advertising as A
from backtester.data import fundamentals as f

s = A.signals(A.panel())
s["tic"] = s.cik.map(A.cik_tickers())
s = s[s.tic.notna()]

tics=sorted(set(s.tic)|{"SPY"})
b=f.load_bars(tics,start="2009-06-01")
px=pd.DataFrame({k:v["close"] for k,v in b.items()}).sort_index()
# month-start trading-day closes (entry 1 trading day after month end)
first=px.groupby(px.index.to_period("M")).apply(lambda d:d.iloc[1] if len(d)>1 else d.iloc[0])
first.index=first.index.to_timestamp("M")  # label by month
mret=first.pct_change().shift(-1)  # return earned holding from this month's entry to next month's entry
mret.index=first.index
# signal date = month end labelled m-1 -> hold month m... align: signal at end of month M, holds entry M+1 -> M+2
def run(sig, hold_min=0, hold_max=12, nq=5, label=""):
    rows=[]
    months=first.index
    for M in months[:-2]:
        t=M  # signal at end of M, use rows available <= M
        d=s[(s.available<=t)&(s.available>t-pd.Timedelta(days=30*hold_max))&(s.available<=t-pd.Timedelta(days=30*hold_min))]
        d=d.sort_values("end").groupby("cik").tail(1)
        d=d.dropna(subset=[sig])
        d=d[d.tic.isin(px.columns)]
        if len(d)<40: continue
        d=d.assign(q=pd.qcut(d[sig].rank(method="first"),nq,labels=False))
        nxt=mret.loc[M+pd.offsets.MonthEnd(1)] if (M+pd.offsets.MonthEnd(1)) in mret.index else None
        if nxt is None: continue
        r=nxt.reindex(d.tic).values
        d=d.assign(r=r).dropna(subset=["r"])
        rec={"M":M+pd.offsets.MonthEnd(1),"n":len(d),"univ":d.r.mean(),"spy":nxt["SPY"]}
        for q in range(nq): rec[f"Q{q+1}"]=d[d.q==q].r.mean()
        rows.append(rec)
    return pd.DataFrame(rows).set_index("M")
def summ(df,name,nq=5):
    out=[]
    for lab,sl in [("ALL",slice(None)),("IS<=2019",slice(None,"2019-12-31")),("OOS2020+",slice("2020-01-01",None))]:
        x=df.loc[sl]
        ls=x[f"Q{nq}"]-x["Q1"]; lu=x[f"Q{nq}"]-x["univ"]
        hu=x["Q1"]-x["univ"]
        t=lambda v:v.mean()/v.std()*np.sqrt(len(v))
        out.append(f"{name:28s}{lab:9s} n={len(x):3d} top {x[f'Q{nq}'].mean()*1200:5.1f}%/yr  bot {x['Q1'].mean()*1200:5.1f}  univ {x.univ.mean()*1200:5.1f}  top-univ {lu.mean()*1200:5.1f} (t{t(lu):4.1f})  top-bot {ls.mean()*1200:5.1f} (t{t(ls):4.1f})")
    print("\n".join(out))
def main():
    for sig in ["ad_g","intensity","d_intensity","excess_g"]:
        for hmin,hmax,l in [(0,12,"0-12m"),(0,3,"0-3m"),(3,12,"3-12m"),(12,24,"12-24m")]:
            df=run(sig,hmin,hmax); df.to_pickle(f"/tmp/claude/ad/cal_{sig}_{l}.pkl"); summ(df,f"{sig} {l}")


def contemporaneous():
    """Rank correlation of each signal with year-demeaned returns over the fiscal year
    (contemporaneous) and the 12 months after the filing is public (predictive)."""
    d = s[s.tic.isin(px.columns)].copy()
    def ret(t, a, z):
        p = px[t].dropna()
        if z > p.index[-1] or a < p.index[0]:
            return np.nan
        return p[:z].iloc[-1] / p[:a].iloc[-1] - 1
    d["r_fy"] = [ret(t, e - pd.Timedelta(days=365), e) for t, e in zip(d.tic, d.end)]
    d["r_post12"] = [ret(t, a + pd.Timedelta(days=1), a + pd.Timedelta(days=366)) for t, a in zip(d.tic, d.available)]
    d["yr"] = d.end.dt.year
    for c in ("r_fy", "r_post12"):
        d[c + "_x"] = d[c] - d.groupby("yr")[c].transform("mean")
    for sig in ["ad_g", "rev_g", "excess_g", "d_intensity", "intensity"]:
        for c in ["r_fy_x", "r_post12_x"]:
            x = d[[sig, c, "yr"]].dropna()
            by = x.groupby("yr").apply(lambda g: g[sig].corr(g[c], method="spearman"))
            print(f"{sig:12s}{c:11s} pooled {x[sig].corr(x[c], method='spearman'):6.3f}  "
                  f"mean-by-year {by.mean():6.3f} t {by.mean() / by.std() * np.sqrt(len(by)):5.1f} n={len(x)}")


if __name__ == '__main__':
    contemporaneous()
    main()
