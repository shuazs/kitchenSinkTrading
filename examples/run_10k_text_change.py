"""10-K text-change signals (Lazy Prices style) from HF milosand1/sec-filings-tone carryover.parquet.
Per filing and item (1A risk factors, 7 MD&A): share of text not carried from the prior year, log length change.
Events/returns come from run_10k_llm_tone.py (/tmp/ec/ev10k.parquet). Ranked within filing year.
Usage: PYTHONPATH=. python examples/run_10k_text_change.py CARRYOVER.parquet EV10K.parquet"""
import sys
import numpy as np, pandas as pd
from scipy import stats

c = pd.read_parquet(sys.argv[1]); ev = pd.read_parquet(sys.argv[2])
g = c.groupby(["ticker", "fy", "item"]).agg(chars=("chars", "sum"), carried=("carried", "sum")).reset_index()
g["new"] = 1 - g.carried / g.chars
g["len"] = np.log(g.chars)
g = g.sort_values(["ticker", "item", "fy"]); g["dlen"] = g.groupby(["ticker", "item"]).len.diff()
w = g.pivot(index=["ticker", "fy"], columns="item", values=["new", "dlen"]); w.columns = [f"{a}_{b}" for a, b in w.columns]
d = ev[["ticker", "fy", "entry", "r5", "r20", "r60", "d_pol"]].merge(w.reset_index(), on=["ticker", "fy"])
d["yr"] = d.entry.dt.year
d = d[d.fy > 2019]  # FY2019 has no prior year
sigs = [x for x in w.columns]
print(len(d), "events")
for s in sigs:
    for h in (5, 20, 60):
        ic = d.groupby("yr").apply(lambda x: stats.spearmanr(x[s], x[f"r{h}"], nan_policy="omit")[0]).dropna()
        r = []
        for y, x in d.dropna(subset=[s]).groupby("yr"):
            if len(x) < 100: continue
            r.append(x[x[s] <= x[s].quantile(.2)][f"r{h}"].mean() - x[x[s] >= x[s].quantile(.8)][f"r{h}"].mean())  # low change minus high change
        r = np.array(r)
        print(f"{s:10s} h={h:2d} IC {ic.mean():6.3f} t {ic.mean()/ic.std()*np.sqrt(len(ic)):5.2f} | lowchg-minus-highchg {r.mean()*100:5.2f}% t {r.mean()/r.std()*np.sqrt(len(r)):5.2f} pos {np.mean(r>0):.2f} n_yr {len(r)}")
