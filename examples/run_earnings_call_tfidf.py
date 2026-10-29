"""Learned-lexicon test: hashed TF-IDF of the transcript -> walk-forward ridge on 20d excess return.
Usage: PYTHONPATH=. python examples/run_earnings_call_tfidf.py EVENTS.parquet SHARD...  (events from run_earnings_call_backtest.py)"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge

ev = pd.read_parquet(sys.argv[1]).reset_index(drop=True)
key = {(r.ticker, str(pd.Timestamp(r.earnings_date).date())): i for i, r in ev.iterrows()}
hv = HashingVectorizer(n_features=2**18, ngram_range=(1, 2), alternate_sign=False, stop_words="english", norm=None)
rows, idx = [], []
for sh in sys.argv[2:]:
    f = pq.ParquetFile(sh)
    for g in range(f.num_row_groups):
        df = f.read_row_group(g, columns=["ticker", "earnings_date", "transcript"]).to_pandas()
        for t, e, x in zip(df.ticker, df.earnings_date, df.transcript):
            i = key.get((t, str(pd.Timestamp(e).date())))
            if i is not None and x:
                rows.append(hv.transform([x[:60000]])); idx.append(i)
        print(sh, g, len(idx), flush=True)
X = TfidfTransformer(sublinear_tf=True).fit_transform(sp.vstack(rows)).tocsr()
d = ev.loc[idx].reset_index(drop=True)
d["q"] = pd.to_datetime(d.entry).dt.to_period("Q")
res = {}
for h in (5, 20, 60):
    pred = pd.Series(np.nan, index=d.index)
    for t in sorted(d.q.unique()):
        if t < pd.Period("2017Q1"): continue
        tr = (pd.to_datetime(d.entry) < t.start_time - pd.Timedelta(days=90)) & d[f"r{h}"].notna()
        te = (d.q == t).values
        if tr.sum() < 3000: continue
        y = d.loc[tr, f"r{h}"].clip(-.3, .3); y = y - y.groupby(d.loc[tr, "q"]).transform("mean")
        m = Ridge(alpha=3.0).fit(X[tr.values], y); pred[te] = m.predict(X[te])
    d[f"tf{h}"] = pred
    r = []
    for qq, g in d.dropna(subset=[f"tf{h}", f"r{h}"]).groupby("q"):
        lo = g[g[f"tf{h}"] <= g[f"tf{h}"].quantile(.2)][f"r{h}"].mean(); hi = g[g[f"tf{h}"] >= g[f"tf{h}"].quantile(.8)][f"r{h}"].mean()
        r.append((qq, hi - lo, hi, lo))
    r = pd.DataFrame(r, columns=["q", "ls", "long", "short"]).set_index("q")
    for lab, rr in (("all", r), ("2020+", r[r.index >= pd.Period("2020Q1")])):
        print(f"tfidf h={h} {lab}: LS {rr.ls.mean()*100:.2f}% t {rr.ls.mean()/rr.ls.std()*np.sqrt(len(rr)):.2f} long {rr['long'].mean()*100:.2f}% short {rr.short.mean()*100:.2f}% n_q {len(rr)}")
