"""Form 144 (notice of proposed insider sale, HF ZipLime/insider-sale-notices 'notices' config) event study.
Entry = open of the first trading day after filed_date, exit close H days later, excess vs SPY.
Slices: 10b5-1 plan or not, size (% outstanding, $ value), officer/director, repeat sellers, cluster of insiders.
Usage: PYTHONPATH=. python examples/run_form144.py NOTICES_DIR OUT.parquet"""
import glob, sys
import numpy as np, pandas as pd
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch

H = (5, 20, 60)
n = pd.concat([pd.read_parquet(f) for f in glob.glob(sys.argv[1] + "/n*.parquet")])
n = n[(~n.is_amendment) & n.issuer_ticker.notna() & (n.aggregate_market_value_usd >= 2e6)].copy()
n["filed"] = pd.to_datetime(n.filed_date)
n["tk"] = n.issuer_ticker.str.replace(".", "-", regex=False)
print(len(n), n.tk.nunique(), "notices >= $2M", flush=True)
prefetch(list(n.tk.unique()) + ["SPY"], verbose=True)
spy = load_bars("SPY", "2022-06-01", "2026-10-30")
n = n.sort_values("filed")
n["cluster"] = n.groupby("issuer_cik").filed.transform(lambda s: s.map(lambda t: ((s > t - pd.Timedelta(days=10)) & (s <= t)).sum()))
rows = []
for tk, g in n.groupby("tk"):
    try:
        b = load_bars(tk, "2022-06-01", "2026-10-30")
    except Exception:
        continue
    if b is None or len(b) < 100: continue
    seen = set()
    for _, r in g.iterrows():
        i = b.index.searchsorted(r.filed, side="right")
        if i + max(H) >= len(b) or i < 21 or (r.owner_cik, i) in seen: continue
        seen.add((r.owner_cik, i))
        e = r.drop(["remarks", "signature", "plan_adoption_dates", "relationships"]).to_dict(); e["entry"] = b.index[i]
        e["prior20"] = b["close"].iloc[i - 1] / b["close"].iloc[i - 21] - 1
        for h in H:
            j = spy.index.searchsorted(b.index[i]); k = spy.index.searchsorted(b.index[i + h - 1])
            e[f"r{h}"] = b["close"].iloc[i + h - 1] / b["open"].iloc[i] - 1 - (spy["close"].iloc[k] / spy["open"].iloc[j] - 1)
        rows.append(e)
d = pd.DataFrame(rows); d.to_parquet(sys.argv[2]); print(len(d), "events")
d["half"] = np.where(d.entry < "2025-01-01", "2023-24", "2025+")
def show(name, m):
    for h in H:
        x = d.loc[m, f"r{h}"].dropna()
        print(f"{name:28s} h={h:2d} n={len(x):6d} mean {x.mean()*100:6.2f}% med {x.median()*100:6.2f}% t {x.mean()/x.std()*np.sqrt(len(x)):5.2f}")
show("all", d.index == d.index)
for nm, m in [("10b5-1 plan", d.has_10b5_1_plan), ("no plan", ~d.has_10b5_1_plan), ("officer", d.is_officer), ("director", d.is_director),
              ("10% owner", d.is_ten_percent_owner), ("pct>0.5%", d.pct_of_units_outstanding > .005), ("pct<0.05%", d.pct_of_units_outstanding < .0005),
              ("cluster>=3", d.cluster >= 3), ("value>=20M", d.aggregate_market_value_usd >= 2e7), ("run-up>20%", d.prior20 > .2), ("sold last 3m", d.past_3m_sale_count > 0),
              ("plan<=30d old", d.days_from_plan_to_notice <= 30), ("plan>180d old", d.days_from_plan_to_notice > 180)]:
    show(nm, m)
