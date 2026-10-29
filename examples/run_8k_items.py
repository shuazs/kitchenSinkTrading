"""8-K item-code event study for current S&P 500 members (EDGAR submissions JSON `items` field, 2013+).
Entry = open of the first trading day after the filing date, exit close H days later, excess vs SPY.
Usage: SEC_USER_AGENT="name email" PYTHONPATH=. python examples/run_8k_items.py COMPANIES.csv OUT.parquet"""
import os, sys, time
import numpy as np, pandas as pd, requests
from backtester.data.yahoo import load_bars

UA = {"User-Agent": os.environ["SEC_USER_AGENT"]}
H = (5, 20, 60)
comp = pd.read_csv(sys.argv[1])

def get(u):
    for _ in range(4):
        r = requests.get(u, headers=UA, timeout=30)
        if r.status_code == 200: return r.json()
        time.sleep(2)
    return None

rows = []
for _, c in comp.iterrows():
    j = get(f"https://data.sec.gov/submissions/CIK{int(c.cik):010d}.json")
    if not j: continue
    blocks = [j["filings"]["recent"]]
    for f in j["filings"]["files"]:
        if f["filingTo"] >= "2013-01-01":
            b = get("https://data.sec.gov/submissions/" + f["name"])
            if b: blocks.append(b)
    for b in blocks:
        for form, fd, acc, items in zip(b["form"], b["filingDate"], b.get("acceptanceDateTime", [""] * len(b["form"])), b["items"]):
            if form == "8-K" and fd >= "2013-01-01":
                rows.append((c.ticker, fd, acc, items))
    time.sleep(0.15)
ev = pd.DataFrame(rows, columns=["ticker", "fdate", "acc", "items"])
ev.to_parquet(sys.argv[2].replace(".parquet", "_raw.parquet"))
print(len(ev), "8-Ks", flush=True)
spy = load_bars("SPY", "2012-01-01", "2026-10-30")
out = []
for tk, g in ev.groupby("ticker"):
    try: b = load_bars(tk.replace(".", "-"), "2012-01-01", "2026-10-30")
    except Exception: continue
    if b is None or len(b) < 300: continue
    for _, r in g.iterrows():
        i = b.index.searchsorted(pd.Timestamp(r.fdate), side="right")
        if i < 1 or i + max(H) >= len(b): continue
        e = {"ticker": tk, "fdate": r.fdate, "items": r["items"], "entry": b.index[i]}
        for h in H:
            j = spy.index.searchsorted(b.index[i]); k = spy.index.searchsorted(b.index[i + h - 1])
            e[f"r{h}"] = b["close"].iloc[i + h - 1] / b["open"].iloc[i] - 1 - (spy["close"].iloc[k] / spy["open"].iloc[j] - 1)
        out.append(e)
d = pd.DataFrame(out); d.to_parquet(sys.argv[2])
d["m"] = d.entry.dt.to_period("M"); d["half"] = np.where(d.entry < "2020-01-01", "2013-19", "2020+")
codes = sorted({c for s in d["items"] for c in s.split(",") if c})
for code in codes:
    m = d["items"].str.split(",").map(lambda l: code in l)
    if m.sum() < 150: continue
    for h in H:
        x = d[m]; gm = x.groupby("m")[f"r{h}"].mean()
        print(f"item {code:5s} h={h:2d} n={m.sum():6d} mean {x[f'r{h}'].mean()*100:6.2f}% month-t {gm.mean()/gm.std()*np.sqrt(len(gm)):5.2f} | {x[x.half=='2013-19'][f'r{h}'].mean()*100:6.2f}% / {x[x.half=='2020+'][f'r{h}'].mean()*100:6.2f}%")
