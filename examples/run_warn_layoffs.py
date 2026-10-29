"""WARN Act layoff notices (HF APProjects/us-warn-act-layoffs-notices-daily) matched by name to S&P 500 members.
notice_date is the employer's notice date; states publish with a lag, so entry is notice_date + LAG calendar days
(LAG 0 is look-ahead, shown only as an upper bound). Excess vs SPY. Usage: PYTHONPATH=. python examples/run_warn_layoffs.py WARN.csv COMPANIES.csv"""
import re, sys
import numpy as np, pandas as pd
from backtester.data.yahoo import load_bars

SUF = r"\b(inc|incorporated|corp|corporation|co|company|companies|llc|lp|ltd|plc|holdings|group|the|and|&)\b"
def norm(s): return re.sub(r"\s+", " ", re.sub(SUF, " ", re.sub(r"[^a-z0-9& ]", " ", str(s).lower()))).strip()
w = pd.read_csv(sys.argv[1], low_memory=False)
w["nd"] = pd.to_datetime(w.notice_date, errors="coerce")
w = w[(w.nd >= "2013-01-01") & (w.nd <= "2026-09-30") & (w.employees_affected >= 50)].copy()
c = pd.read_csv(sys.argv[2]); c["n"] = c.name.map(norm)
c = c[c.n.str.len() >= 4]
w["n"] = w.company_canonical.fillna(w.company).map(norm)
# match: warn name equals S&P name, or starts with it followed by a space (e.g. "amazon" -> "amazon fulfillment")
m = {}
for n in w.n.unique():
    for _, r in c.iterrows():
        if n == r.n or n.startswith(r.n + " "): m[n] = r.ticker; break
w["ticker"] = w.n.map(m); w = w.dropna(subset=["ticker"])
print(len(w), "matched notices;", w.ticker.nunique(), "tickers; sample:", w.company_canonical.drop_duplicates().head(12).tolist(), flush=True)
spy = load_bars("SPY", "2012-01-01", "2026-10-30")
H = (5, 20, 60)
for LAG in (0, 14):
    out = []
    for tk, g in w.groupby("ticker"):
        try: b = load_bars(tk.replace(".", "-"), "2012-01-01", "2026-10-30")
        except Exception: continue
        if b is None or len(b) < 300: continue
        last = None
        for _, r in g.sort_values("nd").iterrows():
            t0 = r.nd + pd.Timedelta(days=LAG)
            i = b.index.searchsorted(t0, side="right")
            if i < 1 or i + max(H) >= len(b) or (last is not None and i - last < 20): continue  # one event per 20 trading days
            last = i
            e = {"ticker": tk, "entry": b.index[i], "emp": r.employees_affected}
            for h in H:
                j = spy.index.searchsorted(b.index[i]); k = spy.index.searchsorted(b.index[i + h - 1])
                e[f"r{h}"] = b["close"].iloc[i + h - 1] / b["open"].iloc[i] - 1 - (spy["close"].iloc[k] / spy["open"].iloc[j] - 1)
            out.append(e)
    d = pd.DataFrame(out); d["m"] = d.entry.dt.to_period("M")
    for lab, x in (("all", d), ("emp>=300", d[d.emp >= 300])):
        for h in H:
            gm = x.groupby("m")[f"r{h}"].mean()
            print(f"LAG {LAG:2d} {lab:9s} h={h:2d} n={len(x):5d} mean {x[f'r{h}'].mean()*100:6.2f}% month-t {gm.mean()/gm.std()*np.sqrt(len(gm)):5.2f}")
