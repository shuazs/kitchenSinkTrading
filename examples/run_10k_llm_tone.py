"""LLM-scored 10-K tone (HF milosand1/sec-filings-tone, Claude Haiku topic polarity per 10-K) as a stock signal.
Entry open the trading day after filing_date, exit close H days later, excess vs SPY. Signals ranked within
filing year (cross-section of that year's filers only, no look-ahead; changes use the prior 10-K).
Usage: PYTHONPATH=. python examples/run_10k_llm_tone.py CELLS.parquet"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from backtester.data.yahoo import load_bars

H = (5, 20, 60)
c = pd.read_parquet(sys.argv[1])
c = c[c.version == "model"].copy()
c["filing_date"] = pd.to_datetime(c.filing_date)
g = c.groupby(["ticker", "fy", "filing_date"])
f = pd.DataFrame({"pol": g.polarity.mean(), "unc": g.uncertainty.mean(), "mag": g.magnitude.mean(),
                  "neg_share": g.polarity.apply(lambda s: (s < 0).mean() if s.notna().any() else np.nan),
                  "lm": g.lm_polarity.mean(), "fb": g.finbert_polarity.mean()}).reset_index()
piv = c.pivot_table(index=["ticker", "fy"], columns="topic", values="polarity").add_prefix("pol_").reset_index()
f = f.merge(piv, on=["ticker", "fy"]).sort_values(["ticker", "fy"])
for k in ["pol", "unc", "mag", "neg_share", "lm", "fb"] + [x for x in f if x.startswith("pol_")]:
    f["d_" + k] = f.groupby("ticker")[k].diff()
spy = load_bars("SPY", "2012-01-01", "2026-10-30")
rows = []
for tk, gg in f.groupby("ticker"):
    try:
        b = load_bars(tk.replace(".", "-"), "2012-01-01", "2026-10-30")
    except Exception:
        continue
    if b is None or len(b) < 300: continue
    for _, r in gg.iterrows():
        i = b.index.searchsorted(r.filing_date, side="right")  # first day after filing
        if i + max(H) >= len(b) or i < 1: continue
        e = r.to_dict(); e["entry"] = b.index[i]
        for h in H:
            j = spy.index.searchsorted(b.index[i]); k = spy.index.searchsorted(b.index[i + h - 1])
            e[f"r{h}"] = b["close"].iloc[i + h - 1] / b["open"].iloc[i] - 1 - (spy["close"].iloc[k] / spy["open"].iloc[j] - 1)
        rows.append(e)
d = pd.DataFrame(rows)
d.to_parquet("/tmp/ec/ev10k.parquet")
d["yr"] = d.entry.dt.year
sigs = [x for x in d if (x.startswith("d_") or x in ("pol", "unc", "mag", "neg_share", "lm", "fb") or x.startswith("pol_"))]
out = []
for s in sigs:
    for h in H:
        ic = d.groupby("yr").apply(lambda x: stats.spearmanr(x[s], x[f"r{h}"], nan_policy="omit")[0] if x[s].notna().sum() > 40 else np.nan).dropna()
        out.append((s, h, ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), (ic > 0).mean(), len(ic)))
o = pd.DataFrame(out, columns=["sig", "h", "ic", "t", "posyr", "n"]).sort_values("t")
print(len(d), "events;", len(sigs) * len(H), "tests")
print(o.head(10).to_string()); print(o.tail(10).to_string())
