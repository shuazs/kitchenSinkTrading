"""Hacker News attention (daily keyword counts, /mnt/project-files/nonfin-data/hn_daily.parquet) -> ticker and SPY returns.
Signal at day t close = z-score of 3-day mention share vs trailing 90 days; trade open t+1, hold H days. Excess vs SPY.
Usage: PYTHONPATH=. python examples/run_hn_attention.py"""
import numpy as np, pandas as pd
from backtester.data.yahoo import load_bars

hn = pd.read_parquet("/mnt/project-files/nonfin-data/hn_daily.parquet")
T = {"apple": "AAPL", "google": "GOOGL", "msft": "MSFT", "amzn": "AMZN", "meta": "META", "nvda": "NVDA", "tsla": "TSLA"}
spy = load_bars("SPY", "2010-01-01", "2026-10-30")
def z(s, w=90):
    m = s.rolling(w).mean().shift(1); sd = s.rolling(w).std().shift(1)
    return (s - m) / sd
share = lambda c: hn[c].rolling(3).sum() / hn["n"].rolling(3).sum()
rows = []
for k, tk in T.items():
    b = load_bars(tk, "2010-01-01", "2026-10-30")
    s = z(share(k)).reindex(b.index, method="ffill")  # HN day t known at t close
    for H in (1, 5, 20):
        o = b["open"].shift(-1); c = b["close"].shift(-H)
        ex = (c / o - 1) - (spy["close"].reindex(b.index).shift(-H) / spy["open"].reindex(b.index).shift(-1) - 1)
        df = pd.DataFrame({"z": s, "ex": ex, "y": b.index.year}).dropna()
        df = df[df.index >= "2012-01-01"]
        hi = df[df.z > 2]
        hi = hi[~hi.index.to_series().diff().dt.days.fillna(99).lt(H)] if False else hi
        rows.append((tk, H, len(hi), hi.ex.mean(), df.ex.mean(), np.corrcoef(df.z, df.ex)[0, 1]))
r = pd.DataFrame(rows, columns=["tk", "H", "n_z>2", "ex_after_spike", "ex_base", "corr"])
print(r.assign(ex_after_spike=lambda x: (x.ex_after_spike * 100).round(2), ex_base=lambda x: (x.ex_base * 100).round(2), corr=lambda x: x["corr"].round(3)).to_string())
print("pooled H=5 spike-minus-base: %.2f%%" % ((r[r.H == 5].ex_after_spike - r[r.H == 5].ex_base).mean() * 100))
# market-level: layoff / recession / ai attention -> SPY next 5d/20d
for k in ("layoff", "recession", "ai"):
    s = z(share(k), 90).reindex(spy.index, method="ffill")
    for H in (5, 20):
        f = (spy["close"].shift(-H) / spy["open"].shift(-1) - 1)
        df = pd.DataFrame({"z": s, "f": f}).dropna(); df = df[df.index >= "2013-01-01"]
        hi = df[df.z > 1.5]
        print(f"SPY after {k} spike H={H}: n={len(hi)} mean {hi.f.mean()*100:.2f}% vs all {df.f.mean()*100:.2f}% corr {np.corrcoef(df.z, df.f)[0,1]:.3f}")
