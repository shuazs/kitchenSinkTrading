"""Indeed Hiring Lab US job-postings index by sector (raw.githubusercontent.com hiring-lab/job_postings_tracker) -> sector ETF rotation.
Signal each Friday: 28-day change of the sector's 'total postings' index, known with a 7-day publication lag, ranked across sectors;
hold next 4 weeks. Reports IC vs the mapped ETF's excess return over SPY and top-minus-bottom basket.
Usage: PYTHONPATH=. python examples/run_indeed_postings.py"""
import io
import numpy as np, pandas as pd, requests
from scipy import stats
from backtester.data.yahoo import load_bars

U = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv"
d = pd.read_csv(io.StringIO(requests.get(U, timeout=60).text), parse_dates=["date"])
d = d[d.variable == "total postings"].pivot(index="date", columns="display_name", values="indeed_job_postings_index")
MAP = {"Software Development": "IGV", "Banking & Finance": "XLF", "Insurance": "KIE", "Construction": "ITB", "Retail": "XRT",
       "Loading & Stocking": "XRT", "Hospitality & Tourism": "PEJ", "Medical Technician": "XLV", "Nursing": "XLV", "Pharmacy": "XLV",
       "Mining & Quarrying": "XME", "Energy & Utilities": "XLE", "Real Estate": "XLRE", "Manufacturing": "XLI", "Transport & Logistics": "IYT",
       "Food Preparation & Service": "XLY", "Entertainment & Events": "XLC", "Telecommunications": "XLC", "Legal": "XLI", "Education & Instruction": "XLY",
       "Marketing": "XLC", "Beauty & Wellness": "XLY", "Civil Engineering": "XLI", "Electrical Engineering": "XLI", "Production & Manufacturing": "XLI",
       "Information Design & Documentation": "XLK", "Banking & Finance ": "XLF", "Data & Analytics": "IGV", "Hardware": "XLK"}
cols = [c for c in d.columns if c in MAP]
print("sectors mapped:", len(cols), cols)
etfs = sorted(set(MAP[c] for c in cols))
px = pd.concat({e: load_bars(e, "2019-06-01", "2026-10-30")["close"] for e in etfs + ["SPY"]}, axis=1).dropna(subset=["SPY"])
sig = d[cols].rolling(7, min_periods=5).mean()
chg = np.log(sig / sig.shift(28))
rows = []
for t in chg.index[chg.index.dayofweek == 4]:
    t_use = t + pd.Timedelta(days=7)  # publication lag
    i = px.index.searchsorted(t_use, side="right")
    if i + 20 >= len(px) or chg.loc[t].isna().all(): continue
    for c in cols:
        e = MAP[c]
        if pd.isna(chg.loc[t, c]) or pd.isna(px[e].iloc[i]) or pd.isna(px[e].iloc[i + 20]): continue
        rows.append((t, c, e, chg.loc[t, c], px[e].iloc[i + 20] / px[e].iloc[i] - px["SPY"].iloc[i + 20] / px["SPY"].iloc[i]))
r = pd.DataFrame(rows, columns=["t", "sector", "etf", "chg", "ex"])
r = r[r.t.dt.dayofweek == 4]
r4 = r[r.t.map(lambda x: (x - r.t.min()).days // 7 % 4 == 0)]  # non-overlapping 4-week
ic = r4.groupby("t").apply(lambda g: stats.spearmanr(g.chg, g.ex)[0] if len(g) > 8 else np.nan).dropna()
ls = r4.groupby("t").apply(lambda g: g[g.chg >= g.chg.quantile(.8)].ex.mean() - g[g.chg <= g.chg.quantile(.2)].ex.mean() if len(g) > 8 else np.nan).dropna()
print(f"{len(ic)} non-overlapping 4wk periods | mean IC {ic.mean():.3f} t {ic.mean()/ic.std()*np.sqrt(len(ic)):.2f} | top-minus-bottom quintile {ls.mean()*100:.2f}%/4wk t {ls.mean()/ls.std()*np.sqrt(len(ls)):.2f}")
for lab, m in (("2020-22", ls.index < "2023-01-01"), ("2023+", ls.index >= "2023-01-01")):
    print(lab, f"{ls[m].mean()*100:.2f}% t {ls[m].mean()/ls[m].std()*np.sqrt(m.sum()):.2f}")
