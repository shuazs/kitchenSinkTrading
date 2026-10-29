"""Alpha hunt batch 13 (2026-10-29): ideas outside commodities, pre-registered
in /mnt/project-files/commodity-futures/HYPOTHESES.md (rounds 5-6).

    python examples/run_alpha_batch13.py

E1-E3 macro release days (CPI, payrolls, PPI, retail sales, GDP advance and
jobless claims; HF ZipLime/us-economic-events, 2010-2026) on TLT/IEF/GLD/SPY;
S1 Halloween (Nov-Apr SPY, else BIL); S2 crypto weekends; S3 52-week-high
nearness on 40 ETFs. Design to Jun 2024, holdout Jul 2024+. All fail.
"""
import numpy as np
import pandas as pd
import requests

from backtester.data.cache import cache_dir
from backtester.data.fred import load_series
from backtester.data.yahoo import load_bars

HOLD = "2024-07-01"
RELEASES = "https://huggingface.co/datasets/ZipLime/us-economic-events/resolve/main/data/releases/part-00000.parquet"


def ev(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean() * 1e4:6.1f} bp t {x.mean() / x.std() * np.sqrt(len(x)):5.2f} n {len(x):4d}"


def sharpe(r: pd.Series) -> str:
    r = r.dropna()
    return f"{r.mean() * 252:6.1%}/yr Sharpe {r.mean() / r.std() * np.sqrt(252):5.2f}"


def sim(w: pd.DataFrame, cost: float = 4e-4) -> pd.Series:
    """Weights decided at the close, held open(t+1) -> open(t+2), cost per side."""
    o = pd.DataFrame({s: load_bars(s).open for s in w.columns}).reindex(w.index)
    held = w.fillna(0).shift(1).fillna(0)
    return (held * (o.shift(-1) / o - 1).fillna(0)).sum(axis=1) - held.diff().abs().fillna(held.abs()).sum(axis=1) * cost


def releases() -> pd.DataFrame:
    path = cache_dir("econ") / "releases.parquet"
    if not path.exists():
        r = requests.get(RELEASES, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    rel = pd.read_parquet(path)
    rel["d"] = pd.to_datetime(rel.release_date_et)
    return rel


def macro_days():
    print("E1-E3 macro release days (close-to-close), design | holdout")
    rel = releases()
    big = rel.release_family.isin(["CPI", "EMPLOYMENT", "PPI", "RETAIL_SALES"]) | (
        (rel.release_family == "GDP") & (rel.estimate_stage == "advance"))
    hid = pd.DatetimeIndex(rel[big].d.unique())
    claims = pd.DatetimeIndex(rel[rel.release_family == "JOBLESS_CLAIMS"].d.unique())
    for s in ["TLT", "IEF", "GLD", "SPY"]:
        r = np.log(load_bars(s).close).diff().loc["2010":]
        on = r.index.isin(hid)
        pre = pd.Series(on, index=r.index).shift(-1, fill_value=False).values
        print(f"  {s}: release days {ev(r[on][:'2024-06'])} | {ev(r[on][HOLD:])}; day before {ev(r[pre][:'2024-06'])}; "
              f"other days {ev(r[~on & ~pre][:'2024-06'])}; claims-only days {ev(r[r.index.isin(claims) & ~on][:'2024-06'])}")
        print("      " + " | ".join(f"{f} {ev(r[r.index.isin(rel[rel.release_family == f].d)][:'2024-06'])}"
                                   for f in ["CPI", "EMPLOYMENT", "PPI", "RETAIL_SALES"]))


def halloween():
    print("S1 Halloween: winter (Nov-Apr) minus summer excess return per year, then SPY/BIL switch")
    g = np.log(load_bars("^GSPC", start="1950-01-01").close).diff()
    tb = (load_series("DTB3") / 100 / 252).reindex(g.index).ffill()
    for a, b in [("1950", "2001"), ("2002", "2024-06")]:
        r = g.loc[a:b]
        win = r.index.month.isin([11, 12, 1, 2, 3, 4])
        yr = pd.DataFrame({"w": r.where(win, 0), "s": (r - tb.reindex(r.index)).where(~win, 0)}).groupby(r.index.year).sum()
        d = yr.w - yr.s
        print(f"  ^GSPC {a}-{b}: winter minus summer excess {d.mean():+.3f}/yr t {d.mean() / d.std() * np.sqrt(len(d)):.2f}")
    idx = load_bars("SPY").loc["2007-06":].index
    w = pd.Series(idx.month.isin([11, 12, 1, 2, 3, 4]).astype(float), index=idx)
    sw, spy = sim(pd.DataFrame({"SPY": w, "BIL": 1 - w})), sim(pd.DataFrame({"SPY": 1.0}, index=idx))
    print(f"  switch design {sharpe(sw[:'2024-06'])} vs SPY {sharpe(spy[:'2024-06'])} | holdout {sharpe(sw[HOLD:])} vs {sharpe(spy[HOLD:])}")


def crypto_weekend():
    print("S2 crypto weekend vs weekday daily returns")
    for s in ["BTC-USD", "ETH-USD"]:
        r = np.log(load_bars(s).close).diff().loc["2017":]
        we = r.index.dayofweek >= 5
        print(f"  {s} weekend {ev(r[we][:'2024-06'])} | {ev(r[we][HOLD:])}; weekday {ev(r[~we][:'2024-06'])} | {ev(r[~we][HOLD:])}")


def high_52w():
    print("S3 52-week-high nearness, top 5 of 40 ETFs monthly vs equal weight")
    u = ("XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC EWJ EWG EWU EWC EWA EWZ EWW EWY EWT FXI INDA EZA EWS "
         "TLT IEF LQD HYG TIP EMB GLD SLV USO DBA DBC VNQ IWM QQQ EEM EFA").split()
    p = pd.DataFrame({s: load_bars(s).close for s in u})
    me = p.index.to_series().groupby(p.index.to_period("M")).last().values
    near = (p / p.rolling(252).max()).loc[me]
    n = near.notna().sum(axis=1)
    top = (near.rank(axis=1, ascending=False) <= 5).astype(float).div(5).where(n >= 25).reindex(p.index).ffill()
    ew = near.notna().astype(float).div(n, axis=0).where(n >= 25).reindex(p.index).ffill()
    x = sim(top) - sim(ew)
    print(f"  top 5 minus EW: design {sharpe(x[:'2024-06'])} | holdout {sharpe(x[HOLD:])}")


if __name__ == "__main__":
    for f in (macro_days, halloween, crypto_weekend, high_52w):
        f()
