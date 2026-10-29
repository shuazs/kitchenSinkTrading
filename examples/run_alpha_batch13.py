"""Alpha hunt batch 13 (2026-10-29): ideas outside commodities, pre-registered
in /mnt/project-files/commodity-futures/HYPOTHESES.md (rounds 5-6).

    python examples/run_alpha_batch13.py

E1-E3 macro release days (CPI, payrolls, PPI, retail sales, GDP advance and
jobless claims; HF ZipLime/us-economic-events, 2010-2026) on TLT/IEF/GLD/SPY;
S1 Halloween (Nov-Apr SPY, else BIL); S2 crypto weekends; S3 52-week-high
nearness on 40 ETFs; P1-P3 pre-report run-up reversal outside natural gas;
K1 drift after 8-K non-reliance restatements; A1 Treasury auction strength. Design to Jun 2024, holdout Jul 2024+. All fail.
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


def pre_report_reversal():
    print("P1-P3 short on report day only after an up prior day (the storage-day filter elsewhere)")
    rel = releases()
    idx = load_bars("SPY").index
    sets = [("USO", idx[idx.dayofweek == 2], "EIA petroleum Wed"), ("UGA", idx[idx.dayofweek == 2], "EIA petroleum Wed")]
    sets += [(s, pd.DatetimeIndex(rel[rel.release_family == f].d.unique()), f) for f in ["CPI", "EMPLOYMENT"] for s in ["TLT", "SPY", "GLD"]]
    for sym, days, label in sets:
        r = np.log(load_bars(sym).close).diff()
        d = pd.DatetimeIndex(sorted(days)).intersection(r.index)
        up = -r.reindex(d)[r.shift(1).reindex(d) > 0]
        print(f"  {label:18s} {sym}: {ev(up[:'2024-06'])} | {ev(up[HOLD:])}")


SEC8K = "https://huggingface.co/datasets/ZipLime/sec-8k-events/resolve/main/data/{}/part-00000.parquet"


def restatements():
    """K1: SEC 8-K Item 4.02 non-reliance filings (HF ZipLime/sec-8k-events, the second the SEC
    accepted them). Survivor-only (today's CIK -> ticker map), which biases against a negative drift.
    First run downloads ~1,100 tickers from Yahoo (~10 min)."""
    import yfinance as yf

    print("K1 8-K 4.02 restatements: stock minus SPY/IWM from the first open after acceptance")
    d = cache_dir("sec8k")
    for name in ("restatement_events", "entity_map"):
        if not (d / f"{name}.parquet").exists():
            (d / f"{name}.parquet").write_bytes(requests.get(SEC8K.format(name), timeout=300).content)
    m = pd.read_parquet(d / "entity_map.parquet")
    m = m[m.ticker.notna() & m.exchange.isin(["NYSE", "Nasdaq"])].drop_duplicates("cik")
    ev_ = pd.read_parquet(d / "restatement_events.parquet")
    ev_ = ev_[~ev_.is_amendment.astype(bool)].merge(m[["cik", "ticker"]], on="cik")
    ev_["ticker"] = ev_.ticker.str.replace(".", "-", regex=False)
    pxd = cache_dir("sec8k/px")
    todo = [t for t in sorted(set(ev_.ticker)) if not (pxd / f"{t}.parquet").exists()]
    for i in range(0, len(todo), 60):
        ch = todo[i:i + 60]
        raw = yf.download(ch, start="2004-01-01", auto_adjust=True, progress=False, group_by="ticker")
        for t in ch:
            try:
                x = raw[t][["Open", "Close", "Volume"]].dropna()
            except KeyError:
                x = pd.DataFrame(columns=["Open", "Close", "Volume"])
            x.columns = ["open", "close", "volume"]
            x.to_parquet(pxd / f"{t}.parquet")
    bm = {s: load_bars(s, start="2004-01-01") for s in ["SPY", "IWM"]}

    def table(shift_days: int) -> pd.DataFrame:
        rows, last = [], {}
        for _, e in ev_.sort_values("knowledge_date").iterrows():
            p = pd.read_parquet(pxd / f"{e.ticker}.parquet")
            if len(p) < 300:
                continue
            p.index = pd.to_datetime(p.index).tz_localize(None)
            kd = pd.Timestamp(e.knowledge_date).tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(days=shift_days)
            i = p.index.searchsorted(kd.normalize() if (kd.hour, kd.minute) < (9, 30) else kd.normalize() + pd.Timedelta(days=1))
            if i < 21 or i + 61 >= len(p) or (e.ticker in last and (p.index[i] - last[e.ticker]).days < 120):
                continue
            last[e.ticker] = p.index[i]
            pre = p.iloc[i - 21:i - 1]
            row = {"date": p.index[i], "px": pre.close.iloc[-1], "dv": (pre.close * pre.volume).mean()}
            for b, s in bm.items():
                si = s.index.searchsorted(p.index[i])
                if si + 60 < len(s):
                    for h in (20, 60):
                        row[f"{b}{h}"] = np.log(p.close.iloc[i + h - 1] / p.open.iloc[i]) - np.log(s.close.iloc[si + h - 1] / s.open.iloc[si])
            rows.append(row)
        x = pd.DataFrame(rows)
        return x[(x.px > 5) & (x.dv > 2e6)]

    def pct(x):
        x = x.dropna()
        return f"{x.mean():+.1%} t {x.mean() / x.std() * np.sqrt(len(x)):5.2f} n {len(x)}"

    for name, shift in [("events", 0), ("placebo 1y earlier", -365)]:
        x = table(shift)
        des, hold = x[(x.date.dt.year != 2021) & (x.date < HOLD)], x[x.date >= HOLD]
        print(f"  {name}: " + " | ".join(f"{c} {pct(des[c])} hold {pct(hold[c])}" for c in ["SPY20", "SPY60", "IWM20"]))


def auction_strength():
    """A1: 10y/30y bid-to-cover vs the trailing 6 same-tenor auctions -> TLT after the day-0 reaction."""
    from backtester.data.treasury_auctions import LONG_TERMS, load_auctions

    print("A1 Treasury auction strength (|z| > 1) -> TLT from next open, net 6 bp")
    a = load_auctions()
    a = a[a.term.astype(str).str.contains(LONG_TERMS)].dropna(subset=["bid_to_cover"]).sort_values("date")
    a["grp"] = a.term.astype(str).str.extract(r"(\d+)-Year")[0].astype(int).map(lambda y: 10 if y <= 10 else 30)
    a["z"] = a.groupby("grp").bid_to_cover.transform(lambda s: (s - s.rolling(6).mean().shift(1)) / s.rolling(6).std().shift(1))
    t = load_bars("TLT", start="2008-01-01")
    rows = []
    for _, e in a[a.z.abs() > 1].iterrows():
        if e.date in t.index and t.index.get_loc(e.date) + 6 < len(t):
            i = t.index.get_loc(e.date)
            rows.append({"date": e.date, "d1": np.sign(e.z) * np.log(t.close.iloc[i + 1] / t.open.iloc[i + 1]) - 6e-4,
                         "d5": np.sign(e.z) * np.log(t.close.iloc[i + 5] / t.open.iloc[i + 1]) - 6e-4})
    x = pd.DataFrame(rows).set_index("date")
    for h in ("d1", "d5"):
        print(f"  {h}: design {ev(x[h][:'2024-06'])} | holdout {ev(x[h][HOLD:])}")


if __name__ == "__main__":
    for f in (macro_days, halloween, crypto_weekend, high_52w, pre_report_reversal, restatements, auction_strength):
        f()
