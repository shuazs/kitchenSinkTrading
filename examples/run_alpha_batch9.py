"""Alpha hunt batch 9: dividend, IPO and split events (all failed).

Each test prints sub-periods and the Jul 2024+ holdout so the verdicts in
docs/guide/strategies.md can be re-checked (needs `SEC_USER_AGENT`):

- symbols from HF siddharthmb/stocks-dividends with 4+ ex-dates since 2005;
  Yahoo raw prices and Yahoo's own (split-consistent) dividends, fetched in
  chunks of 100; about 30% of the symbols have no Yahoo history (delisted);
- dividend run-up (H1): close at ex-6/-11/-21 to the close before the ex-date
  vs SPY, $5M+/day names, t over ex-dates (events on one date are pooled);
- ex-dividend capture for a tax-free account (H5): close before ex to the
  ex-date open plus the dividend, vs SPY overnight, by yield;
- dividend initiations and 25%+ raises (H4): ex-date+1 open to +60 days vs IWM;
- CEF ex-distribution overreaction (H2): discount widens > 1 pt on the
  ex-date, next 5 days vs all CEFs; CEF tax-loss January (H6): worst YTD
  decile with a wider discount, Dec 20 to Jan 31;
- IPO lockup expiry (H3): first 424B4 per CIK whose Yahoo history starts
  within 10 days, IPO + 180 days, CAR vs IWM before and after;
- forward splits (H7): 20/60-day CAR vs SPY of liquid dividend payers.
"""
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtester.data.cache import cache_dir
from run_alpha_batch8 import SEC, cef_list, periods, tickers, ua

DIV = "https://huggingface.co/datasets/siddharthmb/stocks-dividends/resolve/main/dividend.csv"
SP = [("2005", "2016"), ("2017", "2024-06"), ("2024-07", "2026")]


def bench() -> pd.DataFrame:
    return yf.download(["SPY", "IWM"], start="2004-06-01", auto_adjust=True, progress=False)


def div_events(b: pd.DataFrame) -> pd.DataFrame:
    path = cache_dir("alpha_batch9") / "div_events.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    src = cache_dir("hf_dividends") / "dividend.csv"
    if not src.exists():
        src.write_bytes(requests.get(DIV, timeout=300).content)
    d = pd.read_csv(src, parse_dates=["ex_date"])
    n = d[d.ex_date >= "2005-01-01"].groupby("act_symbol").size()
    syms = [s.replace(".", "-") for s in n[n >= 4].index if isinstance(s, str) and s.replace(".", "").isalpha()]
    spy, iwm, spyo = b["Close"]["SPY"], b["Close"]["IWM"], b["Open"]["SPY"]
    rows = []
    for k in range(0, len(syms), 100):
        x = yf.download(syms[k:k + 100], start="2004-06-01", auto_adjust=False, actions=True, progress=False)
        for s in syms[k:k + 100]:
            if s not in x["Close"]:
                continue
            c = x["Close"][s].dropna()
            if len(c) < 300:
                continue
            o, dv = x["Open"][s].reindex(c.index), x["Dividends"][s].reindex(c.index).fillna(0)
            dvol = (c * x["Volume"][s].reindex(c.index)).rolling(20).mean().shift(1)
            jump = c.pct_change().abs() > 0.4
            idx, prev = c.index, None

            def ex(a, z, bm):
                return c.iloc[z] / c.iloc[a] - 1 - (bm.asof(idx[z]) / bm.asof(idx[a]) - 1)
            for i, p in enumerate(np.flatnonzero(dv.to_numpy() > 0)):
                amt = dv.iloc[p]
                if 260 <= p < len(c) - 62 and not jump.iloc[p - 21:p + 2].any():
                    e = min(p + 61, len(c) - 1)
                    rows.append(dict(sym=s, date=idx[p], yld=amt / c.iloc[p - 1], px=c.iloc[p - 1], dvol=dvol.iloc[p],
                                     run5=ex(p - 6, p - 1, spy), run10=ex(p - 11, p - 1, spy), run20=ex(p - 21, p - 1, spy),
                                     cap=(o.iloc[p] + amt) / c.iloc[p - 1] - 1 - (spyo.asof(idx[p]) / spy.asof(idx[p - 1]) - 1),
                                     post60=c.iloc[e] / o.iloc[p + 1] - 1 - (iwm.asof(idx[e]) / iwm.asof(idx[p]) - 1),
                                     init=i == 0 and (idx[p] - idx[0]).days > 730,
                                     raised=prev is not None and amt >= 1.25 * prev))
                prev = amt
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def by_date(s: pd.Series) -> str:
    return periods(s.groupby(level=0).mean(), SP, 1e4, "bp")


def dividends(b):
    E = div_events(b).set_index("date").sort_index()
    E = E[E.px >= 3]
    L = E[E.dvol >= 5e6]
    for w in ("run5", "run10", "run20"):
        print(f"Dividend run-up {w} vs SPY ($5M+/day): {by_date(L[w])}")
    print(f"  run10 net 13 bp: {by_date(L.run10 - 0.0013)}")
    for lo, hi in ((0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 1)):
        x = L[(L.yld >= lo) & (L.yld < hi)]
        print(f"Ex-dividend capture, yield {lo:.1%}-{hi:.1%}, gross: {by_date(x.cap)}")
    print(f"  all liquid, net of a 10 bp spread: {by_date(L.cap - 0.001)}")
    print(f"Dividend initiations, 60d vs IWM: {periods(E.post60[E.init], SP)}")
    print(f"Dividend raises >= 25%, 60d vs IWM: {periods(E.post60[E.raised & ~E.init], SP)}")


def cef_events(cefs: pd.DataFrame):
    tk = sorted(cefs.ticker)
    tr = yf.download(tk, start="2005-01-01", auto_adjust=True, progress=False)["Close"].sort_index()
    x = yf.download(tk, start="2005-01-01", auto_adjust=False, actions=True, progress=False).reindex(tr.index)
    nav = yf.download([f"X{s}X" for s in tk], start="2005-01-01", auto_adjust=False, progress=False)["Close"]
    nav.columns = [c[1:-1] for c in nav.columns]
    disc = x["Close"] / nav.reindex(tr.index).where(nav > 0) - 1
    r = tr.pct_change(fill_method=None).clip(-0.5, 0.5)
    bm = r.mean(axis=1)
    rows = {}
    for s in tk:
        if s not in disc:
            continue
        for p in np.flatnonzero(x["Dividends"][s].fillna(0).to_numpy() > 0):
            ch = disc[s].iloc[p] - disc[s].iloc[p - 1] if p else np.nan
            if p + 6 < len(tr) and -0.2 < ch < -0.01:
                rows[(tr.index[p], s)] = (1 + r[s].iloc[p + 1:p + 6]).prod() - (1 + bm.iloc[p + 1:p + 6]).prod() - 0.003
    h = pd.Series(rows)
    h.index = h.index.get_level_values(0)
    print(f"CEF ex-date discount widened > 1 pt, 5d vs CEFs, net 30 bp: {periods(h.sort_index(), SP, 1e4, 'bp')}")
    ex = {}
    for y in range(2006, 2026):
        d0 = tr.index[tr.index.searchsorted(pd.Timestamp(y, 12, 20)) - 1]
        d1 = tr.index[min(tr.index.searchsorted(pd.Timestamp(y + 1, 1, 31), side="right") - 1, len(tr) - 1)]
        y0 = tr.index[tr.index.searchsorted(pd.Timestamp(y, 1, 1))]
        ytd, dw = tr.loc[d0] / tr.loc[y0] - 1, disc.loc[d0] - disc.loc[y0]
        ok = ytd.notna() & dw.notna() & tr.loc[d1].notna()
        pick = ytd[ok][(ytd[ok].rank(pct=True) <= 0.1) & (dw[ok] < 0)].index
        ret = tr.loc[d1] / tr.loc[d0] - 1
        ex[pd.Timestamp(y, 12, 20)] = ret[pick].mean() - ret[ok].mean() - 0.004
    ex = pd.Series(ex)
    print(f"CEF tax-loss January (worst YTD decile, wider discount): {periods(ex, SP)} | years positive {(ex > 0).mean():.0%}")


def ipo_lockups(b, tk: pd.DataFrame):
    path = cache_dir("edgar_forms") / "424B4.csv"
    if not path.exists():
        rows = []
        for y in range(2005, pd.Timestamp.today().year + 1):
            for q in range(1, 5):
                if pd.Timestamp(y, 3 * q - 2, 1) + pd.offsets.QuarterEnd(0) >= pd.Timestamp.today():
                    continue
                with requests.get(f"{SEC}edgar/full-index/{y}/QTR{q}/form.idx", headers=ua(), timeout=300, stream=True) as r:
                    for ln in r.iter_lines():
                        ln = ln.decode("latin-1")
                        if ln.startswith("424B4 "):
                            f = ln.split()
                            rows.append((" ".join(f[1:-3]), int(f[-3]), f[-2]))
        pd.DataFrame(rows, columns=["name", "cik", "date"]).to_csv(path, index=False)
    f = pd.read_csv(path, parse_dates=["date"]).sort_values("date").drop_duplicates("cik")
    tk = tk[tk.exchange.isin(["Nasdaq", "NYSE", "NYSE American", "NYSE Arca"])]
    m = f.merge(tk[["cik", "ticker"]], on="cik")
    m = m[~m.name.str.contains(r"ACQUISITION|SPAC|FUND|TRUST|BDC|PARTNERS LP|HOLDINGS CORP I", case=False)
          & m.ticker.str.match(r"^[A-Z]{1,5}$")]
    iwm = b["Close"]["IWM"]
    out = {k: {} for k in ("pre10", "post10", "post20")}
    for k in range(0, len(m), 100):
        px = yf.download(m.ticker.iloc[k:k + 100].tolist(), start="2004-06-01", auto_adjust=True, progress=False)["Close"]
        for r in m.iloc[k:k + 100].itertuples():
            c = px[r.ticker].dropna() if r.ticker in px else pd.Series(dtype=float)
            if len(c) < 150 or not pd.Timedelta(0) <= c.index[0] - r.date <= pd.Timedelta(days=10):
                continue                                   # history must start at the IPO
            p = c.index.searchsorted(c.index[0] + pd.Timedelta(days=180))
            if p + 21 >= len(c) or c.iloc[p] < 2:
                continue
            for key, a, z in (("pre10", p - 10, p), ("post10", p, p + 10), ("post20", p, p + 20)):
                out[key][(c.index[p], r.ticker)] = min(c.iloc[z] / c.iloc[a] - 1, 2) - (iwm.asof(c.index[z]) / iwm.asof(c.index[a]) - 1)
    for key, v in out.items():
        s = pd.Series(v)
        s.index = s.index.get_level_values(0)
        print(f"IPO lockup expiry (IPO+180d) {key} vs IWM: {periods(s.sort_index(), SP)}")


def splits(b):
    E = div_events(b)
    syms = sorted(E[E.dvol >= 5e6].sym.unique())
    spy = b["Close"]["SPY"]
    car = {20: {}, 60: {}}
    for k in range(0, len(syms), 100):
        x = yf.download(syms[k:k + 100], start="2004-06-01", auto_adjust=True, actions=True, progress=False)
        for s in syms[k:k + 100]:
            if s not in x["Close"]:
                continue
            c = x["Close"][s].dropna()
            for p in np.flatnonzero(x["Stock Splits"][s].reindex(c.index).fillna(0).to_numpy() > 1.2):
                if p + 61 < len(c):
                    for H in car:
                        car[H][(c.index[p], s)] = c.iloc[p + H] / c.iloc[p] - 1 - (spy.asof(c.index[p + H]) / spy.asof(c.index[p]) - 1)
    for H, v in car.items():
        s = pd.Series(v)
        s.index = s.index.get_level_values(0)
        print(f"Forward splits {H}d vs SPY (liquid dividend payers only): {periods(s.sort_index(), SP)}")


def main():
    b, tk = bench(), tickers()
    dividends(b)
    cef_events(cef_list(tk))
    ipo_lockups(b, tk)
    splits(b)


if __name__ == "__main__":
    main()
