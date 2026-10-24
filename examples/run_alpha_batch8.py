"""Alpha hunt batch 8: corporate actions, fund events and Kalshi combos.

Each test prints sub-periods and the Jul 2024+ holdout so the verdicts in
docs/guide/strategies.md can be re-checked (needs `SEC_USER_AGENT`):

- odd-lot tender offers (H1/H2): every company SC TO-I since 2008 whose offer
  gives odd lots (< 100 shares) priority over proration; buy 99 shares at the
  close after the filing if the price is under the offer (the top of a Dutch
  range), tender, and get the final price (from the last SC TO-I/A) or sell
  at expiry if the market is higher. Fund tenders are left out: most give odd
  lots no priority, so they are prorated;
- CEF activist 13D filings (H3): 252-day CAR of the fund vs all CEFs;
- CEF rights offerings (H4): short the fund vs the CEF basket for 25 days
  after its offer prospectus (497 / 424B*, EDGAR full-text search);
- MSCI factor ETFs around the May/Nov index reviews (H5);
- BTC around Deribit's monthly/quarterly expiry (H6, 1-minute BTC bars);
- Kalshi multi-leg combos (H8-K, KXMVE* series): buy No on combos priced
  5-40c (Rule B), at the prices of real taker-No trades vs at the prices
  parlay buyers paid (what the quoting market makers earn).
"""
import html
import io
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import yfinance as yf

from backtester import event_windows as ew
from backtester.data.cache import cache_dir

SEC = "https://www.sec.gov/Archives/"
HF = "https://huggingface.co/datasets/TrevorJS/kalshi-trades/resolve/main/"
SP = [("2008", "2016"), ("2017", "2024-06"), ("2024-07", "2026")]
FORMS = ("SC TO-I", "SC TO-I/A", "SC 13D")
ODD_YES = (r"(?:first|priority)[^.]{0,250}(?:fewer than 100|less than 100|not more than 99|odd lot)|odd lots?[^.]{0,300}"
           r"(?:will not be subject to proration|not subject to proration|without proration|without regard to proration|in full)")
ODD_NO = (r"no (?:special consideration or )?priority will be given[^.]{0,60}odd|odd lots?[^.]{0,80}"
          r"(?:subject to the same proration|will be subject to proration)|same proration terms|only tenders that are .?round lots")
MON = "(january|february|march|april|may|june|july|august|september|october|november|december)"


def ua() -> dict:
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def periods(x: pd.Series, splits=SP, scale=100, unit="%") -> str:
    return " | ".join(f"{a}-{b} {x[a:b].mean() * scale:6.2f}{unit} t{tstat(x[a:b]):.2f} n{len(x[a:b].dropna())}"
                      for a, b in splits)


def num(s: str) -> float:
    return float(s.replace(",", "").rstrip("."))


def text(path: str) -> str:
    r = requests.get(SEC + path, headers=ua(), timeout=60)
    time.sleep(0.12)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", r.text[:3_000_000]))).lower()


def form_index(y: int, q: int) -> pd.DataFrame:
    """SC TO-I, SC TO-I/A and SC 13D rows of one quarter's form.idx (streamed, cached)."""
    path = cache_dir("edgar_forms") / f"TO13D_{y}Q{q}.csv"
    if not path.exists():
        rows = []
        with requests.get(f"{SEC}edgar/full-index/{y}/QTR{q}/form.idx", headers=ua(), timeout=300, stream=True) as r:
            r.raise_for_status()
            for ln in r.iter_lines():
                ln = ln.decode("latin-1")
                for F in FORMS:
                    if ln.startswith(F + "  "):
                        m = re.match(r"(.*?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)", ln[len(F):].strip())
                        if m:
                            rows.append((F, m.group(1).strip(), int(m.group(2)), m.group(3), m.group(4)))
                        break
        pd.DataFrame(rows, columns=["form", "name", "cik", "date", "path"]).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"])


def filings() -> pd.DataFrame:
    today = pd.Timestamp.today()
    return pd.concat([form_index(y, q) for y in range(2008, today.year + 1) for q in range(1, 5)
                      if pd.Timestamp(y, 3 * q - 2, 1) + pd.offsets.QuarterEnd(0) < today])


def tickers() -> pd.DataFrame:
    j = requests.get("https://www.sec.gov/files/company_tickers_exchange.json", headers=ua(), timeout=60).json()
    return pd.DataFrame(j["data"], columns=j["fields"]).drop_duplicates("cik")


def cef_list(tk: pd.DataFrame) -> pd.DataFrame:
    """Listed funds with a Yahoo NAV symbol (X{ticker}X); cached."""
    path = cache_dir("alpha_batch8") / "cef_list.csv"
    if not path.exists():
        c = tk[tk.name.str.contains(r"FUND|INCOME|MUNI|TRUST|OPPORTUNIT|DIVIDEND|CREDIT|YIELD|CAPITAL|GLOBAL|EQUITY|STRATEG",
                                    case=False, na=False) & (tk.exchange != "OTC")]
        cand = [x for x in c.ticker.astype(str) if x.isalpha() and len(x) <= 4]
        ok = []
        for i in range(0, len(cand), 100):
            d = yf.download([f"X{x}X" for x in cand[i:i + 100]], start="2005-01-01", progress=False, auto_adjust=False)["Close"]
            ok += [col[1:-1] for col in d.columns if d[col].notna().sum() >= 250]
        c[c.ticker.isin(ok)][["cik", "name", "ticker"]].to_csv(path, index=False)
    return pd.read_csv(path)


def unsplit(px: pd.DataFrame) -> pd.DataFrame:
    """Yahoo closes are split-adjusted even with auto_adjust=False; undo it so
    prices compare with the dollar offer in the filing."""
    def splits(s):
        x = yf.Ticker(s).splits
        x.index = pd.DatetimeIndex(x.index).tz_localize(None) if len(x) else pd.DatetimeIndex([])
        return x
    sp = pd.DataFrame({s: splits(s) for s in px.columns}).reindex(px.index).fillna(0).replace(0, 1.0)
    return px * sp[::-1].cumprod()[::-1].shift(-1).fillna(1.0)


def parse_offer(t: str) -> dict:
    rng = re.search(r"(?:not|no) (?:greater|more) than \$\s?([\d,]+\.?\d*)\s?(?:nor|or|and not) (?:less|lower) than \$\s?([\d,]+\.?\d*)", t)
    rng2 = re.search(r"(?:price|prices) (?:of|between|ranging from) (?:not less than )?\$\s?([\d,]+\.?\d*) (?:and|to) \$\s?([\d,]+\.?\d*)", t)
    lo = hi = None
    if rng:
        hi, lo = num(rng.group(1)), num(rng.group(2))
    elif rng2:
        lo, hi = sorted([num(rng2.group(1)), num(rng2.group(2))])
    fx = re.search(r"(?:at|for) a (?:purchase )?price of \$\s?([\d,]+\.?\d*) per (?:share|common share)", t)
    ex = re.search(r"expire[sd]? at[^.]{0,80}?" + MON + r" (\d{1,2}), (\d{4})", t)
    return dict(dutch=bool(re.search(r"dutch auction", t)) or lo is not None, lo=lo, hi=hi,
                fixed=num(fx.group(1)) if fx else None, odd=bool(re.search(ODD_YES, t)) and not re.search(ODD_NO, t),
                other=bool(re.search(r"offer to (?:exchange|purchase)[^.]{0,80}(?:warrants|notes|debentures|preferred)", t[:20000])),
                cad=bool(re.search(r"canadian dollars|cdn\s?\$", t[:200000])),
                exp=" ".join(ex.groups()) if ex else None)


def final_price(paths) -> tuple:
    """(price in the final/preliminary results, terminated?) from SC TO-I/A texts."""
    clear, term = None, False
    for p in paths:
        u = text(p)
        if re.search(r"(?:has|have) (?:terminated|withdrawn) the (?:tender )?offer|offer (?:has been|was) (?:terminated|withdrawn)", u):
            term = True
        if re.search(r"final results|preliminary (?:count|results)|accepted for (?:purchase|payment)", u):
            m = re.findall(r"purchase price of \$\s?([\d,]+\.?\d*)", u) or re.findall(r"at a price of \$\s?([\d,]+\.?\d*) per share", u)
            if m:
                clear = num(m[-1])
    return clear, term


def odd_lot_tenders(f: pd.DataFrame, tk: pd.DataFrame, cefs: pd.DataFrame):
    path = cache_dir("alpha_batch8") / "tenders.jsonl"
    done = {json.loads(ln)["path"] for ln in open(path)} if path.exists() else set()
    sym = dict(zip(tk.cik, tk.ticker))
    to = f[(f.form == "SC TO-I") & f.cik.isin(sym) & ~f.cik.isin(cefs.cik)].drop_duplicates("path")
    am = f[f.form == "SC TO-I/A"]
    with open(path, "a") as fh:
        for r in to.itertuples():
            if r.path in done:
                continue
            t = text(r.path)
            if "odd lot" not in t and "odd-lot" not in t:
                fh.write(json.dumps(dict(path=r.path, odd=False)) + "\n")
                continue
            d = parse_offer(t)
            d.update(path=r.path, cik=int(r.cik), tk=sym[r.cik], date=str(r.date.date()), nav="net asset value" in t)
            if d["odd"] and not d["nav"]:
                a = am[(am.cik == r.cik) & (am.date > r.date) & (am.date <= r.date + pd.Timedelta(days=150))].sort_values("date")
                d["clear"], d["term"] = final_price(a.path)
            fh.write(json.dumps(d) + "\n")
    D = pd.DataFrame([json.loads(ln) for ln in open(path)])
    D = D[D.odd & ~D.nav.astype(bool) & ~D.other.astype(bool) & ~D.cad.astype(bool)
          & ~D.tk.astype(str).str.contains("-")].drop_duplicates(["tk", "date"])
    D["date"] = pd.to_datetime(D.date)
    D = D[D.date < pd.Timestamp.today() - pd.Timedelta(days=45)]
    px = unsplit(yf.download(sorted(D.tk.unique()), start="2007-12-01", auto_adjust=False, progress=False)["Close"])
    rows = []
    for r in D.itertuples():
        base = r.lo if r.dutch else r.fixed
        cap = (r.hi if r.dutch else r.fixed) if pd.notna(r.hi if r.dutch else r.fixed) else r.clear
        # the final price counts only when it sits inside the offer's range (else it is an aggregate $ amount)
        ok = pd.notna(r.clear) and ((r.dutch and pd.notna(r.lo) and 0.99 * r.lo <= r.clear <= 1.01 * r.hi)
                                    or (not r.dutch and pd.notna(r.fixed) and r.fixed <= r.clear <= 1.25 * r.fixed)
                                    or (pd.isna(base) and r.clear < 1e4))
        pay = r.clear if ok else base
        if r.tk not in px or pd.isna(cap) or pd.isna(pay):
            continue
        p = px[r.tk].dropna()
        i = p.index.searchsorted(r.date, side="right")
        if i >= len(p) - 2 or not 0 < cap / p.iloc[i] - 1 <= 0.3:
            continue
        exp = pd.to_datetime(r.exp, format="%B %d %Y", errors="coerce")
        if pd.isna(exp) or not 10 <= (exp - r.date).days <= 120:
            exp = r.date + pd.offsets.BDay(22)
        j = min(p.index.searchsorted(exp, side="right") - 1, len(p) - 1)
        if j <= i:
            continue
        out = p.iloc[j] if r.term else max(pay, p.iloc[j])
        rows.append(dict(date=r.date, tk=r.tk, dutch=r.dutch, entry=p.iloc[i], gross=out / p.iloc[i] - 1,
                         days=(p.index[j] - p.index[i]).days + 3))
    X = pd.DataFrame(rows).set_index("date").sort_index()
    for fee in (0, 25):
        ret = X.gross - (1 + fee) / (99 * X.entry)        # $1 commission, optional broker tender fee
        for lab, m in (("fixed", ~X.dutch), ("Dutch", X.dutch), ("all", X.dutch | ~X.dutch)):
            print(f"Odd-lot tenders, {lab}, ${fee} fee: {periods(ret[m])} | median {ret[m].median():.1%} | "
                  f"$ per event {(99 * X.entry * ret)[m].median():.0f}")
    print(f"  {len(X) / X.index.year.nunique():.1f} events/yr, median capital ${(99 * X.entry).median():.0f}, "
          f"hold {X.days.mean():.0f} days")


def cef_activists(f: pd.DataFrame, cefs: pd.DataFrame, tr: pd.DataFrame):
    sym = dict(zip(cefs.cik, cefs.ticker))
    e = f[(f.form == "SC 13D") & f.cik.isin(sym)].sort_values("date")
    r = tr.pct_change(fill_method=None).clip(-0.5, 0.5)
    bench = r.mean(axis=1)
    car, last = {}, {}
    for x in e.itertuples():
        s = sym[x.cik]
        if s not in tr or (s in last and (x.date - last[s]).days < 365):
            continue
        last[s] = x.date
        p = tr.index.searchsorted(x.date, side="right")
        rr = r[s].iloc[p + 1:p + 253]
        if len(rr) == 252 and rr.notna().sum() > 200:
            car[x.date] = (1 + rr.fillna(0)).prod() - (1 + bench.iloc[p + 1:p + 253]).prod()
    print(f"CEF activist 13D, 252d CAR vs all CEFs: {periods(pd.Series(car).sort_index())}")


def rights_offerings(cefs: pd.DataFrame, tr: pd.DataFrame):
    path = cache_dir("alpha_batch8") / "rights_fts.csv"
    if not path.exists():
        rows = []
        for y in range(2008, pd.Timestamp.today().year + 1):
            for frm in range(0, 1000, 100):
                q = {"q": '"rights offering" "subscription period" "net asset value"', "forms": "497,424B1,424B2,424B3,424B5",
                     "dateRange": "custom", "startdt": f"{y}-01-01", "enddt": f"{y}-12-31", "from": frm}
                h = requests.get("https://efts.sec.gov/LATEST/search-index", params=q, headers=ua(), timeout=60).json()["hits"]["hits"]
                time.sleep(0.15)
                rows += [(int(x["_source"]["ciks"][0]), x["_source"]["file_date"]) for x in h]
                if len(h) < 100:
                    break
        pd.DataFrame(rows, columns=["cik", "date"]).to_csv(path, index=False)
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    sym = dict(zip(cefs.cik, cefs.ticker))
    d = d[d.cik.isin(sym)]
    ev, last = [], {}
    for r in d.itertuples():
        s = sym[r.cik]
        if s in last and (r.date - last[s]).days < 120:
            continue
        last[s] = r.date
        ev.append((r.date, s))
    bench = tr.pct_change(fill_method=None).mean(axis=1)

    def short(H, shift=0, cost=0.0):
        out = {}
        for dt, s in ev:
            if s not in tr:
                continue
            p = tr[s].dropna()
            i = p.index.searchsorted(dt + pd.Timedelta(days=shift), side="right")
            if 1 <= i and i + H < len(p):
                b = (1 + bench.loc[p.index[i + 1]:p.index[i + H]]).prod() - 1
                out[dt] = -(p.iloc[i + H] / p.iloc[i] - 1 - b) - cost
        return pd.Series(out).sort_index()
    for H in (10, 25, 40):
        print(f"CEF rights offering, short fund vs CEFs {H}d, gross: {periods(short(H))}")
    print(f"  25d net of 0.5% spreads + 10%/yr borrow: {periods(short(25, cost=0.005 + 0.10 * 25 / 252))}")
    print(f"  25d net of 0.5% spreads + 3%/yr borrow: {periods(short(25, cost=0.005 + 0.03 * 25 / 252))}")
    for sh in (-365, -180):
        print(f"  placebo {sh} days: {periods(short(25, shift=sh))}")


def factor_etfs():
    c = yf.download("MTUM USMV QUAL VLUE SPY".split(), start="2013-01-01", auto_adjust=True, progress=False)["Close"]
    idx = c.SPY.dropna().index
    anch = [p for y in range(2013, 2027) for m in (5, 11)
            for p in [idx.searchsorted(pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0), side="right") - 1]
            if p > 0 and idx[p].month == m]
    pool = pd.concat([ew.event_returns(c[s].reindex(idx), anch, -6, 0, c.SPY.reindex(idx)) for s in ("MTUM", "USMV", "QUAL", "VLUE")])
    print(f"MSCI factor ETFs vs SPY, last 6 days before May/Nov review: {periods(pool.sort_index(), [('2013', '2018'), ('2019', '2024-06'), ('2024-07', '2026')], 1e4, 'bp')}")


def deribit_expiry():
    b = pd.read_parquet("/mnt/project-files/minute-data/BTC.parquet", columns=["close"]).close.resample("1h").last().ffill()
    b.index = b.index.tz_convert(None) if b.index.tz is not None else b.index
    rows = {}
    for y in range(2021, 2027):
        for m in range(1, 13):
            e = ew.last_friday(y, m) + pd.Timedelta(hours=8)
            if b.index[0] + pd.Timedelta(days=3) < e < b.index[-1] - pd.Timedelta(days=3):
                rows[e] = b.asof(e + pd.Timedelta(hours=24)) / b.asof(e) - 1
    print(f"BTC 24h after Deribit monthly expiry: {periods(pd.Series(rows), [('2021', '2022'), ('2023', '2024-06'), ('2024-07', '2026')], 1e4, 'bp')}")


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so pyarrow reads only
    the row groups and columns it needs from a remote parquet file."""

    def __init__(self, url):
        self.s = requests.Session()
        h = self.s.head(url, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = h.url, int(h.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for k in range(4):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if k == 3:
                    raise
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def kalshi_combo_data():
    """KXMVE* markets (settled) and their trades from the TrevorJS mirror, cached (~1 h first run)."""
    mp, tp = cache_dir("alpha_batch8") / "mve_markets.parquet", cache_dir("alpha_batch8") / "mve_trades.parquet"
    if not mp.exists():
        out = []
        for i in range(4):
            f = pq.ParquetFile(HTTPFile(f"{HF}markets-{i:04d}.parquet"))
            for g in range(f.metadata.num_row_groups):
                t = f.read_row_group(g, columns=["ticker", "volume"])
                m = pc.and_(pc.starts_with(t["ticker"], "KXMVE"), pc.greater(t["volume"], 0))
                if pc.any(m).as_py():
                    out.append(f.read_row_group(g, columns=["ticker", "title", "result", "volume"]).filter(m))
        pq.write_table(pa.concat_tables(out), mp)
    if not tp.exists():
        out, cut = [], pd.Timestamp("2025-09-17", tz="UTC")
        for i in range(16):
            f = pq.ParquetFile(HTTPFile(f"{HF}trades-{i:04d}.parquet"))
            ti = f.schema_arrow.get_field_index("created_time")
            for g in range(f.metadata.num_row_groups):
                if f.metadata.row_group(g).column(ti).statistics.max >= cut:
                    t = f.read_row_group(g, columns=["ticker", "count", "yes_price", "taker_side", "created_time"])
                    m = pc.starts_with(t["ticker"], "KXMVE")
                    if pc.any(m).as_py():
                        out.append(t.filter(m))
        pq.write_table(pa.concat_tables(out), tp)
    return pd.read_parquet(mp), pd.read_parquet(tp)


def kalshi_combos():
    m, tr = kalshi_combo_data()
    m = m[m.result.isin(["yes", "no"])]
    tr = tr.drop_duplicates().merge(m[["ticker", "result"]], on="ticker")
    tr["yes"] = tr.yes_price / 100
    # No buyer's P&L per contract at the trade's No price, after the 0.07 p(1-p) taker fee
    tr["net"] = (tr.result == "no") - (1 - tr.yes) - 0.07 * tr.yes * (1 - tr.yes)
    y = tr[tr.taker_side == "yes"].sort_values("created_time").groupby("ticker").first()
    cal = pd.cut(y.yes, [0, .05, .1, .2, .4, .6, 1])
    print("Kalshi combos, Yes price vs Yes rate:", y.groupby(cal, observed=True).apply(
        lambda g: f"{g.yes.mean():.3f}/{(g.result == 'yes').mean():.3f}").to_dict())
    b = tr[tr.yes.between(0.05, 0.40)]
    for side, lab in (("no", "real taker-No fills"), ("yes", "parlay buyers' prices (makers' side)")):
        f = b[b.taker_side == side].sort_values("created_time").groupby("ticker").first()  # one entry per combo
        f.index = pd.DatetimeIndex(f.created_time.dt.tz_convert(None))
        f = f.sort_index()
        print(f"Kalshi combos, buy No at 5-40c, {lab}: {periods(f.net, [('2025-09', '2025-11'), ('2025-12', '2026-01')], 100, 'c')}")


def main():
    f, tk = filings(), tickers()
    cefs = cef_list(tk)
    tr = yf.download(sorted(cefs.ticker), start="2005-01-01", auto_adjust=True, progress=False)["Close"].sort_index()
    odd_lot_tenders(f, tk, cefs)
    cef_activists(f, cefs, tr)
    rights_offerings(cefs, tr)
    factor_etfs()
    deribit_expiry()
    kalshi_combos()


if __name__ == "__main__":
    main()
