"""Alpha hunt batch 5: supply flows, positioning and Kalshi favourites.

Each test prints in-sample sub-periods and the Jul 2024+ holdout (Kalshi:
events closing before / from Jul 2025) so the verdicts in
docs/guide/strategies.md can be re-checked:

- IG issuance rate-lock: daily count of S&P 500 issuers filing an FWP
  (EDGAR form.idx, streamed one quarter at a time; issuers that file >15 a
  quarter, i.e. structured-note shelves, dropped). 5-day count z > 1.5 ->
  TLT/IEF minus BIL over the next 3 days, split by overlap with the auction
  window;
- vol-control deleveraging: SPY 5 days after RV10/RV60 > 1.5 and rising;
- MBS convexity: TLT 5 days after a 10-day 10y yield move of 25/35/50 bp;
- auction window on XLU/VNQ/ITB and its correlation with the TLT window;
- CTA trend flip: 200d filter plus "out when E-mini leveraged-fund net long
  is in its top decile and SPY < 50d" (COT, placed by publication date);
- Kalshi tennis favourites (Yes at 80-95c) and EFL Championship No at
  50-65c, taker net after fees, event-level t.
"""

import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtester import auction_cycle as ac
from backtester.data.cache import cache_dir
from backtester.data.cot import load_pit
from backtester.data.fred import load_series
from backtester.data.sp500 import membership

SPLITS = [("2008", "2016"), ("2017", "2024-06"), ("2024-07", "2026")]
EV = "/mnt/project-files/event-markets/"


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def thin(mask: pd.Series, gap: int) -> list[int]:
    """Positions where `mask` is True, at most one per `gap` days."""
    keep, last = [], -gap
    for i in np.flatnonzero(mask.to_numpy()):
        if i - last >= gap:
            keep.append(i)
            last = i
    return keep


def split_line(ev: pd.Series, splits=SPLITS) -> str:
    return " | ".join(f"{lo}-{hi} {ev[lo:hi].mean() * 1e4:6.1f}bp n{len(ev[lo:hi])} t{tstat(ev[lo:hi]):.2f}"
                      for lo, hi in splits)


def auction_dates():
    a = pd.read_parquet("/mnt/project-files/alpha-hunt/treasury_auctions.parquet")
    return a[a.type.isin(["Note", "Bond"]) & a.term.str.match(r"(10-Year|9-Year|30-Year|29-Year)")].date.unique()


def ua() -> dict:
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}  # see docs/guide/data.md


def sp500_ciks() -> set[int]:
    tick = set().union(*membership()["2007":].values)
    j = requests.get("https://www.sec.gov/files/company_tickers.json", headers=ua(), timeout=60).json()
    cik = {v["ticker"].replace("-", "."): v["cik_str"] for v in j.values()}
    return {cik[t] for t in tick if t in cik}


def form_counts(y: int, q: int, ciks: set[int]) -> pd.DataFrame:
    """(form, date, cik, n) for FWP/424B2/424B5 filings by `ciks` in one
    quarter; streamed (the index is ~50 MB) and only the counts cached."""
    path = cache_dir("edgar_forms") / f"{y}Q{q}.csv"
    if not path.exists():
        url = f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        rows: dict = {}
        with requests.get(url, headers=ua(), timeout=300, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                f = line.split()
                if len(f) > 4 and f[0] in ("FWP", "424B2", "424B5") and f[-3].isdigit() and int(f[-3]) in ciks:
                    k = (f[0], f[-2], int(f[-3]))
                    rows[k] = rows.get(k, 0) + 1
        pd.DataFrame([(*k, n) for k, n in rows.items()], columns=["form", "date", "cik", "n"]).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"]).assign(y=y, q=q)


def ig_issuance(px):
    ciks = sp500_ciks()
    f = pd.concat([form_counts(y, q, ciks) for y in range(2008, 2027) for q in range(1, 5)
                   if pd.Timestamp(y, 3 * q - 2, 1) < pd.Timestamp.today()])
    bad = f.groupby(["cik", "y", "q"]).n.sum().groupby("cik").median().loc[lambda s: s > 15].index
    deals = f[~f.cik.isin(bad) & (f.form == "FWP")].groupby("date").cik.nunique()
    idx = px["TLT"].dropna().index
    pos = idx.searchsorted(deals.index).clip(max=len(idx) - 1)  # weekend filings count on the next session
    d = deals.groupby(idx[pos]).sum().reindex(idx, fill_value=0)[deals.index.min():]
    s5 = d.rolling(5).sum()
    z = (s5 - s5.rolling(250).mean()) / s5.rolling(250).std()
    print(f"IG issuance: {len(bad)} shelf issuers dropped, {d.mean():.2f} deals/day")
    for etf in ("TLT", "IEF"):
        c = px[etf].dropna()
        bil = px["BIL"].reindex(c.index).ffill()
        fwd = (c.shift(-4) / c.shift(-1) - 1) - (bil.shift(-4) / bil.shift(-1) - 1)  # filings seen after close t
        aw = ac.window_mask(c.index, ac.cycles(auction_dates(), c.index), 5)
        overlap = aw.shift(-2, fill_value=False) | aw.shift(-3, fill_value=False) | aw.shift(-4, fill_value=False)
        for thr in (1.0, 1.5, 2.0):
            ev = fwd.iloc[thin(z.reindex(c.index) > thr, 3)].dropna()
            print(f"  {etf} z>{thr}: {split_line(ev)} | uncond {fwd.mean() * 1e4:.1f}bp")
            if thr == 1.5:
                o = overlap.reindex(ev.index)
                print(f"    overlaps auction window {ev[o].mean() * 1e4:.1f}bp n{o.sum()}, "
                      f"outside {ev[~o].mean() * 1e4:.1f}bp n{(~o).sum()} (t {tstat(ev[~o]):.2f})")


def vol_control(px):
    spy = px["SPY"].dropna()
    r = spy.pct_change()
    ratio = r.rolling(10).std() / r.rolling(60).std()
    fwd5 = spy.shift(-6) / spy.shift(-1) - 1
    ev = fwd5.iloc[thin((ratio > 1.5) & (ratio > ratio.shift(1)), 5)].dropna()
    print(f"Vol-control: {split_line(ev)} | uncond {fwd5.mean() * 1e4:.1f}bp")


def mbs_convexity(px):
    y = load_series("DGS10").dropna()
    y.index = pd.to_datetime(y.index)
    tlt = px["TLT"].dropna()
    dy = (y - y.shift(10)).reindex(tlt.index).ffill() * 100
    f5 = tlt.shift(-6) / tlt.shift(-1) - 1  # FRED posts the next day, so enter at close t+1
    for thr in (25, 35, 50):
        for side, cond in (("rise", dy > thr), ("fall", dy < -thr)):
            print(f"MBS {thr}bp {side}: {split_line(f5.iloc[thin(cond, 10)].dropna())}")


def auction_other_assets(px):
    d = auction_dates()
    c = px[["TLT", "XLU", "VNQ", "ITB"]].dropna()
    cyc = ac.cycles(d, c.index)
    w = pd.DataFrame({s: ac.window_returns(c[s], cyc, 5, px["BIL"].reindex(c.index).ffill()) for s in c})
    for s in w:
        print(f"Auction window {s}: {split_line(w[s])}")
    print("  correlation with TLT window:", w.corr()["TLT"].round(2).to_dict())


def cta_flip(px):
    spy = px["SPY"].dropna()
    bil = px["BIL"].reindex(spy.index).ffill()
    c = load_pit()
    c = c[(c.entity_id == "13874A") & c.net_leveraged_funds.notna()].copy()
    c["kd"] = c.knowledge_date.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    c = c.sort_values("kd").drop_duplicates("kd", keep="last")
    x = c.net_leveraged_funds / c.open_interest
    c["pct"] = x.expanding(104).apply(lambda s: (s <= s.iloc[-1]).mean())
    pos = spy.index.searchsorted(c.kd, side="right")  # usable from the session after publication
    pct = pd.Series(c.pct.to_numpy()[pos < len(spy)], index=spy.index[pos[pos < len(spy)]])
    pct = pct[~pct.index.duplicated(keep="last")].reindex(spy.index).ffill()
    r, rb = spy.pct_change(), bil.pct_change().fillna(0)
    s50, s200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    start = pct.first_valid_index()

    def run(on: pd.Series, name: str):
        on = on.astype(float)
        tr = on.diff().abs().fillna(0)
        ret = (on.shift(1) * r + (1 - on.shift(1)) * rb - tr.shift(1) * 5e-4)[start:].dropna()
        parts = []
        for lo, hi in (("2010", "2017"), ("2018", "2024-06"), ("2024-07", "2026")):
            x = ret[lo:hi]
            parts.append(f"{lo}-{hi} Sh {x.mean() / x.std() * np.sqrt(252):.2f}")
        print(f"CTA {name:26s} in-mkt {on[start:].mean():.2f} | " + " | ".join(parts))

    run(pd.Series(True, index=spy.index), "SPY")
    run(spy > s200, "200d filter")
    for th in (0.8, 0.9, 0.95):
        run((spy > s200) & ~((pct >= th) & (spy < s50)), f"200d + COT top{int(th * 100)} & <50d")
    run((spy > s200) & ~((pct <= 0.1) & (spy < s50)), "200d + COT bottom10 (placebo)")


def kalshi(name, series, side, buckets):
    e = pd.read_parquet(EV + "ev_agg.parquet")
    close = pd.read_parquet(EV + "nf_markets.parquet", columns=["event_ticker", "close_time"]).groupby(
        "event_ticker").close_time.max()
    x = e[e.series.isin(series) & (e.taker_side == side) & e.bucket.isin(buckets)].copy()
    x["hold"] = x.event_ticker.map(close).dt.tz_localize(None) >= pd.Timestamp("2025-07-01")
    for h, g in x.groupby("hold"):
        v = g.groupby("event_ticker").agg(c=("cw", "sum"), n=("nw", "sum"))
        p = v.n / v.c
        print(f"Kalshi {name} {'holdout' if h else 'fit    '}: {100 * v.n.sum() / v.c.sum():+.2f}c/contract, "
              f"event mean {100 * p.mean():+.2f}c t {tstat(p):.2f}, {len(v)} events")


def main():
    px = yf.download("SPY TLT IEF BIL XLU VNQ ITB".split(), start="1999-01-01", auto_adjust=True,
                     progress=False)["Close"]
    ig_issuance(px)
    vol_control(px)
    mbs_convexity(px)
    auction_other_assets(px)
    cta_flip(px)
    fav = ["[0.8, 0.9)", "[0.9, 0.95)"]
    kalshi("ATP Yes 80-95c", ["KXATPMATCH", "KXATPCHALLENGERMATCH"], "yes", fav)
    kalshi("WTA Yes 80-95c (control)", ["KXWTAMATCH"], "yes", fav)
    kalshi("EFL No 50-65c", ["KXEFLCHAMPIONSHIPGAME"], "no", ["[0.5, 0.65)"])
    kalshi("EPL No 50-65c (control)", ["KXEPLGAME"], "no", ["[0.5, 0.65)"])


if __name__ == "__main__":
    main()
