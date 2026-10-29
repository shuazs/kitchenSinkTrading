"""Unusual non-financial data as trading signals (thread "Find unusual
non-financial data edges", 2026-10-29). Pre-registrations and outputs:
/mnt/project-files/nonfin-data/ (batch1-3 prereg.md, results.md).

Every test prints in-sample (< Jul 2024) and holdout numbers plus three
in-sample sub-periods, after 5 bp/side costs (10 bp crypto), signal known
before the entry open. Data loaders are in `backtester/data/alt_data.py`.

Batch 1: Wikipedia top-100 attention spikes (S&P names, crypto, fear
articles), football losses -> country ETFs, NFL bettor-friendly weekends ->
sportsbooks, Ramadan, lunar phase, DST Mondays.
Batch 2: Wikipedia top-100 as an event feed: all listed firms, new wars ->
ITA / USO, outbreaks -> vaccine stocks, air crashes -> BA / JETS.
Batch 3: Indeed sector postings, WARN layoff notices, Hacker News crypto
attention, M6.5+ earthquakes -> country ETFs; with --heavy also daily
Wikipedia views for 3,711 tickers, CFPB complaints and lobbying jumps (SOVAI
sets end 2024-05, so a late 2022+ test replaces the holdout; ~1 h of Yahoo).

    python examples/run_nonfin_data.py [--heavy]    (needs SEC_USER_AGENT)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtester.data import alt_data as ad  # noqa: E402
from backtester.data.cache import cache_dir  # noqa: E402

HO = pd.Timestamp("2024-07-01")
COST = 0.001  # round trip, 5 bp per side
SPLITS = (("IS-a", None, "2019-01-01"), ("IS-b", "2019-01-01", "2021-10-01"), ("IS-c", "2021-10-01", HO))
ETF = {"Germany": "EWG", "France": "EWQ", "Italy": "EWI", "Spain": "EWP", "England": "EWU", "Netherlands": "EWN",
       "Switzerland": "EWL", "Sweden": "EWD", "Austria": "EWO", "Belgium": "EWK", "Denmark": "EDEN",
       "Norway": "ENOR", "Republic of Ireland": "EIRL", "Greece": "GREK", "Poland": "EPOL", "Turkey": "TUR",
       "Brazil": "EWZ", "Mexico": "EWW", "Argentina": "ARGT", "Chile": "ECH", "Peru": "EPU", "Japan": "EWJ",
       "South Korea": "EWY", "Australia": "EWA", "South Africa": "EZA", "Indonesia": "EIDO", "Malaysia": "EWM",
       "Saudi Arabia": "KSA", "Qatar": "QAT", "United Arab Emirates": "UAE", "Canada": "EWC", "India": "INDA",
       "China PR": "MCHI", "Thailand": "THD", "Philippines": "EPHE", "New Zealand": "ENZL", "Taiwan": "EWT"}
# Wikipedia titles -> tickers beyond exact S&P 500 / SEC name matches (fixed before running).
WIKI_EXTRA = {"Facebook": "META", "Google": "GOOGL", "Amazon_(company)": "AMZN", "Apple_Inc.": "AAPL",
              "Tesla,_Inc.": "TSLA", "Tesla": "TSLA", "Nvidia": "NVDA", "GameStop": "GME", "Walt_Disney": "DIS",
              "Luigi_Mangione": "UNH", "Brian_Thompson_(businessman)": "UNH", "Robinhood_(company)": "HOOD",
              "Uber_(company)": "UBER", "Moderna": "MRNA", "BioNTech": "BNTX", "Novavax": "NVAX", "Reddit": "RDDT",
              "Coinbase": "COIN", "Palantir_Technologies": "PLTR", "Zoom_Video_Communications": "ZM"}
# Extra names with their own Wikipedia article, kept by hand from the SEC-name matches (films, places dropped).
WIKI_WL = {"Abercrombie_&_Fitch": "ANF", "Roblox": "RBLX", "Harley-Davidson": "HOG", "Duolingo": "DUOL",
           "AstraZeneca": "AZN", "Cloudflare": "NET", "Playboy": "PLBY", "Infosys": "INFY", "Newsmax": "NMAX",
           "BuzzFeed": "BZFD", "Mattel": "MAT", "Virgin_Galactic": "SPCE", "Snowflake_Inc.": "SNOW", "Snap_Inc.": "SNAP",
           "Chegg": "CHGG", "Fastly": "FSLY", "Lululemon_Athletica": "LULU", "Match_Group": "MTCH", "Knightscope": "KSCP",
           "Houlihan_Lokey": "HLI", "Nasdaq": "NDAQ", "Texas_Roadhouse": "TXRH", "Stryker": "SYK",
           "The_Carlyle_Group": "CG", "Wendy's": "WEN", "Whirlpool_Corporation": "WHR", "Zscaler": "ZS"}
TOURNAMENTS = ["FIFA World Cup", "UEFA Euro", "Copa América", "AFC Asian Cup", "African Cup of Nations", "Gold Cup"]
SECTOR_JOBS = {"XLK": ["Software Development", "IT Systems & Solutions", "IT Infrastructure, Operations & Support",
                       "Data & Analytics"],
               "XLF": ["Banking & Finance", "Insurance", "Accounting"],
               "XLV": ["Nursing", "Physicians & Surgeons", "Medical Technician", "Pharmacy", "Dental", "Therapy"],
               "XLI": ["Production & Manufacturing", "Mechanical Engineering", "Industrial Engineering",
                       "Electrical Engineering", "Driving", "Logistic Support", "Loading & Stocking", "Aviation"],
               "XLY": ["Retail", "Hospitality & Tourism", "Food Preparation & Service"],
               "XLC": ["Media & Communications", "Marketing", "Arts & Entertainment"],
               "ITB": ["Construction", "Civil Engineering", "Architecture", "Installation & Maintenance"],
               "XBI": ["Scientific Research & Development"]}
QUAKE_ETF = {"Japan": "EWJ", "Chile": "ECH", "Turkey": "TUR", "Mexico": "EWW", "Taiwan": "EWT", "Indonesia": "EIDO",
             "Philippines": "EPHE", "New Zealand": "ENZL", "Peru": "EPU", "Italy": "EWI", "Greece": "GREK",
             "China": "MCHI"}


def prices(tickers, start="1999-01-01", end=None):
    """(open, close) panels, auto-adjusted; one cached parquet per ticker (missing ones downloaded)."""
    import yfinance as yf

    root = cache_dir("alt_data/px")
    tickers = sorted(set(tickers))
    todo = [t for t in tickers if not (root / f"{t}.parquet").exists()]
    for k in range(0, len(todo), 100):
        d = yf.download(todo[k:k + 100], start="1999-01-01", auto_adjust=True, progress=False)
        for t in todo[k:k + 100]:
            if t in d["Close"] and d["Close"][t].notna().any():
                pd.DataFrame({"Open": d["Open"][t], "Close": d["Close"][t]}).dropna().to_parquet(root / f"{t}.parquet")
    got = {t: pd.read_parquet(root / f"{t}.parquet") for t in tickers if (root / f"{t}.parquet").exists()}
    o = pd.DataFrame({t: x.Open for t, x in got.items()}).loc[start:end]
    c = pd.DataFrame({t: x.Close for t, x in got.items()}).loc[start:end]
    return o, c


class Px:
    def __init__(self, tickers, start="1999-01-01", end=None):
        o, c = prices(tickers, start, end)
        self.idx = c["SPY"].dropna().index
        self.O, self.C = o.reindex(self.idx), c.reindex(self.idx)

    def fwd(self, tk, d, h, bench="SPY", lag=1):
        """Open of the lag-th trading day after d -> close h-1 days later, minus the same for bench."""
        i = self.idx.searchsorted(pd.Timestamp(d), side="right") + lag - 1
        if tk not in self.C or i + h - 1 >= len(self.idx):
            return np.nan
        r = self.C[tk].iloc[i + h - 1] / self.O[tk].iloc[i] - 1
        if bench:
            r -= self.C[bench].iloc[i + h - 1] / self.O[bench].iloc[i] - 1
        return r


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def report(label, df, col="r", holdout=HO):
    """Mean (bp) and t clustered by month, in-sample / holdout / sub-periods."""
    def one(x):
        if len(x) == 0:
            return "n0"
        m = x.groupby(x.d.dt.to_period("M"))[col].mean()
        return f"n{len(x)} {x[col].mean() * 1e4:.0f}bp t{tstat(m):.2f}"
    df = df.dropna(subset=[col])
    parts = [f"IS {one(df[df.d < holdout])}", f"HO {one(df[df.d >= holdout])}"]
    parts += [f"{n} {one(df[(df.d >= (a or '1900')) & (df.d < b)])}" for n, a, b in SPLITS]
    print(f"{label:<42} " + " | ".join(parts))


def wiki_events(w, names, extra=WIKI_EXTRA):
    hit = w.assign(ticker=w.article.map(lambda a: extra.get(a) or names.get(ad.norm_company(a))))
    return ad.first_in(hit.dropna(subset=["ticker"]), "ticker", 30)


def batch1(px, w, sp500_names):
    print("\n== Batch 1")
    ev = wiki_events(w, sp500_names)
    ev = ev[ev.d > "2016-01-31"]
    for h in (5, 20):
        report(f"H1 S&P-name attention, long {h}d vs SPY", ev.assign(r=[px.fwd(t, d, h, lag=2) - COST
                                                                       for t, d in zip(ev.ticker, ev.d)]))
    fear = w[w.article.str.contains(r"^(?:Black_Monday|Recession|List_of_stock_market_crashes|Stock_market_crash"
                                    r"|20\d\d_stock_market_crash|Tariff|Smoot|Great_Depression)")]
    f = ad.first_in(fear.assign(k=1), "k", 10)
    report("H2 fear articles -> SPY 20d", f.assign(r=[px.fwd("SPY", d, 20, bench=None, lag=2) for d in f.d]))

    g = ad.football()
    g = g[g.tournament.isin(TOURNAMENTS) & (g.d >= "2000-01-01") & g.team.isin(ETF)]
    rows = []
    for d, team, res in zip(g.d, g.team, g.res):
        tk, i = ETF[team], px.idx.searchsorted(d, side="right")
        if tk not in px.C or i >= len(px.idx):
            continue
        cc = px.C[tk].iloc[i] / px.C[tk].iloc[i - 1] - px.C["EFA"].iloc[i] / px.C["EFA"].iloc[i - 1]
        oc = px.C[tk].iloc[i] / px.O[tk].iloc[i] - px.C["EFA"].iloc[i] / px.O["EFA"].iloc[i]
        rows.append(dict(d=d, team=team, res=res, cc=cc, oc=oc))
    F = pd.DataFrame(rows)
    report("H3 loss -> country ETF vs EFA close-close", F[F.res == "L"], "cc")
    report("H3 short losers open->close net", F[F.res == "L"].assign(r=lambda x: -x.oc - COST))

    n = ad.nfl_games()
    n = n[(n.season >= 2018) & (n.gameday.dt.weekday != 0)]
    n["mon"] = n.gameday + pd.to_timedelta((7 - n.gameday.dt.weekday) % 7, unit="D")
    s = n.groupby("mon").agg(fc=("fav_cover", "mean"), ov=("over", "mean"), k=("result", "size"))
    s = s[s.k >= 8]
    s["score"] = s.fc + s.ov - 1
    books = ["DKNG", "PENN", "RSI", "FLUT"]
    rows = []
    for d, sc in s.score.items():
        r = np.nanmean([px.fwd(b, d - pd.Timedelta(days=1), 5) for b in books])
        rows.append(dict(d=d, r=-np.sign(sc) * r - COST if sc else np.nan))
    report("H4 NFL bettor score -> books Mon-Fri", pd.DataFrame(rows))

    R = px.C.pct_change()
    idx = px.idx[px.idx >= "2008-01-01"]
    ram = pd.Series(ad.in_ramadan(idx), index=idx)
    rows = []
    for tk in ["TUR", "EIDO", "EWM", "KSA", "QAT", "UAE"]:
        x = (R[tk] - R["EEM"]).loc[idx].dropna()
        for _, grp in x.groupby(x.index.year):
            m = ram.loc[grp.index].values
            if m.sum() >= 10:
                rows.append(dict(d=grp.index[m][0], r=grp[m].sum() - grp[~m].mean() * m.sum()))
    report("H5 Ramadan, Muslim-market ETFs vs EEM", pd.DataFrame(rows))

    spy = R["SPY"].loc["2002":].dropna()
    new = ad.new_moon_half(spy.index)
    report("H6 SPY new-moon minus full-moon half", pd.DataFrame({"d": spy.index, "r": np.where(new, spy, -spy)}))
    mon = R["SPY"].loc["2000":].dropna()
    mon = mon[mon.index.weekday == 0]
    dst = []
    for y in range(2000, 2027):
        sun = lambda m: pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month, freq="W-SUN")
        dst += [sun(3)[1], sun(11)[0]] if y >= 2007 else [sun(4)[0], sun(10)[-1]]
    rows = [dict(d=s + pd.Timedelta(days=1), r=mon.get(s + pd.Timedelta(days=1), np.nan) - mon.mean()) for s in dst]
    report("H7 DST Monday SPY minus avg Monday", pd.DataFrame(rows))


def batch2(px, w, sp500_names):
    print("\n== Batch 2")
    ev = wiki_events(w, sp500_names, {**WIKI_EXTRA, **WIKI_WL})
    ev = ev[ev.d > "2016-01-31"]
    report("B2-H1 any listed name, short 5d vs SPY", ev.assign(r=[-px.fwd(t, d, 5, lag=2) - COST
                                                               for t, d in zip(ev.ticker, ev.d)]))
    first = w[w.d == w.groupby("article").d.transform("min")]
    first = first[first.d > "2016-01-31"]
    yr = first.d.dt.year.astype(str)
    cur = np.array([a.find(y) >= 0 or a.find(str(int(y) - 1)) >= 0 for a, y in zip(first.article, yr)])
    war = first[cur & first.article.str.contains(r"(?i)_war$|_war_|invasion|strikes_on|airstrike|offensive|attack_on_")
                & ~first.article.str.contains(r"(?i)film|season|game|series|album|song")]
    war = ad.first_in(war.assign(k=1), "k", 20)
    report("B2-H2 new war -> ITA vs SPY 20d", war.assign(r=[px.fwd("ITA", d, 20, lag=2) - COST for d in war.d]))
    ob = first[first.article.str.contains(r"(?i)pandemic|outbreak|pox|_flu|virus|measles|ebola|epidemic")
               & ~first.article.str.contains(r"(?i)film|season|series|in_the|by_country|tornado|spanish|1918")]
    ob = ad.first_in(ob.assign(k=1), "k", 30)
    vac = ["MRNA", "NVAX", "BNTX", "EBS", "SIGA"]
    report("B2-H3 outbreak -> vaccines vs XBI 20d",
           ob.assign(r=[np.nanmean([px.fwd(t, d, 20, bench="XBI", lag=2) for t in vac]) - COST for d in ob.d]))
    cr = first[first.article.str.contains(r"(?i)flight_\d|air_disaster|plane_crash|mid-air_collision")]
    report("B2-H4 air crash -> short BA vs SPY 5d", cr.assign(r=[-px.fwd("BA", d, 5, lag=2) - COST for d in cr.d]))


def batch3(px, sec):
    print("\n== Batch 3")
    p = ad.indeed_sectors()
    S = pd.DataFrame({e: np.log(p[c]).mean(axis=1) for e, c in SECTOR_JOBS.items()})
    mom = (S - S.shift(91)).dropna(how="all")
    me = px.C.loc["2020"].resample("ME").last().index.union(px.C.loc["2021":].resample("ME").last().index)
    rows = []
    for a, b in zip(me[:-1], me[1:]):
        q = mom.loc[:a - pd.Timedelta(days=7)]
        if len(q) == 0:
            continue
        sig = q.iloc[-1].dropna()
        r = px.C.loc[a:b, list(SECTOR_JOBS)].iloc[[0, -1]].pct_change().iloc[-1]
        rows.append(dict(d=b, r=r[sig.nlargest(3).index].mean() - r.mean() - COST * 2 / 3))
    report("B3-H2 Indeed postings momentum top-3 vs EW", pd.DataFrame(rows))

    w = ad.warn_notices()
    w["ticker"] = w.company_canonical.map(lambda s: sec.get(ad.norm_company(s)))
    e = w.dropna(subset=["ticker", "notice_date"])
    e = e[e.notice_date >= "2005-01-01"].groupby(["ticker", "notice_date"]).employees_affected.sum().reset_index()
    e = ad.first_in(e.rename(columns={"notice_date": "d"}), "ticker", 60)
    e = e[e.employees_affected >= 250]
    wp = Px(set(e.ticker) | {"SPY"}, start="2005-01-01")
    report("B3-H3 WARN >= 250 -> long 20d vs SPY (gross)",
           e.assign(r=[wp.fwd(t, d, 20, lag=6) for t, d in zip(e.ticker, e.d)]))

    h = ad.hn_daily(2014)
    wk = h.resample("W-SUN").sum()
    share = wk.crypto / wk.n
    z = (share.rolling(4).mean() - share.rolling(52).mean()) / share.rolling(52).std()
    btc = px.C["BTC-USD"].dropna() if "BTC-USD" in px.C else None
    if btc is not None:
        f = btc.resample("W-SUN").last().pct_change().shift(-1)
        D = pd.DataFrame({"z": z, "f": f}).dropna().loc["2015":]
        D["d"], D["r"] = D.index, np.where(D.z > 1, -D.f, 0.0)
        report("B3-H4 HN crypto attention z>1 -> BTC out", D[D.z > 1])

    q = ad.quakes(6.5)
    q = q[q.type == "earthquake"].copy()
    q["tk"] = q.place.fillna("").map(lambda s: next((v for k, v in QUAKE_ETF.items()
                                                     if s.endswith(k) or f", {k}" in s), None))
    q = ad.first_in(q.dropna(subset=["tk"]).rename(columns={"time": "d"}), "tk", 5)
    lag = [1 if t.hour < 14 else 2 for t in q.d]  # US open ~14:30 UTC
    report("B3-H7 M6.5+ quake -> buy country ETF 5d vs EFA",
           q.assign(r=[px.fwd(t, d.normalize() - pd.Timedelta(days=1 if k == 1 else 0), 5, bench="EFA", lag=1) - COST
                       for t, d, k in zip(q.tk, q.d, lag)]))


def heavy():
    """SOVAI daily Wikipedia views (weekly attention quintiles), CFPB complaints, lobbying (2015-2024-05)."""
    print("\n== Batch 3 heavy (SOVAI, late test 2022-01..2024-05 instead of a holdout)")
    late = pd.Timestamp("2022-01-01")
    v = ad.sovai("wikipedia_views")
    v["ticker"] = v.ticker.str.replace(".", "-")
    px = Px(set(v.ticker) | {"SPY"}, start="2015-06-01", end="2024-07-01")
    P = v[v.ticker.isin(px.C.columns)].pivot_table(index="date", columns="ticker", values="views")
    P = P.reindex(pd.date_range(P.index.min(), P.index.max())).ffill(limit=3)
    av = np.log1p(P.rolling(5).mean()) - np.log1p(P.shift(5).rolling(60, min_periods=40).median())
    rows = []
    for f in [d for d in px.idx if d.weekday() == 4 and d in av.index]:
        i = px.idx.searchsorted(f, side="right")
        if i + 5 >= len(px.idx):
            continue
        a = av.loc[f].dropna()
        a = a[px.C.loc[f, a.index] > 5]
        r = (px.O.iloc[i + 5] / px.O.iloc[i] - 1)[a.index].dropna()
        if len(r) < 200:
            continue
        qn = pd.qcut(a[r.index].rank(method="first"), 5, labels=False)
        rows.append(dict(d=px.idx[i], r=r[qn == 0].mean() - r[qn == 4].mean()))
    report("B3-H1 low minus high attention, weekly (gross)", pd.DataFrame(rows), holdout=late)


def main():
    ua = os.environ.get("SEC_USER_AGENT", "kitchensink research research@example.com")
    w = ad.wiki_top100()
    import requests

    spx = pd.read_csv(io_bytes(requests.get(ad.GH + "datasets/s-and-p-500-companies/main/data/constituents.csv",
                                            timeout=60).content))
    sp500_names = {ad.norm_company(n): t.replace(".", "-") for n, t in zip(spx.Security, spx.Symbol)}
    sec = ad.sec_names(ua)
    hits = set(wiki_events(w, sp500_names, {**WIKI_EXTRA, **WIKI_WL}).ticker)
    tick = hits | set(ETF.values()) | set(QUAKE_ETF.values()) | set(WIKI_WL.values())
    tick |= {"SPY", "EFA", "EEM", "ITA", "USO", "XBI", "BA", "JETS", "DKNG", "PENN", "RSI", "FLUT", "BTC-USD",
             "MRNA", "NVAX", "BNTX", "EBS", "SIGA"} | set(SECTOR_JOBS)
    px = Px(tick)
    batch1(px, w, sp500_names)
    batch2(px, w, sp500_names)
    batch3(px, sec)
    if "--heavy" in sys.argv:
        heavy()


def io_bytes(b):
    import io

    return io.BytesIO(b)


if __name__ == "__main__":
    main()
