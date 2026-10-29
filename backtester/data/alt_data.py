"""Unusual non-financial data sources reachable from the cloud (HF, raw GitHub,
SEC), used by `examples/run_nonfin_data.py`.

Each loader caches under `cache_dir("alt_data")`. Sources:

- Wikipedia daily top-100 articles, 2016+ (HF `vtasca/wikipedia-pageviews`;
  day D is public by D+1 or D+2).
- International football results + shootouts, 1872+ (GitHub `martj42`).
- NFL games with closing spread / total lines (GitHub `nflverse/nfldata`).
- WARN Act layoff notices, 48 states, 1988+ (HF `APProjects`).
- Indeed Hiring Lab US postings by occupation, daily 2020-02+ (GitHub
  `hiring-lab`; revised index, not point-in-time).
- USGS earthquakes M6+, 2001-01..2025-10 (HF `mnemoraorg`).
- Hacker News daily keyword story counts, every story 2006+ (HF
  `open-index/hacker-news`, monthly parquet read by range requests).
- sovai ticker-matched sets (`wikipedia_views`, `cfpb_complaints`,
  `lobbying_data`): free tier has a 6-month lag and ends 2024-05.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .cache import cache_dir

HF = "https://huggingface.co/datasets/"
GH = "https://raw.githubusercontent.com/"
URLS = {
    "wiki_top100": HF + "vtasca/wikipedia-pageviews/resolve/main/pageviews.csv",
    "football": GH + "martj42/international_results/master/results.csv",
    "shootouts": GH + "martj42/international_results/master/shootouts.csv",
    "nfl": GH + "nflverse/nfldata/master/data/games.csv",
    "warn": HF + "APProjects/us-warn-act-layoffs-notices-daily/resolve/main/data/warn_notices.csv",
    "indeed_sectors": GH + "hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv",
    "sec_tickers": "https://www.sec.gov/files/company_tickers_exchange.json",
}
SOVAI = {"wikipedia_views": ["ticker", "date", "views"],
         "cfpb_complaints": ["date", "ticker", "product"],
         "lobbying_data": ["ticker", "date", "spend", "quarter"]}
HN_KEYWORDS = {"crypto": r"\bbitcoin|\bbtc\b|crypto|ethereum|blockchain|\bnft|stablecoin|coinbase|binance",
               "ai": r"\bopenai|\bgpt|\bllm|chatgpt|\bclaude\b|anthropic|\bai\b|nvidia"}
_SUFFIX = (r"(,? (inc|incorporated|corporation|corp|company|co|plc|ltd|limited|group|holdings?|nv|sa|se|ag"
           r"|llc|lp|the|class a|/de/|/new/))+$")


def _get(name: str, refresh: bool = False, headers: dict | None = None) -> bytes:
    path = cache_dir("alt_data") / URLS[name].rsplit("/", 1)[-1]
    if refresh or not path.exists():
        import requests

        r = requests.get(URLS[name], timeout=300, headers=headers or {})
        r.raise_for_status()
        path.write_bytes(r.content)
    return path.read_bytes()


def _csv(name: str, refresh: bool = False, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(name, refresh)), **kw)


def wiki_top100(refresh: bool = False) -> pd.DataFrame:
    """rank, article, views, d (the day viewed)."""
    w = _csv("wiki_top100", refresh).dropna(subset=["article"])
    w["d"] = pd.to_datetime(w["date"])
    return w.drop(columns="date")


def football(refresh: bool = False) -> pd.DataFrame:
    """Results with shootouts folded in: one row per team per match (d, team, opp, res W/L/D, tournament)."""
    g = _csv("football", refresh, parse_dates=["date"]).dropna(subset=["home_score"])
    sh = _csv("shootouts", refresh, parse_dates=["date"])
    win = {(d, frozenset((h, a))): w for d, h, a, w in zip(sh.date, sh.home_team, sh.away_team, sh.winner)}
    rows = []
    for r in g.itertuples():
        for team, opp, gf, ga in ((r.home_team, r.away_team, r.home_score, r.away_score),
                                  (r.away_team, r.home_team, r.away_score, r.home_score)):
            res = "W" if gf > ga else "L" if gf < ga else "D"
            w = win.get((r.date, frozenset((team, opp))))
            if res == "D" and w is not None:
                res = "W" if w == team else "L"
            rows.append((r.date, team, opp, res, r.tournament))
    return pd.DataFrame(rows, columns=["d", "team", "opp", "res", "tournament"])


def nfl_games(refresh: bool = False) -> pd.DataFrame:
    """Finished NFL games with fav_cover (1/0, NaN on push or pick'em) and over (1/0)."""
    g = _csv("nfl", refresh, parse_dates=["gameday"]).dropna(subset=["result", "spread_line", "total_line"])
    s, res = g.spread_line, g.result
    g["fav_cover"] = np.where(s > 0, res > s, np.where(s < 0, res < s, np.nan)).astype(float)
    g.loc[(res == s) | (s == 0), "fav_cover"] = np.nan
    g["over"] = np.where(g.total > g.total_line, 1.0, np.where(g.total < g.total_line, 0.0, np.nan))
    return g


def warn_notices(refresh: bool = False) -> pd.DataFrame:
    w = _csv("warn", refresh)
    w["notice_date"] = pd.to_datetime(w.notice_date, errors="coerce")
    return w


def indeed_sectors(refresh: bool = False) -> pd.DataFrame:
    """Daily total-postings index (Feb 2020 = 100), one column per occupation."""
    d = _csv("indeed_sectors", refresh, parse_dates=["date"])
    d = d[d.variable == "total postings"]
    return d.pivot_table(index="date", columns="display_name", values="indeed_job_postings_index")


def norm_company(name: str) -> str:
    """Lower-case company name without suffixes, parentheses or punctuation, for matching."""
    s = str(name).replace("_", " ").lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[.,]", "", s).replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):
        s = re.sub(_SUFFIX, "", s).strip()
    return re.sub(r"^the ", "", s)


def sec_names(user_agent: str, refresh: bool = False) -> dict:
    """normalised company name -> ticker for NYSE/Nasdaq filers (first ticker per CIK)."""
    import json

    j = json.loads(_get("sec_tickers", refresh, headers={"User-Agent": user_agent}))
    t = pd.DataFrame(j["data"], columns=j["fields"])
    t = t[t.exchange.isin(["NYSE", "Nasdaq"]) & ~t.ticker.str.contains("-")].drop_duplicates("cik")
    out = {}
    for n, tk in zip(t.name, t.ticker):
        k = norm_company(n)
        if len(k) >= 4:
            out.setdefault(k, tk)
    return out


def first_in(events: pd.DataFrame, key: str, gap_days: int, date: str = "d") -> pd.DataFrame:
    """Keep rows whose `key` was not seen in the previous `gap_days` days (first row of each key kept)."""
    e = events.drop_duplicates([key, date]).sort_values(date)
    gap = e.groupby(key)[date].diff().dt.days
    return e[gap.isna() | (gap > gap_days)]


def quakes(min_mag: float = 6.0, refresh: bool = False) -> pd.DataFrame:
    """USGS catalog rows with mag >= min_mag (time UTC, mag, place, type), 2001-01..2025-10."""
    path = cache_dir("alt_data") / "quakes6.parquet"
    if refresh or not path.exists():
        import requests

        out = []
        for d in pd.date_range("2001-01-01", "2025-10-01", freq="MS"):
            r = requests.get(HF + "mnemoraorg/usgs-global-earthquake-catalog/resolve/main/"
                             + d.strftime("%Y%m%d") + ".csv", timeout=120)
            if r.status_code == 200:
                x = pd.read_csv(io.BytesIO(r.content), usecols=["time", "mag", "place", "type"])
                out.append(x[x.mag >= 6.0])
        q = pd.concat(out)
        q["time"] = pd.to_datetime(q.time, utc=True, format="mixed").dt.tz_localize(None)
        q.to_parquet(path)
    q = pd.read_parquet(path)
    return q[q.mag >= min_mag]


def hn_daily(start_year: int = 2010, end_month: str = "2026-08", keywords: dict | None = None) -> pd.DataFrame:
    """Daily Hacker News story counts: n (all titled items) plus one column per keyword regex.

    Reads only the type/time/title columns of each monthly parquet by HTTP range
    requests (~0.1-0.3 GB per month otherwise); months are cached one by one."""
    import pyarrow.parquet as pq

    from .httpfile import HTTPFile

    kw = keywords or HN_KEYWORDS
    out = []
    for m in pd.period_range(f"{start_year}-01", end_month, freq="M"):
        path = cache_dir("alt_data/hn") / f"{m}.parquet"
        if not path.exists():
            f = pq.ParquetFile(HTTPFile(f"{HF}open-index/hacker-news/resolve/main/data/{m.year}/{m}.parquet"))
            t = pd.concat([f.read_row_group(g, columns=["time", "title"]).to_pandas()
                           for g in range(f.num_row_groups)])
            t = t[t.title != ""]
            d = t.time.dt.tz_convert(None).dt.floor("D")
            tl = t.title.str.lower()
            agg = {"n": t.groupby(d).size()}
            for k, p in kw.items():
                agg[k] = tl.str.contains(p, regex=True).groupby(d).sum()
            pd.DataFrame(agg).to_parquet(path)
        out.append(pd.read_parquet(path))
    return pd.concat(out).groupby(level=0).sum()


def sovai(name: str, refresh: bool = False) -> pd.DataFrame:
    """One of SOVAI's ticker-matched sets, only the columns the tests use."""
    path = cache_dir("alt_data") / f"sovai_{name}.parquet"
    if refresh or not path.exists():
        import pyarrow.parquet as pq

        from .httpfile import HTTPFile

        f = pq.ParquetFile(HTTPFile(f"{HF}sovai/{name}/resolve/main/{name}.parquet"))
        t = pd.concat([f.read_row_group(g, columns=SOVAI[name]).to_pandas() for g in range(f.num_row_groups)])
        t.to_parquet(path)
    return pd.read_parquet(path)


def new_moon_half(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on days within 7 days of a new moon (needs `ephem`)."""
    import ephem

    age = [(d.to_pydatetime() - ephem.Date(ephem.previous_new_moon(d.to_pydatetime())).datetime()).days
           for d in dates]
    return (np.array(age) <= 7) | (np.array(age) >= 22)


def in_ramadan(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on days in the Islamic month of Ramadan (needs `hijridate`)."""
    from hijridate import Gregorian

    return np.array([Gregorian(d.year, d.month, d.day).to_hijri().month == 9 for d in dates])
