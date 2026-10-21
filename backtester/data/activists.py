"""Schedule 13D filings (5%+ stakes with intent to influence) from EDGAR.

The quarterly form index (https://www.sec.gov/Archives/edgar/full-index/)
lists every filing once per entity on it, so a 13D appears under both the
subject company and each filer. Grouping by accession number gives
(subject, filers, filing date). Form names changed in December 2024 from
"SC 13D" to "SCHEDULE 13D" when the SEC moved 13D to structured data.

Subjects are matched to tickers through the SEC's current ticker list, so
targets that were later acquired or delisted, a common activist outcome,
drop out. The activist list is names well known for public campaigns; it is
chosen with hindsight about who is an activist, not about who made money.
"""
from __future__ import annotations

import gzip
import io
import re

import pandas as pd

from .cache import cache_dir
from .edgar import _get, ticker_to_cik

FORMS = ("SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A")
ACTIVISTS = {
    "Elliott": r"ELLIOTT (ASSOCIATES|INVESTMENT|INTERNATIONAL|MANAGEMENT)",
    "Starboard": r"STARBOARD VALUE",
    "Icahn": r"ICAHN",
    "Trian": r"TRIAN (FUND|PARTNERS)",
    "Pershing Square": r"PERSHING SQUARE",
    "ValueAct": r"VALUEACT",
    "Third Point": r"THIRD POINT",
    "Jana": r"JANA PARTNERS",
    "Engaged": r"ENGAGED CAPITAL",
    "Land & Buildings": r"LAND (&|AND) BUILDINGS",
    "Sachem Head": r"SACHEM HEAD",
    "Corvex": r"CORVEX",
    "Legion": r"LEGION PARTNERS",
    "Ancora": r"ANCORA",
    "Barington": r"BARINGTON",
    "Engine Capital": r"ENGINE CAPITAL",
    "Politan": r"POLITAN",
    "Mantle Ridge": r"MANTLE RIDGE",
    "Glenview": r"GLENVIEW CAPITAL",
    "Marcato": r"MARCATO",
    "Blue Harbour": r"BLUE HARBOUR",
    "Scopia": r"SCOPIA",
    "Cevian": r"CEVIAN",
    "HG Vora": r"HG VORA",
    "FrontFour": r"FRONTFOUR",
    "Impactive": r"IMPACTIVE",
    "Sarissa": r"SARISSA",
    "Voce": r"VOCE CAPITAL",
    "Irenic": r"IRENIC",
    "Anson": r"ANSON (FUNDS|ADVISORS|MANAGEMENT)",
}


def form_index(year: int, qtr: int, refresh: bool = False) -> pd.DataFrame:
    """13D rows of one quarter's EDGAR form index, cached as csv."""
    path = cache_dir("edgar/13d") / f"{year}q{qtr}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"cik": str}, parse_dates=["filed"])
    raw = _get(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.gz").content
    rows = []
    for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="latin-1"):
        form = line[:17].strip()
        if form not in FORMS:
            continue
        # Fixed-width columns: form, company, CIK, date, path.
        m = re.match(r"(.{17})(.*?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)", line)
        if m:
            rows.append((form, m[2].strip(), m[3], m[4], m[5].rsplit("/", 1)[-1].replace(".txt", "")))
    df = pd.DataFrame(rows, columns=["form", "name", "cik", "filed", "accession"])
    df["filed"] = pd.to_datetime(df["filed"])
    df.to_csv(path, index=False)
    return df


def filings(since: int = 2010, until: int | None = None) -> pd.DataFrame:
    """All 13D index rows from `since` (a year) to now."""
    until = until or pd.Timestamp.today().year
    parts = []
    for y in range(since, until + 1):
        for q in range(1, 5):
            if pd.Timestamp(year=y, month=3 * q - 2, day=1) > pd.Timestamp.today():
                break
            parts.append(form_index(y, q))
    return pd.concat(parts, ignore_index=True)


def events(since: int = 2010, activists: dict | None = None, originals_only: bool = True) -> pd.DataFrame:
    """One row per 13D on a company with a current ticker.

    Columns: date, symbol, subject, filer, activist (the matching name from
    `activists`, or None for other filers), form. A filing is kept only when
    exactly one entity on it has a ticker, so filings between two public
    companies are dropped rather than guessed.
    """
    pats = {k: re.compile(v) for k, v in (activists or ACTIVISTS).items()}
    cik_ticker = {}
    for t, c in ticker_to_cik().items():
        cik_ticker.setdefault(str(c), t)  # first listed is the primary class
    df = filings(since)
    if originals_only:
        df = df[df["form"].isin(("SC 13D", "SCHEDULE 13D"))]
    df = df.assign(cik=df["cik"].str.lstrip("0"))
    df["symbol"] = df["cik"].map(cik_ticker)

    def who(name):
        return next((k for k, p in pats.items() if p.search(name.upper())), None)

    df["activist"] = df["name"].map(who)
    rows = []
    for acc, g in df.groupby("accession", sort=False):
        subj = g[g["symbol"].notna() & g["activist"].isna()]
        if len(subj["symbol"].unique()) != 1:
            continue
        filers = g.drop(subj.index)
        act = filers["activist"].dropna()
        rows.append((g["filed"].iloc[0], subj["symbol"].iloc[0], subj["name"].iloc[0],
                     "; ".join(filers["name"]), act.iloc[0] if len(act) else None, g["form"].iloc[0]))
    out = pd.DataFrame(rows, columns=["date", "symbol", "subject", "filer", "activist", "form"])
    return out.sort_values("date").reset_index(drop=True)
