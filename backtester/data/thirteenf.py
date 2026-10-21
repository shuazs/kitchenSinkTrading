"""Hedge fund holdings from the SEC's Form 13F data sets.

https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
publishes every 13F filed in a period (calendar quarters up to 2023, then
three-month windows) as one zip of TSV tables, from 2013 Q2 when filings
became structured XML. Managers with over $100M in US stocks must file within
45 days of quarter end, listing long positions only: no shorts, no cash,
no bonds, and options are flagged in PUTCALL.

Each zip (up to ~100 MB) is reduced once to a small csv.gz holding, per
original 13F-HR filing: manager CIK and name, filing date, period, total
value, number of stock positions, and that filing's `TOP_N` largest stock
positions by CUSIP with their portfolio weight. Weights are computed from
share positions only (options and principal amounts dropped).

VALUE was reported in thousands of dollars until the end of 2022 and in
dollars from 2023 filings on; `value_usd` is normalised to dollars.

CUSIPs are mapped to tickers with Yahoo's search and then by issuer name
against the SEC's ticker list. Both only know listings that exist today, so
names that were delisted or acquired drop out.
"""
from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd

from .cache import cache_dir
from .edgar import _get

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
TOP_N = 25


def zip_links(refresh: bool = False) -> list[str]:
    """Download URLs of every 13F data set zip, oldest first."""
    path = cache_dir("edgar/13f") / "links.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    html = _get(INDEX_URL).text
    links = sorted(set(re.findall(r'href="([^"]*?_form13f\.zip)"', html)), key=_sort_key)
    links = [u if u.startswith("http") else "https://www.sec.gov" + u for u in links]
    path.write_text(json.dumps(links))
    return links


def _sort_key(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    m = re.match(r"(\d{4})q(\d)", name)
    if m:
        return f"{m[1]}-{int(m[2]) * 3 - 2:02d}"
    m = re.match(r"\d{2}([a-z]{3})(\d{4})", name)
    return f"{m[2]}-{pd.Timestamp(f'1 {m[1]} 2000').month:02d}"


def _reduce(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    # Some zips put the tables in a subfolder.
    names = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}

    def read(name, cols):
        with z.open(names[name]) as f:
            return pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3, on_bad_lines="skip")

    sub = read("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"]
    cover = read("COVERPAGE.tsv", ["ACCESSION_NUMBER", "FILINGMANAGER_NAME", "REPORTTYPE"])
    sub = sub.merge(cover, on="ACCESSION_NUMBER")
    sub = sub[sub["REPORTTYPE"].fillna("").str.contains("HOLDINGS REPORT")]
    keep = set(sub["ACCESSION_NUMBER"])

    parts = []
    with z.open(names["INFOTABLE.tsv"]) as f:
        cols = ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMTTYPE", "PUTCALL"]
        for chunk in pd.read_csv(f, sep="\t", dtype=str, usecols=cols, quoting=3, on_bad_lines="skip", chunksize=500_000):
            chunk = chunk[chunk["ACCESSION_NUMBER"].isin(keep)]
            chunk = chunk[(chunk["SSHPRNAMTTYPE"] == "SH") & chunk["PUTCALL"].isna()]
            chunk["VALUE"] = pd.to_numeric(chunk["VALUE"], errors="coerce")
            parts.append(chunk[["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE"]])
    info = pd.concat(parts, ignore_index=True)
    info["CUSIP"] = info["CUSIP"].str.upper().str[:9]
    pos = info.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False).agg(
        value=("VALUE", "sum"), issuer=("NAMEOFISSUER", "first")
    )
    pos = pos[pos["value"] > 0]
    totals = pos.groupby("ACCESSION_NUMBER")["value"].agg(total="sum", n_positions="size")
    pos = pos.join(totals, on="ACCESSION_NUMBER")
    pos["weight"] = pos["value"] / pos["total"]
    pos = pos.sort_values(["ACCESSION_NUMBER", "weight"], ascending=[True, False])
    pos["rank"] = pos.groupby("ACCESSION_NUMBER").cumcount() + 1
    top10 = pos[pos["rank"] <= 10].groupby("ACCESSION_NUMBER")["weight"].sum().rename("top10_share")
    pos = pos[pos["rank"] <= TOP_N].join(top10, on="ACCESSION_NUMBER")

    out = pos.merge(sub, on="ACCESSION_NUMBER")
    out["filed"] = pd.to_datetime(out["FILING_DATE"], format="%d-%b-%Y")
    out["period"] = pd.to_datetime(out["PERIODOFREPORT"], format="%d-%b-%Y")
    # Values switched from thousands to dollars for filings made from 2023.
    scale = (out["filed"] < "2023-01-03").map({True: 1000.0, False: 1.0})
    out["total_usd"] = out["total"] * scale
    return out.rename(columns={"ACCESSION_NUMBER": "accession", "CIK": "manager_cik", "FILINGMANAGER_NAME": "manager", "CUSIP": "cusip"})[
        ["accession", "manager_cik", "manager", "filed", "period", "total_usd", "n_positions", "top10_share",
         "rank", "cusip", "issuer", "weight"]
    ]


def load_holdings(since: str = "2013-01-01", refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Top positions of every original 13F holdings report since `since`.

    One row per (filing, position) for each filing's TOP_N largest stock
    positions. First call downloads every zip (~3 GB total, a few minutes);
    later calls read the cached reductions.
    """
    out = []
    for url in zip_links(refresh):
        name = url.rsplit("/", 1)[-1].replace(".zip", "")
        path = cache_dir("edgar/13f") / f"{name}.csv.gz"
        if not path.exists() or refresh:
            if verbose:
                print("13F:", name, flush=True)
            _reduce(_get(url).content).to_csv(path, index=False)
        df = pd.read_csv(path, dtype={"cusip": str, "manager_cik": str}, parse_dates=["filed", "period"])
        out.append(df[df["filed"] >= since])
    df = pd.concat(out, ignore_index=True)
    # A manager sometimes files two originals for one quarter; keep the first seen.
    first = df.groupby(["manager_cik", "period"])["filed"].transform("min")
    df = df[df["filed"] == first]
    return df.drop_duplicates(["manager_cik", "period", "cusip"]).reset_index(drop=True)


FUND_WORDS = re.compile(r"\b(ETF|ETFS|INDEX|INDX|ISHARES|SPDR|VANGUARD|INVESCO|PROSHARES|DIMENSIONAL|SELECT SECTOR|TR|TRUST|FDS|FUND|FUNDS)\b")
SUFFIXES = re.compile(
    r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|NV|SA|AG|LP|LLC|HLDGS?|HOLDINGS?|GROUP|DEL|NEW|THE|"
    r"CL|CLASS|COM|SHS|ORD|ADR|ADS|SPONSORED|SPON|A|B|C)\b"
)
ABBREV = {"MFG": "MANUFACTURING", "INTL": "INTERNATIONAL", "TECHNOLOGIES": "TECHNOLOGY", "TECH": "TECHNOLOGY",
          "SYS": "SYSTEMS", "COMMUNICATIONS": "COMMUNICATION", "PHARMACEUTICALS": "PHARMACEUTICAL", "MANUFAC": "MANUFACTURING"}


def _norm(name: str) -> str:
    words = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    words = [ABBREV.get(w, w) for w in words]
    return " ".join(SUFFIXES.sub(" ", " ".join(words)).split())


def name_tickers(issuers) -> dict[str, str | None]:
    """Match issuer names to the SEC's current ticker list after removing
    punctuation and corporate suffixes. Fund and ETF names are not matched."""
    from .edgar import _cached_json

    data = _cached_json("company_tickers.json", "https://www.sec.gov/files/company_tickers.json", False)
    by_name: dict[str, str] = {}
    for v in data.values():
        by_name.setdefault(_norm(v["title"]), v["ticker"].replace(".", "-"))
    return {n: (None if FUND_WORDS.search(str(n).upper()) else by_name.get(_norm(n))) for n in dict.fromkeys(issuers)}


def cusip_tickers(cusips, issuers=None, refresh: bool = False) -> dict[str, str | None]:
    """Map CUSIPs to current tickers (None where there is no match).

    Tries Yahoo's search by CUSIP first, then, when `issuers` (names in the
    same order as `cusips`) is given, the issuer name against the SEC ticker
    list; Yahoo's CUSIP search misses some large names (Alphabet, Berkshire).
    Yahoo results are cached, including misses, so each CUSIP is looked up once.
    """
    import yfinance as yf

    cusips = list(cusips)
    path = cache_dir("edgar/13f") / "cusip_tickers.json"
    known = json.loads(path.read_text()) if path.exists() and not refresh else {}
    todo = [c for c in dict.fromkeys(cusips) if c not in known]
    for i, c in enumerate(todo):
        try:
            quotes = yf.Search(c, max_results=3).quotes
        except Exception:
            continue
        equity = [q for q in quotes if q.get("quoteType") == "EQUITY" and "." not in q.get("symbol", ".")]
        known[c] = equity[0]["symbol"] if equity else None
        if i % 50 == 49:
            path.write_text(json.dumps(known))
    path.write_text(json.dumps(known))
    out = {c: known.get(c) for c in cusips}
    if issuers is not None:
        by_name = name_tickers(issuers)
        for c, n in zip(cusips, issuers):
            if out[c] is None:
                out[c] = by_name.get(n)
    return out
