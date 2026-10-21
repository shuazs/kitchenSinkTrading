"""Bulk insider transactions from the SEC's Form 3/4/5 quarterly data sets.

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
publishes every Form 3/4/5 filed in a quarter as one zip of TSV tables. One
download per quarter covers every issuer, so a cluster-buying scan does not
need a pre-chosen ticker list or one request per Form 4 (and avoids the
~1000-filing truncation of the submissions API used by `edgar.insider_filings`).

Each quarter is reduced to open-market purchases (code P) and sales (code S)
and cached as a small csv.gz, so the ~15 MB zip is only fetched once.
Dates are FILING dates (when the market could see the trade) plus the
transaction date for reference.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from .cache import cache_dir
from .edgar import _get, ticker_to_cik

INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"

COLUMNS = [
    "accession", "filed", "trans_date", "issuer_cik", "filed_ticker", "issuer_name",
    "owner_cik", "owner_name", "officer", "director", "ten_pct", "title",
    "code", "shares", "price", "value", "plan_10b5_1",
]


def quarter_links(refresh: bool = False) -> dict[str, str]:
    """Map '2023q1' -> download URL, scraped from the SEC index page (the
    path prefix changed in 2026, so links are not hard-coded)."""
    path = cache_dir("edgar/form345") / "links.csv"
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0)["url"]
        return s.to_dict()
    html = _get(INDEX_URL).text
    links = {}
    for href in re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html):
        url, key = href
        links[key] = url if url.startswith("http") else "https://www.sec.gov" + url
    pd.Series(links, name="url").to_csv(path)
    return links


def _parse_zip(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))

    def read(name, cols):
        with z.open(name) as f:
            # Columns added over the years (e.g. AFF10B5ONE in 2023) are optional.
            return pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3, on_bad_lines="skip")

    sub = read("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                  "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    tx = read("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                     "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    own = read("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                      "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])

    tx = tx[tx["TRANS_CODE"].isin(["P", "S"])]
    if "AFF10B5ONE" not in sub:
        sub["AFF10B5ONE"] = None
    # Original Form 4s only: amendments (4/A) restate trades already counted
    # and Form 5s are annual catch-up reports filed long after the trade.
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]

    # A joint filing lists several owners for the same trades; keep one row
    # per filing (the first owner) with the union of their roles.
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
    own = own.assign(
        officer=rel.str.contains("Officer"),
        director=rel.str.contains("Director"),
        ten_pct=rel.str.contains("TenPercent"),
    )
    roles = own.groupby("ACCESSION_NUMBER")[["officer", "director", "ten_pct"]].any()
    first = own.drop_duplicates("ACCESSION_NUMBER").set_index("ACCESSION_NUMBER")[
        ["RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_TITLE"]
    ]
    owners = first.join(roles)

    df = tx.merge(sub, on="ACCESSION_NUMBER").merge(owners, left_on="ACCESSION_NUMBER", right_index=True)
    df = df.rename(columns={
        "ACCESSION_NUMBER": "accession", "FILING_DATE": "filed", "TRANS_DATE": "trans_date",
        "ISSUERCIK": "issuer_cik", "ISSUERTRADINGSYMBOL": "filed_ticker", "ISSUERNAME": "issuer_name",
        "RPTOWNERCIK": "owner_cik", "RPTOWNERNAME": "owner_name", "RPTOWNER_TITLE": "title",
        "TRANS_CODE": "code", "TRANS_SHARES": "shares", "TRANS_PRICEPERSHARE": "price",
        "AFF10B5ONE": "plan_10b5_1",
    })
    df["filed"] = pd.to_datetime(df["filed"], format="%d-%b-%Y", errors="coerce")
    df["trans_date"] = pd.to_datetime(df["trans_date"], format="%d-%b-%Y", errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["value"] = df["shares"] * df["price"]
    df["issuer_cik"] = pd.to_numeric(df["issuer_cik"], errors="coerce")
    df["owner_cik"] = pd.to_numeric(df["owner_cik"], errors="coerce")
    df["plan_10b5_1"] = df.get("plan_10b5_1", pd.Series(index=df.index, dtype=str)).isin(["1", "true"])
    df = df.dropna(subset=["filed", "issuer_cik", "shares"])
    return df[COLUMNS].reset_index(drop=True)


def load_quarter(key: str, refresh: bool = False) -> pd.DataFrame:
    """Open-market buys and sells (codes P and S) filed in one quarter, e.g. '2023q1'."""
    path = cache_dir("edgar/form345") / f"{key}_ps.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["filed", "trans_date"], dtype={"filed_ticker": str})
    url = quarter_links().get(key) or quarter_links(refresh=True)[key]
    df = _parse_zip(_get(url).content)
    df.to_csv(path, index=False)
    return df


def _quarters(start, end) -> list[str]:
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end or pd.Timestamp.today()), freq="Q")
    return [f"{p.year}q{p.quarter}" for p in periods]


def open_market_trades(start="2018-01-01", end=None, codes=("P",), verbose: bool = False) -> pd.DataFrame:
    """All open-market insider trades (default: purchases only) filed in
    [start, end], across every issuer, with the issuer's CURRENT ticker in
    `symbol` (NaN when the company no longer has an SEC ticker, which usually
    means it was delisted or acquired)."""
    links = quarter_links()
    out = []
    for key in _quarters(start, end):
        if key not in links:
            links = quarter_links(refresh=True)
            if key not in links:
                if verbose:
                    print(f"{key}: not published yet, skipped")
                continue
        q = load_quarter(key)
        if verbose:
            print(f"{key}: {len(q)} P/S rows")
        out.append(q)
    df = pd.concat(out, ignore_index=True)
    df = df[df["code"].isin(codes)]
    df = df[(df["filed"] >= pd.Timestamp(start)) & (df["filed"] <= pd.Timestamp(end or pd.Timestamp.today()))]
    return attach_current_tickers(df).reset_index(drop=True)


def attach_current_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Add `symbol`: the ticker SEC lists today for the issuer CIK, in Yahoo
    form (BRK.B -> BRK-B). Names that changed ticker map to the new one, which
    is what yfinance knows them by."""
    cik_to_ticker: dict[int, str] = {}
    for t, c in ticker_to_cik().items():
        # company_tickers.json lists the primary class first; keep that one.
        cik_to_ticker.setdefault(c, t)
    sym = df["issuer_cik"].map(cik_to_ticker)
    return df.assign(symbol=sym.str.replace(".", "-", regex=False))


def shares_outstanding(start="2017-01-01", end=None, refresh: bool = False) -> pd.DataFrame:
    """Shares outstanding for every SEC filer at each calendar quarter end,
    from the XBRL frames API (dei:EntityCommonStockSharesOutstanding, the
    cover-page share count). One request per quarter covers all companies.
    Columns: cik, period (quarter end), shares."""
    out = []
    for key in _quarters(start, end):
        year, q = key.split("q")
        path = cache_dir("edgar/frames") / f"shares_{key}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path)
        else:
            url = f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/CY{year}Q{q}I.json"
            try:
                rows = _get(url).json()["data"]
            except Exception:  # frame not built yet for the latest quarter
                continue
            df = pd.DataFrame(rows)[["cik", "val"]].rename(columns={"val": "shares"})
            df.to_csv(path, index=False)
        df["period"] = pd.Period(f"{year}Q{q}", freq="Q").end_time.normalize()
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    # Dual-class filers report one row per class; sum them.
    return df.groupby(["cik", "period"], as_index=False)["shares"].sum()


def add_market_cap(trades: pd.DataFrame, lag_days: int = 100) -> pd.DataFrame:
    """Add `mcap` = shares outstanding x the trade's own price.

    The share count is the latest quarter-end figure at least `lag_days`
    before the filing date, so it would have been published by then (10-Qs
    are due 40-45 days after quarter end, 10-Ks 60-90). No market price
    feed is needed, so this works for delisted names too."""
    so = shares_outstanding(trades["filed"].min() - pd.Timedelta(days=lag_days + 400), trades["filed"].max())
    so = so.rename(columns={"cik": "issuer_cik"}).sort_values("period")
    t = trades.assign(_asof=trades["filed"] - pd.Timedelta(days=lag_days), _i=range(len(trades)))
    t = t.sort_values("_asof")
    t["issuer_cik"] = t["issuer_cik"].astype("int64")
    so["issuer_cik"] = so["issuer_cik"].astype("int64")
    m = pd.merge_asof(t, so, left_on="_asof", right_on="period", by="issuer_cik",
                      tolerance=pd.Timedelta(days=550))
    m["mcap"] = m["shares_y"] * m["price"]
    m = m.sort_values("_i").rename(columns={"shares_x": "shares", "shares_y": "shares_out"})
    return m.drop(columns=["_asof", "_i", "period"]).reset_index(drop=True)
