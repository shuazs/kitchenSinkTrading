"""Fundamentals and insider filings from SEC EDGAR.

The SEC requires a descriptive User-Agent with contact details. Set the
SEC_USER_AGENT environment variable, e.g. "Josh research josh@example.com".
All dates here are filing dates, so signals only use what was public.
"""
from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .cache import cache_dir

_last_call = [0.0]


def _get(url: str) -> requests.Response:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("Set SEC_USER_AGENT to 'name email' before calling EDGAR")
    # SEC asks for at most 10 requests per second.
    wait = 0.11 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    return r


def _cached_json(name: str, url: str, refresh: bool) -> dict:
    path = cache_dir("edgar") / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = _get(url).json()
    path.write_text(json.dumps(data))
    return data


def ticker_to_cik(refresh: bool = False) -> dict[str, int]:
    data = _cached_json("company_tickers.json", "https://www.sec.gov/files/company_tickers.json", refresh)
    return {v["ticker"]: int(v["cik_str"]) for v in data.values()}


def company_facts(ticker: str, refresh: bool = False) -> dict:
    cik = ticker_to_cik()[ticker.upper()]
    return _cached_json(f"facts_{cik}.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", refresh)


def fact_series(ticker: str, concept: str = "EarningsPerShareDiluted", taxonomy: str = "us-gaap", form: str = "10-Q") -> pd.DataFrame:
    """One XBRL concept as rows of (period end, value, filed date).

    Duplicates from later amendments are dropped so each period keeps its
    first-filed value, which is what the market saw at the time.
    """
    facts = company_facts(ticker)["facts"][taxonomy][concept]["units"]
    unit = next(iter(facts))
    df = pd.DataFrame(facts[unit])
    df = df[df["form"].str.startswith(form[:4])] if form else df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"])
        days = (df["end"] - df["start"]).dt.days
        df = df[(days < 100) | days.isna()]  # quarterly values only
    df = df.sort_values("filed").drop_duplicates("end", keep="first")
    return df[["end", "val", "filed", "form"]].rename(columns={"val": concept}).reset_index(drop=True)


def earnings_surprise(ticker: str, concept: str = "EarningsPerShareDiluted") -> pd.DataFrame:
    """Standardized unexpected earnings (SUE) from a seasonal random walk:
    (EPS_q - EPS_{q-4}) / std of that difference over the prior 8 quarters."""
    df = fact_series(ticker, concept)
    eps = df[concept]
    diff = eps - eps.shift(4)
    df["sue"] = diff / diff.shift(1).rolling(8, min_periods=4).std()
    return df.dropna(subset=["sue"])


def insider_filings(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """List of Form 4 filings for a company (accession numbers and dates)."""
    cik = ticker_to_cik()[ticker.upper()]
    sub = _cached_json(f"subs_{cik}.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json", refresh)
    recent = pd.DataFrame(sub["filings"]["recent"])
    f4 = recent[recent["form"] == "4"].copy()
    f4["cik"] = cik
    f4["filingDate"] = pd.to_datetime(f4["filingDate"])
    return f4[["cik", "accessionNumber", "filingDate", "primaryDocument"]]


def parse_form4(cik: int, accession: str, primary_doc: str) -> pd.DataFrame:
    """Open-market transactions (code P = purchase, S = sale) from one Form 4."""
    acc = accession.replace("-", "")
    doc = primary_doc.split("/")[-1]  # strip the xsl rendering folder
    path = cache_dir("edgar/form4") / f"{acc}.xml"
    if path.exists():
        text = path.read_text()
    else:
        text = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}").text
        path.write_text(text)
    root = ET.fromstring(text)
    owner = root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName")
    is_officer = root.findtext(".//reportingOwnerRelationship/isOfficer") in ("1", "true")
    is_director = root.findtext(".//reportingOwnerRelationship/isDirector") in ("1", "true")
    rows = []
    for t in root.findall(".//nonDerivativeTransaction"):
        code = t.findtext(".//transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        rows.append(
            {
                "owner": owner,
                "officer": is_officer,
                "director": is_director,
                "date": pd.to_datetime(t.findtext(".//transactionDate/value")),
                "code": code,
                "shares": float(t.findtext(".//transactionShares/value") or 0),
                "price": float(t.findtext(".//transactionPricePerShare/value") or 0),
            }
        )
    return pd.DataFrame(rows)


def insider_trades(ticker: str, since: str = "2015-01-01") -> pd.DataFrame:
    """All open-market insider buys and sells with their filing date."""
    filings = insider_filings(ticker)
    filings = filings[filings["filingDate"] >= since]
    out = []
    for f in filings.itertuples():
        tx = parse_form4(f.cik, f.accessionNumber, f.primaryDocument)
        if len(tx):
            tx["filed"] = f.filingDate
            out.append(tx)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
