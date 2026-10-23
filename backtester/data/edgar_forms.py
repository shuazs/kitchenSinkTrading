"""EDGAR quarterly form index and filer status, for structural-event tests
(spin-offs via Form 10-12B, tender offers via SC TO-T).

`filer_status` says whether a registrant became a reporting company and,
if it has no ticker today, whether it was acquired or went bankrupt; that is
what bounds survivorship when Yahoo has no prices for dead tickers.
"""
from __future__ import annotations

import re

import pandas as pd

from .cache import cache_dir
from .edgar import _get

_LINE = re.compile(r"(.*?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)")
_ACQUIRED = {"DEFM14A", "PREM14A", "DEFM14C", "SC 14D9", "SC 14D9/A", "SC TO-T"}


def parse_form_index(text: str, forms: tuple[str, ...]) -> pd.DataFrame:
    """Rows of one form.idx file whose form type is in `forms`."""
    rows = []
    for line in text.splitlines():
        form = line[:12].strip()
        if form in forms:
            m = _LINE.match(line[12:])
            if m:
                rows.append((form, m.group(1).strip(), int(m.group(2)), m.group(3), m.group(4)))
    return pd.DataFrame(rows, columns=["form", "name", "cik", "date", "path"])


def form_index(forms=("10-12B", "10-12B/A"), start_year: int = 2005, end_year: int = 2026) -> pd.DataFrame:
    """Every filing of `forms` in EDGAR's quarterly full index (cached per quarter)."""
    out = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            path = cache_dir("edgar/form_index") / f"{y}q{q}.idx"
            if not path.exists():
                try:
                    path.write_text(_get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx").text)
                except Exception:
                    continue
            out.append(parse_form_index(path.read_text(errors="ignore"), tuple(forms)))
    return pd.concat(out, ignore_index=True) if out else parse_form_index("", tuple(forms))


def classify_filings(filings: pd.DataFrame, since: str) -> dict:
    """Status from a filer's filing list (columns form, date, items):
    periodic reports after `since`, and acquired / bankrupt flags."""
    later = filings[filings["date"] > since]
    eightk = filings.loc[filings["form"] == "8-K", "items"].astype(str)
    return {
        "n_periodic": int(later["form"].isin(["10-K", "10-Q"]).sum()),
        "first_periodic": later.loc[later["form"].isin(["10-K", "10-Q"]), "date"].min(),
        "acquired": bool(filings["form"].isin(_ACQUIRED).any()),
        "bankrupt": bool(eightk.str.contains("1.03").any()),
    }


def filer_status(cik: int, since: str) -> dict:
    """`classify_filings` for one CIK from data.sec.gov submissions, plus its current tickers."""
    j = _get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json").json()
    rec = j["filings"]["recent"]
    f = pd.DataFrame({"form": rec["form"], "date": rec["filingDate"], "items": rec.get("items", [""] * len(rec["form"]))})
    return {"cik": cik, "tickers": ",".join(j.get("tickers", [])), **classify_filings(f, since)}
