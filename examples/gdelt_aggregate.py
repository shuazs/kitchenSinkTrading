"""Stream GDELT 1.0 daily event files (public S3) into small country x CAMEO-root daily aggregates.

gdelt-open-data.s3.amazonaws.com/events/YYYYMMDD.export.csv exists 2013-04-01..2019-09-18 (35-70 MB each,
tab separated, 58 columns). Rows are grouped by DATEADDED (the day GDELT published the event, so no
look-ahead from late-reported old events), ActionGeo FIPS country and CAMEO root code:
n events, mentions, mention-weighted tone and Goldstein. Output: /mnt/project-files/gdelt/daily_agg.parquet.
Run time ~30 min with 4 workers; resumable (per-day parts cached).
"""
from __future__ import annotations

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL = "https://gdelt-open-data.s3.amazonaws.com/events/{}.export.csv"
OUT = Path("/mnt/project-files/gdelt")
PARTS = Path("/tmp/gdelt_parts")
COLS = {1: "sqldate", 28: "root", 29: "quad", 30: "gold", 31: "ment", 34: "tone", 51: "country", 56: "added"}


def one(day: str):
    f = PARTS / f"{day}.parquet"
    if f.exists():
        return day, True
    r = None
    for _ in range(4):
        try:
            r = requests.get(URL.format(day), timeout=180)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
    if r is None or r.status_code != 200:
        return day, False
    df = pd.read_csv(io.BytesIO(r.content), sep="\t", header=None, usecols=list(COLS), names=None, dtype=str, on_bad_lines="skip")
    df.columns = [COLS[c] for c in sorted(COLS)]
    for c in ("gold", "ment", "tone"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["w_tone"] = df.tone * df.ment
    df["w_gold"] = df.gold * df.ment
    g = df.groupby(["added", "country", "root"], dropna=False).agg(n=("ment", "size"), ment=("ment", "sum"), w_tone=("w_tone", "sum"), w_gold=("w_gold", "sum")).reset_index()
    g.to_parquet(f)
    return day, True


def main(start="2013-04-01", end="2019-09-18", workers=4):
    PARTS.mkdir(exist_ok=True)
    days = [d.strftime("%Y%m%d") for d in pd.date_range(start, end)]
    bad = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (d, ok) in enumerate(ex.map(one, days)):
            if not ok:
                bad.append(d)
            if i % 100 == 0:
                print(i, d, len(bad), flush=True)
    agg = pd.concat([pd.read_parquet(PARTS / f"{d}.parquet") for d in days if (PARTS / f"{d}.parquet").exists()])
    agg = agg.groupby(["added", "country", "root"], dropna=False).sum().reset_index()
    OUT.mkdir(exist_ok=True)
    agg.to_parquet(OUT / "daily_agg.parquet")
    print("rows", len(agg), "missing days", len(bad), bad[:10])


if __name__ == "__main__":
    main(*sys.argv[1:3]) if len(sys.argv) > 2 else main()
