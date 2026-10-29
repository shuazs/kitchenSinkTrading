"""USDA WASDE report numbers from Hugging Face `syedzaidi-kiwi/US-Wasde`
(USDA's machine-readable WASDE tables, Apr 2010 - Apr 2024, with release
dates; usda.gov and quickstats are blocked from the cloud). WASDE is
released at 12:00 ET on a pre-announced day, inside the grain session.
"""
import pandas as pd

from .cache import cache_dir

URL = "https://huggingface.co/datasets/syedzaidi-kiwi/US-Wasde/resolve/main/merged_data.csv"
COLS = ["WasdeNumber", "ReportTitle", "Attribute", "Commodity", "Region", "MarketYear", "ProjEstFlag", "Value",
        "ReleaseDate"]


def load_wasde(refresh: bool = False) -> pd.DataFrame:
    """One row per WASDE cell; `release` is the report's release date. The
    133 MB CSV is downloaded once and kept as a slim parquet."""
    path = cache_dir("wasde") / "wasde.parquet"
    if refresh or not path.exists():
        import requests

        raw = cache_dir("wasde") / "merged_data.csv"
        with requests.get(URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(raw, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        df = pd.read_csv(raw, usecols=COLS, low_memory=False)
        df["release"] = pd.to_datetime(df.pop("ReleaseDate"), format="mixed")
        df.to_parquet(path)
        raw.unlink()
    return pd.read_parquet(path)
