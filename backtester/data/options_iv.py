"""Daily single-stock option summaries from Hugging Face `gauss314/options-IV-SP500`
(~3,900 US tickers incl. ETFs, 2019-10-14 to 2023-07-28, 3.2M rows, 520 MB CSV).

Per symbol and day: IV by moneyness bucket (DITM..DOTM, ATM), call/put contracts
traded and open interest, expiry count, historical vol 20..200d and the VIX.
It is an end-of-day snapshot, so a signal built from day t trades at the open of
t+1. The CSV is cached as parquet under `options_iv/`. The moneyness buckets
pool calls and puts, so they give a smile width, not a put-skew.
"""
import pandas as pd

from .cache import cache_dir

URL = "https://huggingface.co/datasets/gauss314/options-IV-SP500/resolve/main/data_IV_USA.csv"
IV_COLS = ["DITM_IV", "ITM_IV", "sITM_IV", "ATM_IV", "sOTM_IV", "OTM_IV", "DOTM_IV"]
HV_COLS = [f"hv_{n}" for n in (20, 40, 60, 75, 90, 120, 180, 200)]


def load_iv(refresh: bool = False) -> pd.DataFrame:
    """Long frame: symbol, date, IV buckets, volumes, open interest, hv_*, VIX.
    `hv_*` outliers (data errors above 500) are set to NaN."""
    path = cache_dir("options_iv") / "iv.parquet"
    if refresh or not path.exists():
        import requests

        csv = cache_dir("options_iv") / "data_IV_USA.csv"
        with requests.get(URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(csv, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        d = pd.read_csv(csv, index_col=0, parse_dates=["date"])
        d.to_parquet(path)
        csv.unlink()
    d = pd.read_parquet(path)
    for c in HV_COLS:
        d.loc[d[c] > 500, c] = float("nan")
    return d
