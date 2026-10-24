"""Alpha hunt batch 4: daily-data and crypto hypotheses (all failed).

Each prints in-sample sub-periods and the Jul 2024+ holdout, so the verdicts
in docs/guide/strategies.md can be re-checked:

- put-write: Cboe ^PUT and a 10% vol-targeted version vs ^SP500TR; XYLD, the
  covered-call ETF a $5k account can hold, vs SPY (SPX puts need ~$600k);
- macro days: SPY on FOMC statement days and NFP (first Friday) days;
- FOMC even-week cycle (Cieslak-Morse-Vissing-Jorgensen);
- GLD overnight (close to open) vs intraday, with round-trip costs;
- 1.5x / vol-targeted USMV vs SPY;
- BTC/ETH perp-discount reversal: buy after the Binance premium index falls
  below its trailing 1-year 1st percentile, hold 48h, 20 bp round trip
  (HF linxy/USDT-M_Perpetual_Futures, 1h, 2020-2026).
"""
import numpy as np
import pandas as pd
import yfinance as yf

from backtester.data.cache import cache_dir
from backtester.data.fomc import statement_dates

SPLITS = [("2008", "2016"), ("2017", "2024-06"), ("2024-07", "2026")]
HF = "https://huggingface.co/datasets/linxy/USDT-M_Perpetual_Futures/resolve/main/{s}/{s}_{k}_1h.parquet"


def stats(r: pd.Series) -> str:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return f"{eq.iloc[-1] ** (252 / len(r)) - 1:6.1%} Sh {r.mean() / r.std() * np.sqrt(252):5.2f} DD {(eq / eq.cummax() - 1).min():5.0%}"


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x))


def vol_target(r: pd.Series, target=0.10, cap=1.5, cost=0.0005) -> pd.Series:
    w = (target / (r.rolling(20).std() * np.sqrt(252))).clip(upper=cap).shift(1)
    return w * r - w.diff().abs() * cost


def put_write(px):
    r = px["Close"][["^PUT", "^SP500TR", "XYLD", "SPY"]].pct_change(fill_method=None)
    for lo, hi in [("1997", "2024-06"), ("2024-07", "2026")]:
        print(f"PUT {lo}-{hi}: {stats(r['^PUT'][lo:hi])} | VT {stats(vol_target(r['^PUT'].dropna())[lo:hi])} | SPX TR {stats(r['^SP500TR'][lo:hi])}")
        x = r[["XYLD", "SPY"]].dropna()[max(lo, "2014"):hi]
        print(f"  XYLD {stats(x['XYLD'])} | VT XYLD {stats(vol_target(x['XYLD']))} | SPY {stats(x['SPY'])}")


def macro_days(px):
    spy = px["Close"]["SPY"].dropna()
    idx, r = spy.index, spy.pct_change()
    fomc = pd.DatetimeIndex(statement_dates())
    nfp = []
    for d in pd.date_range("2008-01-01", idx[-1], freq="MS"):
        nfp.append(d + pd.Timedelta(days=(4 - d.weekday()) % 7))
    for name, days in [("FOMC", fomc), ("NFP", pd.DatetimeIndex(nfp))]:
        on = pd.Series(idx.isin(days), index=idx)
        print(name, " | ".join(f"{lo}-{hi} {r[lo:hi][on[lo:hi]].mean() * 1e4:5.1f}bp (t {tstat(r[lo:hi][on[lo:hi]]):.1f})" for lo, hi in SPLITS))
    pos = pd.Series(np.arange(len(idx)), index=idx)
    week = (pos - pos.where(idx.isin(fomc)).ffill() + 1) // 5
    print("FOMC even-minus-odd weeks", " | ".join(
        f"{lo}-{hi} {(r[lo:hi][week[lo:hi].isin([0, 2, 4, 6])].mean() - r[lo:hi][week[lo:hi].isin([1, 3, 5])].mean()) * 1e4:5.1f}bp"
        for lo, hi in SPLITS))


def gold_overnight(px):
    c, o = px["Close"]["GLD"].dropna(), px["Open"]["GLD"].dropna()
    night, day = o / c.shift(1) - 1, c / o - 1
    for lo, hi in [("2005", "2012"), ("2013", "2018"), ("2019", "2024-06"), ("2024-07", "2026")]:
        print(f"GLD {lo}-{hi}: overnight {night[lo:hi].mean() * 252:6.1%}/yr, intraday {day[lo:hi].mean() * 252:6.1%}/yr, "
              f"overnight net of 8 bp/day {(night[lo:hi].mean() - 0.0008) * 252:6.1%}/yr")


def low_vol(px):
    r = px["Close"][["SPY", "USMV", "BIL"]].dropna().pct_change().dropna()
    lev = 1.5 * r["USMV"] - 0.5 * (r["BIL"] + 0.015 / 252)
    for lo, hi in [("2012", "2024-06"), ("2024-07", "2026")]:
        print(f"{lo}-{hi}: SPY {stats(r['SPY'][lo:hi])} | USMV {stats(r['USMV'][lo:hi])} | 1.5x USMV {stats(lev[lo:hi])}")


def perp_discount(hold=48, pct=0.01, cost=0.002):
    for s in ("BTCUSDT", "ETHUSDT"):
        frames = {}
        for k in ("premiumIndex", "indexPrice"):
            path = cache_dir("linxy") / f"{s}_{k}_1h.parquet"
            if not path.exists():
                path.write_bytes(__import__("requests").get(HF.format(s=s, k=k), timeout=300).content)
            f = pd.read_parquet(path, columns=["open_time", "close"]).set_index("open_time")["close"].astype(float)
            frames[k] = f[~f.index.duplicated()]
        df = pd.DataFrame(frames).dropna().asfreq("h").ffill(limit=3)
        thr = df["premiumIndex"].rolling(8760, min_periods=2000).quantile(pct).shift(1)
        sig = np.flatnonzero((df["premiumIndex"] < thr).to_numpy())
        rows, last = {}, -hold
        for p in sig:  # enter at the close of the next hour, one trade per hold window
            if p - last < hold or p + 1 + hold >= len(df):
                continue
            last = p
            rows[df.index[p]] = df["indexPrice"].iloc[p + 1 + hold] / df["indexPrice"].iloc[p + 1] - 1 - cost
        e = pd.Series(rows)
        print(s, " | ".join(f"{lo}-{hi} {e[lo:hi].mean() * 1e4:5.0f}bp (t {tstat(e[lo:hi]):.1f}, n {len(e[lo:hi])})"
                            for lo, hi in [("2020", "2021"), ("2022", "2024-06"), ("2024-07", "2026")]))


def main():
    px = yf.download("^PUT ^SP500TR XYLD SPY GLD USMV BIL".split(), start="1996-01-01", auto_adjust=True, progress=False)
    put_write(px)
    macro_days(px)
    gold_overnight(px)
    low_vol(px)
    perp_discount()


if __name__ == "__main__":
    main()
