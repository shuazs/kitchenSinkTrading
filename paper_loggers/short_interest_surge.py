"""Paper-trade the short-interest-surge short; logs only.

    python paper_loggers/short_interest_surge.py            # daily: form new baskets, score finished ones
    python paper_loggers/short_interest_surge.py --report   # top decile vs universe so far

Rule (examples/run_short_interest.py): each FINRA short-interest settlement is
usable ~10 business days later (formation date F). On NYSE/Nasdaq common
stocks with average daily volume >= 300k shares, no split flag and price >= $5,
the top decile by short-interest change (%) is shorted equal weight from the
open of F to the open of the next F, against the equal-weight universe. The
Nov 2025 - Sep 2026 backtest: top decile -1.1%/period vs the universe in the
first half (t -1.0) and -2.8% in the second (t -4.6), before borrow fees.

Data: HF FinzData/finra-short-interest (no key) and Yahoo. To keep downloads
small, each period scores the basket against a fixed random 400-name sample
of the universe. Borrow fees are not modelled: fill `borrow_pct` (annual %,
e.g. from your broker's short-locate screen) in baskets.csv for a few names
per period; `--report` subtracts the average. Logs in
~/.cache/kitchensink/paper/short_interest_surge/. Graduates after 12 periods
(~6 months) at <= -1.4% per period net of borrow.
"""
import argparse

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from _common import closes, log_dir, read_log, today, tstat, write_log

URL = "https://huggingface.co/datasets/FinzData/finra-short-interest/resolve/main/data/short_interest.parquet"
BAD = r"ETF|Fund|Trust|Notes|Warrant|Units|Rights|ETN|Shares"
DIR = log_dir("short_interest_surge")
PERIODS, BASKETS = DIR / "periods.csv", DIR / "baskets.csv"
P_FIELDS = ["settlement", "formed", "next_formed", "n_top", "n_sample", "top_ret", "sample_ret", "excess"]
B_FIELDS = ["settlement", "symbol", "role", "change_pct", "days_to_cover", "borrow_pct"]
SAMPLE, BAR = 400, -0.014


def universe() -> pd.DataFrame:
    si = pd.read_parquet(URL)
    si = si[~si["issuer"].str.contains(BAD, case=False, na=False) & (si["avg_daily_volume"] >= 300000)
            & si["market"].isin(["NYSE", "NNM", "SC"]) & si["split_flag"].isna()].copy()
    si["symbol"] = si["ticker"].str.replace(".", "-", regex=False)
    si["settlement"] = pd.to_datetime(si["settlement_date"])
    return si


def form(si: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Baskets for settlements whose formation date has come and that are not logged yet."""
    done = set(pd.to_datetime(b["settlement"]))
    rows = []
    for d, s in si.groupby("settlement"):
        if d in done or d + BDay(10) > today() or d < today() - pd.Timedelta(days=60):
            continue
        dec = pd.qcut(s["change_pct"].rank(method="first"), 10, labels=False)
        top = s[dec == 9]
        rest = s.drop(top.index).sample(min(SAMPLE, len(s) - len(top)), random_state=int(d.strftime("%Y%m%d")))
        for role, g in (("top", top), ("sample", rest)):
            rows += [dict(settlement=d.date(), symbol=r.symbol, role=role, change_pct=r.change_pct,
                          days_to_cover=r.days_to_cover) for r in g.itertuples()]
        print(f"{d:%Y-%m-%d} settlement -> paper SHORT {len(top)} names from the open of {(d + BDay(10)):%Y-%m-%d}: "
              f"{', '.join(top.sort_values('change_pct', ascending=False)['symbol'].head(15))} ...")
    return pd.concat([b, pd.DataFrame(rows)], ignore_index=True) if rows else b


def score(b: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    sets = sorted(pd.to_datetime(b["settlement"]).unique())
    done = set(pd.to_datetime(p["settlement"]))
    for d0, d1 in zip(sets[:-1], sets[1:]):
        f0, f1 = d0 + BDay(10), d1 + BDay(10)
        if d0 in done or f1 > today():
            continue
        g = b[pd.to_datetime(b["settlement"]) == d0]
        o = closes(g["symbol"], f0 - pd.Timedelta(days=10), "Open")
        c = closes(g["symbol"], f0 - pd.Timedelta(days=10))
        i0, i1 = o.index.searchsorted(f0), o.index.searchsorted(f1)
        if i1 >= len(o):
            continue
        ok = c.iloc[i0 - 1] >= 5  # price filter at the prior close
        r = (o.iloc[i1] / o.iloc[i0] - 1)[ok].dropna()
        top, smp = r.reindex(g.loc[g.role == "top", "symbol"]).dropna(), r.reindex(g.loc[g.role == "sample", "symbol"]).dropna()
        row = dict(settlement=d0.date(), formed=o.index[i0].date(), next_formed=o.index[i1].date(), n_top=len(top),
                   n_sample=len(smp), top_ret=round(top.mean(), 5), sample_ret=round(smp.mean(), 5),
                   excess=round(top.mean() - smp.mean(), 5))
        p = pd.concat([p, pd.DataFrame([row])], ignore_index=True)
        print(f"scored {row}")
    return p


def run() -> None:
    b, p = read_log(BASKETS, B_FIELDS), read_log(PERIODS, P_FIELDS)
    b = form(universe(), b)
    write_log(b, BASKETS)
    write_log(score(b, p), PERIODS)


def report() -> None:
    p, b = read_log(PERIODS, P_FIELDS), read_log(BASKETS, B_FIELDS)
    ex = pd.to_numeric(p["excess"], errors="coerce").dropna()
    borrow = pd.to_numeric(b["borrow_pct"], errors="coerce").dropna()
    fee = borrow.mean() / 100 / 24 if len(borrow) else np.nan  # ~24 periods a year
    short_pnl = -ex - (0 if np.isnan(fee) else fee) - 0.004
    print(f"{len(ex)} periods scored: top decile vs sample {ex.mean() * 100:+.2f}% per period (t {tstat(ex):.2f}); "
          f"avg logged borrow {borrow.mean() if len(borrow) else float('nan'):.1f}%/yr -> short P&L net of borrow and "
          f"40 bp costs {short_pnl.mean() * 100:+.2f}% per period")
    print(f"Backtest -1.1% / -2.8% per period; graduates after 12 periods if the excess stays <= {BAR * 100:.1f}% net.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    report() if ap.parse_args().report else run()
