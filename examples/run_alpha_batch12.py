"""Alpha hunt batch 12 (the last): upgrade or kill the near misses with parameters fixed elsewhere.

Each test prints sub-periods / holdouts so the verdicts in
docs/guide/strategies.md can be re-checked:

- Kalshi mention markets, taker No at 20-40c (band taken from the batch 11
  Polymarket result, so all Kalshi history is out of sample), with 40-60c and
  60-80c alongside; first taker-No print per market, t-stats clustered by
  event; robustness: skip the first hour, prints >= 10 contracts, 6 h-7 d
  before close;
- the same rule on other occurrence families (regexes fixed before running):
  appear / perform / attend, anytime scorer, rain / snow, posts "N or more",
  on Kalshi and Polymarket;
- January commodity window: BCOM/GSCI members vs futures in neither index
  (OJ, platinum, palladium, oats, rough rice), plus pre-2000 Januaries from
  FRED energy spot prices;
- single-stock LETF reversal on 2x ETFs launched after the batch 10 list,
  rule frozen from batches 10-11 (pressure >= 0.154 or > 0.02, move < -2x
  20-day vol).
"""
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from run_alpha_batch10 import LETF, tstat  # noqa: E402
from run_alpha_batch11 import pm_fills  # noqa: E402

from backtester.data.cache import cache_dir  # noqa: E402
from backtester.data.fred import load_series  # noqa: E402
from backtester.data.polymarket import load_markets  # noqa: E402
from backtester.event_markets import taker_prints  # noqa: E402

DIR = os.environ.get("KS_EVENT_DIR", "/mnt/project-files/event-markets")
# Occurrence markets only (YES = it happens at least once / at least N); range bins excluded. Fixed before running.
FAM = {"posts_count": (r"tweet|post|truth social", r"at least|or more|\d+\s*\+"),
       "appear_perform": (r"\b(appear|appears|perform|performs|attend|attends|show up)\b", None),
       "anytime_scorer": (r"anytime|to score a|score a (td|touchdown|goal)|scores? (a )?(td|touchdown|goal)", None),
       "rain_snow": (r"\brain\b|\bsnow\b|snowfall|precipitation", None)}
EXCL = r"between|less than|both teams|snowboard|snow white|\d+-\d+"
NEW_LETF = {"AVGO": "AVGU AVGX AVL", "NFLX": "NFXL NFLU", "RDDT": "RDTL RDTX", "IONQ": "IONX IONL", "RGTI": "RGTX RGTU",
            "OKLO": "OKLL OKLX", "SMR": "SMU SMUP", "CRWV": "CRWL CWVX CORW", "MU": "MUU", "ARM": "ARMG", "BABA": "BABX",
            "LLY": "LLYX", "UNH": "UNHG", "SOFI": "SOFX", "ASTS": "ASTX", "QBTS": "QBTX", "APP": "APPX", "CRCL": "CRCA",
            "HIMS": "HIMZ", "NBIS": "NBIL", "RIVN": "RVNL", "UBER": "UBRL", "BA": "BOEU", "AFRM": "AFRU", "SHOP": "SHPU",
            "NET": "NETX", "CRWD": "CRWU", "PANW": "PALU", "ORCL": "ORCX", "SNOW": "SNOU", "INTC": "LINT", "DJT": "DJTU",
            "ACHR": "ACHX", "RKLB": "RKLX", "SOUN": "SOUX", "U": "UXX", "LCID": "LCDL", "CVNA": "CVNX", "BMNR": "BMNU", "SBET": "SBTU"}
HALVES = (("before Jul 2025", None, "2025-07-01"), ("Jul 2025+", "2025-07-01", None))


def family(title: str) -> str:
    t = (title or "").lower()
    if re.search(EXCL, t):
        return ""
    for k, (a, b) in FAM.items():
        if re.search(a, t) and (b is None or re.search(b, t)):
            return k
    return ""


def clustered(name: str, x: pd.DataFrame, time="time", event="event"):
    out = []
    for lab, a, b in HALVES + (("all", None, None),):
        t = x[time].dt.tz_localize(None) if x[time].dt.tz is not None else x[time]
        s = x[(t >= (a or "1900")) & (t < (b or "2100"))]
        e = s.groupby(event).net.mean()
        out.append(f"{lab} {s.net.mean() * 100:+.2f}c n{len(s)} events {len(e)} t{tstat(e):.2f}")
    print(f"{name}: " + " | ".join(out))


def kalshi_prints() -> pd.DataFrame:
    """Taker prints (price paid, net after fee) for Kalshi mention markets and the occurrence families."""
    path = cache_dir("event_markets") / "batch12_prints.parquet"
    m = pd.read_parquet(f"{DIR}/nf_markets.parquet", columns=["ticker", "event_ticker", "title", "result", "close_time", "series", "cat"])
    m["fam"] = np.where(m.cat == "mentions", "mentions", "")
    other = m.cat != "mentions"
    m.loc[other, "fam"] = [family(t) for t in m.loc[other, "title"]]
    m = m[m.fam != ""]
    if not path.exists():
        ids, parts = pa.array(sorted(m.ticker)), []
        for f in sorted(glob.glob(f"{DIR}/raw/nf_trades_*.parquet")):
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(g, columns=["ticker", "count", "yes_price", "taker_side", "created_time"])
                parts.append(t.filter(pc.is_in(t["ticker"], value_set=ids)))
        pq.write_table(pa.concat_tables(parts), path)
    p = taker_prints(pd.read_parquet(path), m).merge(m[["ticker", "fam"]], on="ticker")
    return p.sort_values("created_time").rename(columns={"created_time": "time", "event_ticker": "event"})


def first_no(p: pd.DataFrame) -> pd.DataFrame:
    """First taker-No print per market with No at 20-80c (the price a No buyer really paid)."""
    return p[(p.taker_side == "no") & p.price.between(0.2, 0.8)].groupby("ticker").first().reset_index()


def kalshi_mentions(p: pd.DataFrame):
    m = p[p.fam == "mentions"]
    f = first_no(m)
    for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8001)):
        clustered(f"Kalshi mentions, taker No {lo:.1f}-{hi:.1f}", f[(f.price >= lo) & (f.price < hi)])
    opened = m.groupby("ticker").time.transform("min")
    for lab, q in (("skip first hour", m[m.time >= opened + pd.Timedelta(hours=1)]), ("prints >= 10 contracts", m[m["count"] >= 10]),
                   ("6 h-7 d before close", m[m.tau_h.between(6, 168)])):
        f = first_no(q)
        clustered(f"  No 20-40c, {lab}", f[(f.price >= 0.2) & (f.price < 0.4)])


def occurrence_families(p: pd.DataFrame):
    for fam in ("appear_perform", "anytime_scorer", "rain_snow"):
        f = first_no(p[p.fam == fam])
        clustered(f"Kalshi {fam}, taker No 20-40c", f[(f.price >= 0.2) & (f.price < 0.4)])
    mk = load_markets()
    mk = mk[(mk.volume >= 5000) & (mk.answer1 == "Yes") & ~mk.question.fillna("").str.contains(r"\bsay\b|\bmention", regex=True)]
    mk = mk.assign(id=mk.id.astype(str), fam=[family(q) for q in mk.question.fillna("")])
    mk = mk[(mk.fam != "") & (mk.closed == 1)]
    mk["no_won"] = mk.outcome_prices.astype(str).str.match(r"\['0")
    f = pm_fills(mk.id, "fills_batch12_families")
    f = f[(f.usd_amount >= 20) & (f.side == "SELL") & (1 - f.price).between(0.2, 0.4)].groupby("market_id").first().reset_index()
    f = f.merge(mk[["id", "fam", "event_slug", "no_won", "outcome_prices"]], left_on="market_id", right_on="id")
    f = f[f.outcome_prices.astype(str).str.match(r"\['[01]', '[01]'\]")]
    px = 1 - f.price
    f["net"] = f.no_won - px - 0.05 * px * (1 - px)
    for fam, g in f.groupby("fam"):
        clustered(f"Polymarket {fam}, taker No 20-40c", g, event="event_slug")
    clustered("Polymarket families pooled", f, event="event_slug")


def january(c: pd.DataFrame, a=5, b=10, cost=0.001, minn=4, frac=1 / 3) -> pd.Series:
    out = {}
    for y in range(c.index[0].year + 1, 2027):
        jan, prev = c.index[(c.index.year == y) & (c.index.month == 1)], c[c.index.year == y - 1]
        if len(jan) <= b or len(prev) < 200:
            continue
        perf = prev.ffill().iloc[-1] / prev.bfill().iloc[0] - 1
        ok = perf.notna() & c.loc[jan[a - 1]].notna() & c.loc[jan[b - 1]].notna()
        if ok.sum() < minn:
            continue
        q, r = perf[ok].rank(pct=True), c.loc[jan[b - 1]] / c.loc[jan[a - 1]] - 1
        out[y] = r[q[q > 1 - frac].index].mean() - r[q[q <= frac].index].mean() - cost
    return pd.Series(out)


def january_mechanism():
    groups = {"BCOM/GSCI members": ("CL=F NG=F HO=F RB=F GC=F SI=F HG=F ZC=F ZW=F KE=F ZS=F ZL=F ZM=F KC=F SB=F CT=F LE=F HE=F", 6),
              "in neither index": ("OJ=F PL=F PA=F ZO=F ZR=F", 4)}
    for name, (tk, minn) in groups.items():
        c = yf.download(tk.split(), start="1999-01-01", auto_adjust=True, progress=False)["Close"]
        c = c[c.index.dayofweek < 5]
        for a, b in ((5, 10), (1, 5), (10, 15)):
            s = january(c, a, b, minn=minn)
            print(f"January {name}, days {a + 1}-{b}: {s.mean() * 100:+.2f}% t{tstat(s):.2f} n{len(s)} positive {(s > 0).mean():.0%}"
                  f" | 2013+ {s.loc[2013:].mean() * 100:+.2f}%")
    ids = {"WTI": "DCOILWTICO", "Brent": "DCOILBRENTEU", "heating oil": "DHOILNYH", "gasoline": "DGASNYH", "natural gas": "DHHNGSP"}
    sp = pd.DataFrame({k: load_series(v) for k, v in ids.items()})
    sp = sp[sp.index.dayofweek < 5].ffill(limit=5)
    s = january(sp, minn=4, frac=0.4).loc[:1999]
    print(f"January FRED energy spot before 2000 (top 2 vs bottom 2 of 4-5), days 6-10: {s.mean() * 100:+.2f}% t{tstat(s):.2f} n{len(s)}")


def letf_new_names():
    tick = sorted(set(NEW_LETF) | {e for v in NEW_LETF.values() for e in v.split()} | set(LETF) | {s for v in LETF.values() for s in v} | {"SPY"})
    d = yf.download(tick, start="2022-01-01", auto_adjust=True, progress=False)
    c, o, v = d["Close"], d["Open"], d["Volume"]
    spy = c.SPY.shift(-1) / o.SPY.shift(-1) - 1

    def events(u, legs):
        legs = {s: lv for s, lv in legs.items() if s in c and c[s].notna().sum() >= 126}     # >= 6 months of history
        if not legs or u not in c or c[u].notna().sum() < 100:
            return None
        dv = sum(abs(lv * (lv - 1)) * (c[s] * v[s]).rolling(20, min_periods=5).mean().shift(1).fillna(0) for s, lv in legs.items())
        r = c[u].pct_change()
        e = pd.DataFrame({"u": u, "r": r, "sd": r.rolling(20).std().shift(1), "p": dv / (c[u] * v[u]).rolling(20).mean().shift(1),
                          "oc": c[u].shift(-1) / o[u].shift(-1) - 1 - spy - 0.0005}).dropna()
        return e[(e.r < -2 * e.sd) & (e.p > 0)]
    new = pd.concat([e for u, s in NEW_LETF.items() if (e := events(u, {x: 2 for x in s.split()})) is not None]).sort_index()
    old = pd.concat([e for u, lg in LETF.items() if (e := events(u, lg)) is not None]).sort_index()["2025-07-01":]
    for th in (0.154, 0.02):
        x = new[new.p >= th]
        print(f"New single-stock LETF names, pressure >= {th}: {x.oc.mean() * 1e4:+.0f}bp t{tstat(x.oc):.2f} n{len(x)} names {x.u.nunique()}"
              f" | lower pressure {new.oc[new.p < th].mean() * 1e4:+.0f}bp | original names Jul 2025+ {old.oc[old.p >= th].mean() * 1e4:+.0f}bp"
              f" t{tstat(old.oc[old.p >= th]):.2f}")


def main():
    p = kalshi_prints()
    kalshi_mentions(p)
    occurrence_families(p)
    january_mechanism()
    letf_new_names()


if __name__ == "__main__":
    main()
