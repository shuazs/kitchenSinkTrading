"""Batch 5: GDELT 1.0 country-day conflict / tone shocks vs oil, gold, SPY and country ETFs (2013-04..2019-09).

Needs /mnt/project-files/gdelt/daily_agg.parquet from gdelt_aggregate.py. Shock = z-score of the day's
conflict-event share (CAMEO roots 14,15,17-20, mention weighted) vs the previous 90 days (shifted 1).
GDELT day t (DATEADDED, UTC) is complete at ~8pm ET, so entry is the close of t+1. Events closer than
5 days to the previous one are dropped. Forward return is excess over the asset's mean forward return.
Design 2013-04..2016-12, holdout 2017-01..2019-09 (GDELT daily files end 2019-09-18, so no recent holdout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

AGG = "/mnt/project-files/gdelt/daily_agg.parquet"
CONFLICT = {"14", "15", "17", "18", "19", "20"}
OIL = ["IZ", "IR", "SA", "LY", "NI", "VE", "RS", "UP", "AG", "KU", "AE", "EG", "YM", "SY"]
CTRY_ETF = {"CH": "FXI", "JA": "EWJ", "IN": "INDA", "BR": "EWZ", "TU": "TUR", "IS": "EIS", "MX": "EWW", "KS": "EWY",
            "TW": "EWT", "SF": "EZA", "AS": "EWA", "UK": "EWU", "GM": "EWG", "FR": "EWQ", "IT": "EWI", "SP": "EWP", "RP": "EPHE", "ID": "EIDO"}
SPLIT = "2017-01-01"
HOLD = (1, 3, 5, 10)


def load():
    a = pd.read_parquet(AGG)
    a["root"] = a["root"].astype(str).str.zfill(2)
    a["added"] = pd.to_datetime(a["added"].astype(str), format="%Y%m%d", errors="coerce")
    a = a.dropna(subset=["added"])
    a = a[(a.added >= "2013-04-02") & (a.added <= "2019-09-17")]
    return a


def zshock(s: pd.Series, win=90):
    m, sd = s.rolling(win).mean().shift(1), s.rolling(win).std().shift(1)
    return (s - m) / sd


def events_from(z: pd.Series, thr: float, sign: int = 1, gap=5):
    hits = z[(sign * z) > thr].index
    keep, last = [], None
    for t in hits:
        if last is None or (t - last).days >= gap:
            keep.append(t)
            last = t
    return keep


def study(px: pd.Series, ev, label, rows, bench: pd.Series | None = None):
    px = px.dropna()
    r1 = px.pct_change()
    if bench is not None:
        r1 = r1 - bench.pct_change().reindex(r1.index)
    for h in HOLD:
        fwd = (1 + r1).rolling(h).apply(np.prod, raw=True).shift(-h) - 1  # return from close t to t+h
        # entry close t+1: shift by one more day
        fwd_e = fwd.shift(-1)
        base = fwd_e.mean()
        for name, sel in (("design", [t for t in ev if t < pd.Timestamp(SPLIT)]), ("holdout", [t for t in ev if t >= pd.Timestamp(SPLIT)])):
            idx = [px.index[px.index.searchsorted(t)] for t in sel if px.index.searchsorted(t) < len(px)]
            x = fwd_e.reindex(idx).dropna() - base
            if len(x) >= 5:
                rows.append(dict(test=label, h=h, per=name, n=len(x), mean_pct=100 * x.mean(), t=x.mean() / x.std() * np.sqrt(len(x))))


def main():
    a = load()
    a["conf"] = a.root.isin(CONFLICT)
    tot = a.groupby("added").ment.sum()
    conf_tot = a[a.conf].groupby("added").ment.sum() / tot
    tone = a.groupby("added").w_tone.sum() / tot
    oil_c = a[a.conf & a.country.isin(OIL)].groupby("added").ment.sum() / tot
    syms = ["USO", "GLD", "SPY", "ITA"] + sorted(set(CTRY_ETF.values()))
    px = yf.download(syms + ["CL=F"], start="2013-01-01", end="2019-12-31", auto_adjust=True, progress=False, threads=False)["Close"]
    rows = []
    for thr in (1.5, 2.0, 3.0):
        for sign in (1, -1):
            suf = f"z{'>' if sign > 0 else '<-'}{thr}"
            study(px["CL=F"], events_from(zshock(oil_c), thr, sign), f"oil producers conflict {suf} -> CL=F", rows)
            study(px["USO"], events_from(zshock(oil_c), thr, sign), f"oil producers conflict {suf} -> USO", rows)
            study(px["GLD"], events_from(zshock(conf_tot), thr, sign), f"global conflict share {suf} -> GLD", rows)
            study(px["ITA"], events_from(zshock(conf_tot), thr, sign), f"global conflict share {suf} -> ITA-SPY", rows, bench=px["SPY"])
            study(px["SPY"], events_from(zshock(tone), thr, sign), f"global tone {suf} -> SPY", rows)
        for c, etf in CTRY_ETF.items():
            s = a[a.conf & (a.country == c)].groupby("added").ment.sum() / tot
            for sign in (1,):
                study(px[etf], events_from(zshock(s), thr, sign), f"{c} conflict z>{thr} -> {etf}-SPY", rows, bench=px["SPY"])
    g = pd.DataFrame(rows)
    g.to_csv("/mnt/project-files/gdelt/batch5_grid.csv", index=False)
    pv = g.pivot_table(index=["test", "h"], columns="per", values=["mean_pct", "t", "n"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.dropna()
    print(len(pv), "test rows;", g.test.nunique(), "tests")
    both = pv[(pv.t_design > 2) & (pv.t_holdout > 2)]
    print("t>2 in both periods:\n", both.round(2).to_string())
    print("\nlargest design |t|:\n", pv.reindex(pv.t_design.abs().sort_values(ascending=False).index).head(15).round(2).to_string())
    print("\nsign-consistent share:", (np.sign(pv.mean_pct_design) == np.sign(pv.mean_pct_holdout)).mean().round(2))


if __name__ == "__main__":
    main()
