"""Commodity futures study (2026-10-29): 26 pre-registered ideas on grains,
energy and metals for a $5k account, plus the stress tests of the one lead.

    python examples/run_commodity_futures.py            # everything (~3 min after the first download)
    python examples/run_commodity_futures.py storage    # one section: carry basis grains wasde usda energy pairs misc calendar storage

Design data ends 2024-06-30; Jul 2024+ is the holdout (docs/guide/protocol.md).
Returns are commodity ETFs, open-to-open with the weight decided at the prior
close (signal at close t, trade at open t+1), net of 4 bp per side (10 bp for
the thinner CORN SOYB WEAT CANE CPER BNO UGA). Yahoo `=F` front-month series
(unadjusted) feed signals and event-day returns only. Verdicts are in
docs/guide/strategies.md ("Commodity futures").
"""
import sys

import numpy as np
import pandas as pd

from backtester.commodities import (ETF_ROOT, expiry_mask, ratio_positions, realized_carry, season_returns,
                                    storage_day_returns, storage_report_days, wasde_revisions)
from backtester.data.fred import load_series
from backtester.data.yahoo import load_bars

HOLD = "2024-07-01"
ETFS = list(ETF_ROOT)
COST = {e: 4e-4 for e in ETFS} | {e: 1e-3 for e in ["CORN", "SOYB", "WEAT", "CANE", "CPER", "BNO", "UGA"]}


def panel(syms, field="close"):
    return pd.DataFrame({s: load_bars(s)[field] for s in syms}).sort_index()


def sim(w: pd.DataFrame, cost=None) -> pd.Series:
    """Daily net return of weights decided at each close, held open(t+1) -> open(t+2)."""
    o = panel(list(w.columns), "open").reindex(w.index)
    held = w.fillna(0).shift(1).fillna(0)
    gross = (held * (o.shift(-1) / o - 1).fillna(0)).sum(axis=1)
    c = pd.Series(cost or COST).reindex(w.columns).fillna(4e-4)
    return gross - (held.diff().abs().fillna(held.abs()) * c).sum(axis=1)


def stats(r: pd.Series) -> str:
    r = r.dropna()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return f"{r.mean() * 252:6.1%}/yr Sharpe {sh:5.2f} DD {dd:6.1%} t {r.mean() / r.std() * np.sqrt(len(r)):5.2f}"


def report(name: str, r: pd.Series, bench: pd.Series | None = None) -> None:
    r = r.loc[r.ne(0).idxmax():]
    d = r[:"2024-06-30"]
    thirds = [d.iloc[i] for i in np.array_split(np.arange(len(d)), 3)]
    print(f"  {name:28s} design {stats(d)} | holdout {stats(r[HOLD:])}")
    print(f"  {'':28s} thirds Sharpe " + "  ".join(f"{p.index[0].year}-{p.index[-1].year} {p.mean() / p.std() * 15.87:5.2f}" for p in thirds))
    if bench is not None:
        x = (r - bench.reindex(r.index)).dropna()
        print(f"  {'  minus benchmark':28s} design {stats(x[:'2024-06-30'])} | holdout {stats(x[HOLD:])}")


def ev(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean() * 1e4:6.1f} bp t {x.mean() / x.std() * np.sqrt(len(x)):5.2f} n {len(x):4d}"


def month_ends(idx):
    return idx.to_series().groupby(idx.to_period("M")).last().values


def basket(signal: pd.DataFrame, top: int, ascending=False, min_n: int = 8):
    """Equal weight in the `top` ranked names at each signal date, and the
    equal-weight benchmark of every name with a signal."""
    n = signal.notna().sum(axis=1)
    w = (signal.rank(axis=1, ascending=ascending) <= top).astype(float).div(top).where(n >= min_n)
    ew = signal.notna().astype(float).div(n, axis=0).where(n >= min_n)
    return w, ew


def carry():
    print("C1 carry: realised 12m/6m roll yield, long top 3/5 of 13 ETFs monthly vs equal weight")
    E = panel(ETFS)
    F = panel([ETF_ROOT[e] + "=F" for e in ETFS]).reindex(E.index).ffill(limit=5)
    F.columns = ETFS
    me = month_ends(E.index)
    for lb in (126, 252):
        c = realized_carry(E, F, load_series("DTB3") / 100, lb).loc[me]
        for k in (3, 5):
            w, ew = basket(c, k)
            report(f"lookback {lb} top {k}", sim(w.reindex(E.index).ffill()), sim(ew.reindex(E.index).ffill()))


def basis():
    print("C2 energy basis: FRED spot vs front month (5-day lag), long ETF only when spot > front")
    for etf, sid, root in [("USO", "DCOILWTICO", "CL"), ("UNG", "DHHNGSP", "NG"), ("UGA", "DGASNYH", "RB")]:
        idx = load_bars(etf).index
        b = np.log(load_series(sid).reindex(idx).ffill(limit=3) / load_bars(root + "=F").close.reindex(idx).ffill(limit=3))
        sig = b.rolling(5).mean().shift(5)
        report(f"{etf} if backwardated", sim((sig > 0).astype(float).to_frame(etf)),
               sim(sig.notna().astype(float).to_frame(etf)))


def grains():
    print("C3 grain weather premium: Jul 1 - Sep 30 front-month return per year (expiry windows zeroed)")
    for root, months in [("ZC", (3, 5, 7, 9, 12)), ("ZS", (1, 3, 5, 7, 8, 9, 11)), ("ZW", (3, 5, 7, 9, 12))]:
        c = load_bars(root + "=F").close
        s = season_returns(c, "07-01", "09-30", expiry_mask(c.index, months))
        d = s.loc[:2023]
        print(f"  {root} design {d.mean():+.3f}/season t {d.mean() / d.std() * np.sqrt(len(d)):.2f} down {(d < 0).mean():.0%} "
              f"| 2024-26 {s.loc[2024:].round(3).tolist()}")


def wasde():
    from backtester.data.wasde import load_wasde

    print("C4 post-WASDE drift (t+1 open -> t+5 close), net 20 bp round trip")
    rev = wasde_revisions(load_wasde())
    etf = {"Corn": "CORN", "Oilseed, Soybean": "SOYB", "Wheat": "WEAT"}
    rows = []
    for com, sym in etf.items():
        b = load_bars(sym)
        rc = np.log(b.close).diff()
        sd = rc.rolling(60).std().shift(1)
        for _, e in rev[rev.commodity == com].iterrows():
            if e.date not in b.index or b.index.get_loc(e.date) + 6 >= len(b):
                continue
            i = b.index.get_loc(e.date)
            rows.append({"etf": sym, "r0": rc.iloc[i], "z0": rc.iloc[i] / sd.iloc[i], "rev": e.revision,
                         "drift": np.log(b.close.iloc[i + 5] / b.open.iloc[i + 1])})
    E = pd.DataFrame(rows).dropna(subset=["z0"])
    print(f"  day-0 return vs ending-stocks revision corr {E.r0.corr(E.rev):.2f} (priced the same day)")
    for k in (1.0, 1.5, 2.0):
        s = E[E.z0.abs() > k]
        print(f"  follow |z0| > {k}: {ev(np.sign(s.z0) * s.drift - 2e-3)}")
    s = E[E.rev.abs() > 0.02]
    print(f"  trade the revision sign (|rev| > 2%): {ev(-np.sign(s.rev) * s.drift - 2e-3)}")


def usda():
    print("C5 Prospective Plantings / Acreage (last business day of Mar/Jun): follow day-0 move t+1..t+5")
    rows = []
    for root in ["ZC", "ZS", "ZW"]:
        b = load_bars(root + "=F")
        rc = np.log(b.close).diff()
        sd = rc.rolling(60).std().shift(1)
        for y in range(2000, 2027):
            for m in (3, 6):
                days = b.index[(b.index.year == y) & (b.index.month == m)]
                if len(days) and b.index.get_loc(days[-1]) + 6 < len(b):
                    i = b.index.get_loc(days[-1])
                    rows.append({"date": days[-1], "z0": rc.iloc[i] / sd.iloc[i],
                                 "drift": np.log(b.close.iloc[i + 5] / b.open.iloc[i + 1])})
    E = pd.DataFrame(rows)
    pnl = np.sign(E.z0) * E.drift - 2e-3
    print(f"  design {ev(pnl[E.date < HOLD])} | holdout {ev(pnl[E.date >= HOLD])}")


def energy():
    print("C6/C7/C9 day-of-week, roll window, gasoline spring")
    for sym in ["UNG", "USO"]:
        r = np.log(load_bars(sym).close).diff()
        print(f"  {sym} by weekday (design): " + " | ".join(f"{d} {ev(r[r.index.dayofweek == i][:'2024-06'])}"
                                                      for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])))
    for front, back in [("USO", "USL"), ("UNG", "UNL")]:
        a = np.log(panel([front, back])).diff().dropna()
        sp = a[back] - a[front]
        bd = sp.groupby(sp.index.to_period("M")).cumcount() + 1
        win = sp[(bd >= 5) & (bd <= 9)]
        print(f"  {back}-{front} on GSCI roll days 5-9: design {ev(win[:'2024-06'])} | holdout {ev(win[HOLD:])}")
    for sym in ["RB=F", "UGA"]:
        s = season_returns(load_bars(sym).close, "02-01", "04-30").loc[:2024]
        print(f"  {sym} Feb-Apr {s.mean():+.3f}/yr t {s.mean() / s.std() * np.sqrt(len(s)):.2f} "
              f"({'roll-gap artifact: the April RBOB contract is summer grade' if '=F' in sym else 'what an ETF earned'})")


def pairs():
    print("C13 ratio reversion (60d z > 2 in, cross 0 out), 50/50 legs, 5 bp per side")
    for a, b in [("UGA", "USO"), ("GLD", "SLV"), ("CORN", "WEAT"), ("SOYB", "CORN")]:
        P = panel([a, b])
        pos = ratio_positions(np.log(P[a] / P[b]))
        report(f"{a}/{b}", sim(pd.DataFrame({a: 0.5 * pos, b: -0.5 * pos}), {a: 5e-4, b: 5e-4}))


def misc():
    E = panel(ETFS + ["DBC"])
    idx = E.index
    me, fri = month_ends(idx), idx.to_series().groupby(idx.to_period("W-FRI")).last().values
    print("C8 short UNG in contango (UNL beat UNG over 63d) and downtrend")
    P = panel(["UNG", "UNL"])
    rel = np.log(P.UNL / P.UNL.shift(63)) - np.log(P.UNG / P.UNG.shift(63))
    report("short UNG", sim(pd.DataFrame({"UNG": -((rel > 0) & (np.log(P.UNG / P.UNG.shift(63)) < 0)).astype(float)})))
    print("C10 commodity time-series momentum (12m > 0, inverse-vol) vs inverse-vol all")
    m12 = np.log(E[ETFS] / E[ETFS].shift(252)).loc[me]
    iv = (1 / np.log(E[ETFS]).diff().rolling(20).std().loc[me]).where(m12.notna())
    ok = m12.notna().sum(axis=1) >= 8
    w = iv.where(m12 > 0).div(iv.sum(axis=1), axis=0).fillna(0).where(ok)
    report("TSMOM long/cash", sim(w.reindex(idx).ffill()), sim(iv.div(iv.sum(axis=1), axis=0).where(ok).reindex(idx).ffill()))
    print("C14 long-run reversal: bottom 3 by 5-year front-month change vs equal weight")
    F = panel([ETF_ROOT[e] + "=F" for e in ETFS]).reindex(idx).ffill(limit=5)
    F.columns = ETFS
    w, ew = basket(np.log(F / F.shift(1260)).loc[me].where(E[ETFS].loc[me].notna()), 3, ascending=True)
    report("5y reversal", sim(w.reindex(idx).ffill()), sim(ew.reindex(idx).ffill()))
    print("C15 implied-vol spikes (> 1y 90th pct) -> long 21 days")
    for etf, sid in [("USO", "OVXCLS"), ("GLD", "GVZCLS")]:
        v = load_series(sid).reindex(load_bars(etf).index).ffill()
        sig = (v > v.rolling(252).quantile(0.9)).astype(float).rolling(21).max().fillna(0)
        report(f"{etf} after {sid} spike", sim(sig.to_frame(etf)), sim(sig.notna().astype(float).to_frame(etf)))
    print("C17 AUD+CAD+NZD weekly return > 0 -> long DBC next week")
    fx = pd.DataFrame({k: load_bars(k + "=X").close for k in ["AUDUSD", "CADUSD", "NZDUSD"]}).reindex(idx).ffill()
    report("FX -> DBC", sim((np.log(fx).diff(5).mean(axis=1).loc[fri] > 0).astype(float).reindex(idx).ffill().to_frame("DBC")),
           sim(pd.DataFrame({"DBC": 1.0}, index=idx)))
    print("C18 weekly reversal: 3 worst of 13 vs equal weight")
    w, ew = basket(np.log(E[ETFS]).diff(5).loc[fri], 3, ascending=True)
    report("weekly reversal", sim(w.reindex(idx).ffill()), sim(ew.reindex(idx).ffill()))
    print("C20 10y TIPS yield 5-day change < 0 -> long GLD next week")
    ry = load_series("DFII10").reindex(idx).ffill()
    report("TIPS -> GLD", sim((ry.diff(5).loc[fri] < 0).astype(float).reindex(idx).ffill().to_frame("GLD")),
           sim(pd.DataFrame({"GLD": 1.0}, index=idx)))
    print("C12 GLD/SLV overnight (close -> open) needs > 4 bp/day round trip")
    for s in ["GLD", "SLV"]:
        b = load_bars(s)
        print(f"  {s} overnight design {ev(np.log(b.open / b.close.shift(1))[:'2024-06'])}")


def storage():
    print("C6 lead: short natural gas from the close before each EIA storage report to the report-day close")
    u, ng = load_bars("UNG"), load_bars("NG=F")
    days = storage_report_days(u.index)
    pnl = storage_day_returns(u.close, days)
    print(f"  UNG net 6 bp:   design {ev(pnl[:'2024-06'])} | holdout {ev(pnl[HOLD:])}")
    yr = pnl.groupby(pnl.index.year).sum()
    print(f"  UNG years positive {(yr[:2023] > 0).mean():.0%}; by year " + " ".join(f"{y % 100:02d}:{v:+.2f}" for y, v in yr.items()))
    nr = np.log(ng.close).diff()
    nr = nr.where(nr.groupby(nr.index.to_period("M")).cumcount(ascending=False) > 4)  # NG expires ~3 BDs before month end
    for a, z in [("2002-05", "2009"), ("2010", "2016"), ("2017", "2024-06"), (HOLD, None)]:
        x = nr.loc[a:z]
        rd = storage_report_days(x.index)
        print(f"  NG=F {a}..{z or 'now'}: report day {ev(x.reindex(rd))} | other days {ev(x[~x.index.isin(rd)])}")
    pre = nr.loc[:"2002-04-30"]
    print(f"  NG=F before May 2002 (AGA reported Wednesdays): Wed {ev(pre[pre.index.dayofweek == 2])} | Thu {ev(pre[pre.index.dayofweek == 3])}")
    ttf = np.log(load_bars("TTF=F").close).diff()
    print(f"  Dutch TTF gas (no EIA report) Thursdays {ev(ttf[ttf.index.dayofweek == 3])} vs other days {ev(ttf[ttf.index.dayofweek != 3])}")
    xle = np.log(load_bars("XLE").close).diff()
    gas = pd.concat([(np.log(load_bars(s).close).diff() - xle).reindex(storage_report_days(xle.index)) for s in ["FCG", "EQT", "RRC", "AR", "CNX"]], axis=1).mean(axis=1)
    print(f"  gas producers (FCG EQT RRC AR CNX) minus XLE on report days: design {ev(gas[:'2024-06'])} | holdout {ev(gas[HOLD:])}")
    shifted = days[days.dayofweek == 2]
    print(f"  holiday-week Wednesday reports: UNG {ev(np.log(u.close).diff().reindex(shifted))} (too few to tell)")
    o, c = u.open, u.close
    print("  timing neighbours (UNG, net 6 bp, design | holdout):")
    for name, x in [("Thu open -> close", np.log(c / o)), ("Thu open -> Fri open", np.log(o.shift(-1) / o))]:
        p = -x.reindex(days).dropna() - 6e-4
        print(f"    {name:20s} {ev(p[:'2024-06'])} | {ev(p[HOLD:])}")
    r = np.log(c).diff()[:"2024-06"]
    print("  placebo weekdays, short close->close (design): " + " | ".join(
        f"{d} {ev(-r[r.index.dayofweek == i] - 6e-4)}" for i, d in [(0, "Mon"), (1, "Tue"), (2, "Wed"), (4, "Fri")]))
    prev = np.log(c).diff().shift(1)
    for name, f in [("after an up entry day", lambda x: prev.reindex(x.index) > 0), ("Nov-Mar (withdrawal season)", lambda x: x.index.month.isin([11, 12, 1, 2, 3]))]:
        p_ = storage_day_returns(c, days)
        print(f"  post-hoc filter {name:28s} design {ev(p_[:'2024-06'][f(p_[:'2024-06'])])} | holdout {ev(p_[HOLD:][f(p_[HOLD:])])}")
    rr = np.log(c).diff()
    print("  same up-day filter on other weekdays (design, short): " + " | ".join(
        f"{d} {ev(-rr[(rr.index.dayofweek == i) & (rr.shift(1) > 0)][:'2024-06'] - 6e-4)}" for i, d in [(0, "Mon"), (1, "Tue"), (2, "Wed"), (4, "Fri")]))
    th = r.reindex(days).dropna()
    print(f"  median report day {th.median() * 1e4:.0f} bp vs {r[~r.index.isin(days)].median() * 1e4:.0f} bp; down {(th < 0).mean():.0%}")
    try:
        from backtester.data.yahoo import load_hourly

        h = load_hourly("NG=F")
        h.index = h.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
        hr = np.log(h.close).diff()
        bar = hr[(hr.index.hour == 10)]
        print(f"  NG=F hourly, last ~2 years: 10:00-11:00 ET bar (holds the 10:30 release) Thu {ev(bar[bar.index.dayofweek == 3])} "
              f"| other days {ev(bar[bar.index.dayofweek != 3])}")
    except Exception as e:  # hourly history is optional
        print(f"  hourly check skipped: {e}")


def calendar():
    print("C21-C25 EIA petroleum Wednesday, turn of month, lead-lag, front-month expiry pressure")
    for sym in ["UGA", "HO=F", "USO"]:
        r = np.log(load_bars(sym).close).diff()
        w = r[r.index.dayofweek == 2]
        print(f"  {sym} Wednesday design {ev(w[:'2024-06'])} | holdout {ev(w[HOLD:])}")
    for sym in ["DBC", "GSG"]:
        r = np.log(load_bars(sym).close).diff()
        m = r.index.to_period("M")
        tom = (r.groupby(m).cumcount(ascending=False) == 0) | (r.groupby(m).cumcount() <= 2)
        print(f"  {sym} last + first 3 days of month design {ev(r[tom][:'2024-06'])} | holdout {ev(r[tom][HOLD:])}")
    for lead, lag in [("USO", "UGA"), ("USO", "CANE"), ("GLD", "GDX"), ("SLV", "SIL"), ("CORN", "SOYB")]:
        P = np.log(panel([lead, lag])).diff().dropna()
        pnl = np.sign(P[lead]).shift(1) * P[lag] - 8e-4
        print(f"  {lead} today -> {lag} tomorrow, net 8 bp: design {ev(pnl[:'2024-06'])} | holdout {ev(pnl[HOLD:])}")
    for root in ["CL", "NG"]:
        b = load_bars(root + "=F").loc["2010":]
        v = b.volume.replace(0, np.nan).ffill()
        jump = (v / v.shift(1) > 2.0) & (v.shift(1) < 0.7 * v.rolling(20).median().shift(1))
        rolls = b.index[jump & ~jump.shift(1, fill_value=False).rolling(10).max().astype(bool)]
        r = np.log(b.close).diff()
        pre = pd.Series({d: r.iloc[b.index.get_loc(d) - 4:b.index.get_loc(d)].sum() for d in rolls if b.index.get_loc(d) > 5})
        print(f"  {root}=F 4 days before the roll: design {ev(pre[:'2024-06'])} | holdout {ev(pre[HOLD:])}")
    print("C26 commodity leveraged-ETF rebalancing: fade a > 2 sd day next open -> close, net 6 bp")
    for sym in ["UNG", "USO", "SLV", "GLD"]:
        b = load_bars(sym)
        r = np.log(b.close).diff()
        z = r / r.rolling(60).std().shift(1)
        pnl = (-np.sign(z) * np.log(b.close / b.open).shift(-1))[z.abs() > 2] - 6e-4
        print(f"  {sym}: design {ev(pnl[:'2024-06'])} | holdout {ev(pnl[HOLD:])}")


SECTIONS = {"carry": carry, "basis": basis, "grains": grains, "wasde": wasde, "usda": usda, "energy": energy,
            "pairs": pairs, "misc": misc, "calendar": calendar, "storage": storage}

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        print(f"\n== {name}")
        SECTIONS[name]()
