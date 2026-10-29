"""Weather-based trading hypotheses (thread "Find weather-based trading edges", 2026-10-29).

Pre-registration and outputs: /mnt/project-files/weather/ (HYPOTHESES.md, results/, scripts/).
Every test splits in-sample (before Jul 2024) from the locked holdout (Jul 2024+).

    python examples/run_weather.py actuals     # W3 storage day x weather, W7 cold shocks, W4 Corn Belt
    python examples/run_weather.py revisions   # W1/W9/W10/W11 GEFS revisions -> UNG, gas producers (needs GEFS cache)
    python examples/run_weather.py grains      # W13 grain ETF after-hours lag
    python examples/run_weather.py overseas    # W17 Europe / Asia weather -> TTF, LNG, UNG (world GEFS cache)
    python examples/run_weather.py kalshi      # W6 Kalshi daily-high lock-in (needs the event-markets mirror)
    python examples/run_weather.py hurricanes  # W12 landfalls -> insurers, HD/LOW, UNG

The first GEFS call downloads ~2,200 runs (about 45 min per run hour) into ~/.cache/kitchensink/weather/.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from backtester import weather_trades as wt
from backtester.data import weather as w
from backtester.data.yahoo import load_bars

HO = pd.Timestamp("2024-07-01")
COST = 10e-4  # round trip, UNG / grain ETFs at 5 bp per side


def report(label: str, r: pd.Series) -> None:
    r = r.dropna()
    parts = []
    for name, x in (("IS", r[r.index < HO]), ("HO", r[r.index >= HO])):
        t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan
        parts.append(f"{name} n={len(x):4d} mean={x.mean() * 1e4:+7.1f}bp t={t:+.2f}")
    print(f"{label:52s} " + " | ".join(parts))


def storage_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """EIA storage report sessions: Thursday, or Wednesday when Thursday is closed."""
    out = []
    for thu in pd.date_range(days[0], days[-1], freq="W-THU"):
        if thu in days:
            out.append(thu)
        elif thu - pd.Timedelta(days=1) in days:
            out.append(thu - pd.Timedelta(days=1))
    return pd.DatetimeIndex(out)


def actuals() -> None:
    temps = w.actuals()
    dd = w.demand_index(temps)
    anom = w.anomaly(dd, w.normals(dd))["dd"]
    ung = load_bars("UNG", "2007")
    c, o, days = ung["close"], ung["open"], ung.index
    rep = storage_days(days)
    i = days.get_indexer(rep)
    short = pd.Series(-(c.iloc[i].to_numpy() / c.iloc[i - 1].to_numpy() - 1) - COST, index=rep)
    week = pd.Series([anom.loc[d - pd.Timedelta(days=d.weekday() + 3 + 6): d - pd.Timedelta(days=d.weekday() + 3)].mean()
                      for d in rep], index=rep)
    q = week[week.index < HO].quantile([1 / 3, 2 / 3]).to_numpy()
    report("W3 storage-day short, all weeks", short)
    report("W3 storage-day short, mild report week", short[week < q[0]])
    report("W3 storage-day short, cold/hot report week", short[week >= q[1]])

    a3 = anom.rolling(3).mean()
    winter = a3.index.month.isin([11, 12, 1, 2, 3])
    sd = a3[(a3.index < HO) & winter].std()
    sig = ((a3 > 2 * sd) & winter).shift(1).reindex(days).fillna(False)
    ev, last = [], None
    for d in days[sig.to_numpy()]:
        if last is None or (d - last).days > 7:
            ev.append(d)
            last = d
    report("W7 cold shock -> long UNG 5 days", wt.event_returns(o, c, ev, 5, COST))

    corn = [k for k, v in w.CITIES.items() if v[4] == "corn"]
    tx = w.anomaly(w.actuals(corn, "TMAX")).mean(axis=1).rolling(14).mean()
    pr = w.actuals(corn, "PRCP").fillna(0)
    norm = pd.Series(w.normals(pr).mean(axis=1).loc[pr.index.dayofyear].to_numpy(), index=pr.index)
    ratio = np.log((pr.mean(axis=1).rolling(14).sum() + 0.05) / (norm.rolling(14).sum() + 0.05))
    summer = pd.Series((tx.index.month.isin([7, 8])) | ((tx.index.month == 6) & (tx.index.day >= 15)), index=tx.index)
    ins = summer & (tx.index < HO)
    z = (tx - tx[ins].mean()) / tx[ins].std() - (ratio - ratio[ins].mean()) / ratio[ins].std()
    for sym in ("CORN", "SOYB"):
        b = load_bars(sym, "2010")
        fwd = b["open"].shift(-11) / b["open"].shift(-1) - 1
        zz = z.shift(1).reindex(b.index)
        sm = summer.shift(1).reindex(b.index).fillna(False)
        pos = (zz > 0.5).astype(float)
        report(f"W4 {sym} hot+dry (z>0.5) long 10 days", (pos * fwd - COST)[sm & (pos > 0)].iloc[::10])


def gefs_revisions(run: str = "12", start="2020-10-01", end=None, world=False, value="dd", region=None):
    panel = w.gefs_panel(start, end or pd.Timestamp.now().normalize(), run=run, world=world)
    daily = w.gefs_daily(panel)
    if world:
        pts = {k: v for k, v in w.WORLD_POINTS.items() if v[3] == region}
        wts = pd.Series({k: v[2] for k, v in pts.items()})
        hdd, cdd = w.degree_days(daily[list(pts)])
        dd = (hdd + (cdd if region == "ASIA" else 0)).mul(wts / wts.sum(), axis=1).sum(axis=1)
        table = pd.DataFrame({"init": daily["init"], "date": daily["date"], value: dd})
    else:
        di = w.demand_index(daily.set_index(["init", "date"])[list(w.weights().index)]).reset_index()
        table = di.rename(columns={"dd": value}) if value != "dd" else di
    return table


def revisions() -> None:
    ung = load_bars("UNG", "2020")
    days = ung.index
    table = gefs_revisions("12")
    rev = wt.trading_day_revisions(table, "dd", days, {"1-7": (1, 7), "8-14": (8, 14), "1-14": (1, 14)})
    z = wt.rolling_z(rev)
    o, c = ung["open"], ung["close"]
    same = c.pct_change()
    for k in rev:
        d = pd.DataFrame({"z": z[k], "r": same}).dropna()
        print(f"mechanism: same-day UNG vs revision {k}: corr IS {d[d.index < HO].corr().iloc[0, 1]:+.3f}, HO {d[d.index >= HO].corr().iloc[0, 1]:+.3f}")
    nxt = c.shift(-1) / o.shift(-1) - 1
    for k in rev:
        for thr in (0, 1):
            pos = np.sign(z[k]).where(z[k].abs() > thr, 0)
            report(f"W1 follow revision {k} |z|>{thr}, next day", (pos * nxt - COST)[pos != 0])
    pos = -np.sign(z["8-14"]).where(z["8-14"].abs() > 1.5, 0)
    report("W11 fade |z 8-14| > 1.5, next day", (pos * nxt - COST)[pos != 0])
    x = z["1-14"]
    beta = (x * same).rolling(250, min_periods=120).mean() / (x * x).rolling(250, min_periods=120).mean()
    res = same - beta.shift(1) * x
    rz = wt.rolling_z(res, 250, 120)
    pos = -np.sign(rz).where(rz.abs() > 2, 0)
    report("W9 fade weather-unexplained UNG move, next day", (pos * nxt - COST)[pos != 0])
    prod = pd.concat({s: load_bars(s, "2020") for s in ("EQT", "AR", "RRC", "CNX")}, axis=1)
    xle = load_bars("XLE", "2020").reindex(days)
    po = prod.xs("open", axis=1, level=1).reindex(days)
    pc = prod.xs("close", axis=1, level=1).reindex(days)
    gp = (pc.shift(-1) / po.shift(-1) - 1).mean(axis=1) - (xle["close"].shift(-1) / xle["open"].shift(-1) - 1)
    pos = np.sign(z["1-14"]).where(z["1-14"].abs() > 1, 0)
    report("W10 gas producers - XLE follow revision, next day", (pos * gp - 20e-4)[pos != 0])


def grains() -> None:
    panel = w.gefs_panel("2020-10-01", pd.Timestamp.now().normalize(), run="12")
    corn = [k for k, v in w.CITIES.items() if v[4] == "corn"]
    p = panel[panel["valid"].dt.hour == 18].copy()
    p["date"] = p["valid"].dt.normalize()
    p["tx"] = p[corn].mean(axis=1)
    for sym in ("CORN", "SOYB"):
        b = load_bars(sym, "2020")
        days = b.index
        rev = wt.trading_day_revisions(p, "tx", days, {"6-14": (6, 14)})["6-14"]
        summer = pd.Series(days.month.isin([7, 8]) | ((days.month == 6) & (days.day >= 15)), index=days)
        z = rev / rev[summer].rolling(60, min_periods=20).std().shift(1).reindex(days).ffill()
        gap = b["open"].shift(-1) / b["close"] - 1
        pos = np.sign(z).where(z.abs() > 1, 0)
        report(f"W13 {sym} buy/sell at close on Corn Belt revision, exit open", (pos * gap - COST)[summer & (pos != 0)])


def overseas() -> None:
    days = load_bars("UNG", "2020").index
    out = {}
    for region in ("EU", "ASIA"):
        table = gefs_revisions("12", world=True, region=region)
        out[region] = wt.rolling_z(wt.trading_day_revisions(table, "dd", days, {"1-14": (1, 14)})["1-14"])
    ttf = load_bars("TTF=F", "2017")["close"].pct_change().reindex(days)
    lng, xle, ung = (load_bars(s, "2020").reindex(days) for s in ("LNG", "XLE", "UNG"))
    nxt_lng = (lng["close"].shift(-1) / lng["open"].shift(-1)) - (xle["close"].shift(-1) / xle["open"].shift(-1))
    nxt_ung = ung["close"].shift(-1) / ung["open"].shift(-1) - 1
    for region, z in out.items():
        d = pd.DataFrame({"z": z, "r": ttf}).dropna()
        print(f"mechanism: same-day TTF vs {region} revision: corr {d.corr().iloc[0, 1]:+.3f} (n {len(d)})")
        pos = np.sign(z).where(z.abs() > 1, 0)
        report(f"W17 {region} revision -> LNG - XLE next day", (pos * nxt_lng - 10e-4)[pos != 0])
        report(f"W17 {region} revision -> UNG next day", (pos * nxt_ung - COST)[pos != 0])


def kalshi(markets="/mnt/project-files/weather/kalshi_high_markets.parquet",
           trades="/mnt/project-files/weather/kalshi_high_trades.parquet") -> None:
    """Markets/prints are filtered from /mnt/project-files/event-markets by scripts/pull_kalshi_high.py."""
    from backtester.weather_brackets import taker_fee

    m = wt.kalshi_bounds(pd.read_parquet(markets))
    m["city"] = m["series"].str.replace("KX", "").str.replace("HIGH", "")
    m["date"] = pd.to_datetime(m["ticker"].str.extract(r"-(\d\d[A-Z]{3}\d\d)-")[0], format="%y%b%d")
    m = m[m["kind"].notna() & m["result"].isin(["yes", "no"])]
    tr = pd.read_parquet(trades).merge(m[["ticker", "city", "date", "kind", "a", "b", "result"]], on="ticker")
    parts = []
    for city, (station, off) in w.KALSHI_STATIONS.items():
        ob = w.isd_hourly(station, range(2021, 2026))
        ob["date"] = (ob["ts"] + pd.Timedelta(hours=off)).dt.tz_localize(None).dt.normalize()
        ob["avail"] = ob["ts"] + pd.Timedelta(minutes=10)
        ob["cmax"] = ob.groupby("date")["temp_f"].cummax()
        t = tr[tr["city"] == city].sort_values("created_time")
        t = pd.merge_asof(t, ob[["avail", "date", "cmax"]].rename(columns={"date": "odate"}).sort_values("avail"),
                          left_on="created_time", right_on="avail")
        parts.append(t[t["odate"] == t["date"]])
    t = pd.concat(parts, ignore_index=True)
    t["lock"] = wt.locked_side(t["kind"], t["a"], t["b"], t["cmax"], margin=1.0)
    x = t[t["lock"].notna() & (t["taker_side"] == t["lock"])].copy()
    x["px"] = np.where(x["lock"] == "yes", x["yes_price"], 100 - x["yes_price"]) / 100
    print("W6 locked-side prints by price:", x.groupby(pd.cut(x["px"], [0, .9, .95, .97, .98, .99, 1]))["count"].sum().to_dict())
    x = x[x["px"] <= 0.97]
    x["pnl"] = (x["result"] == x["lock"]) - x["px"] - taker_fee(x["px"])
    for name, y in (("IS", x[x["created_time"] < HO.tz_localize("UTC")]), ("HO", x[x["created_time"] >= HO.tz_localize("UTC")])):
        print(f"W6 buy locked side at <= 97c {name}: prints {len(y)}, days {y['date'].nunique()}, "
              f"{y['pnl'].mean() * 100:+.1f}c/contract, win rate {(y['result'] == y['lock']).mean():.3f}")


def hurricanes() -> None:
    lf = w.us_landfalls()
    names = ["TRV", "ALL", "CB", "HIG", "PGR"]
    px = {s: load_bars(s, "1999") for s in names + ["SPY", "HD", "LOW", "UNG"]}
    o = pd.DataFrame({s: b["open"] for s, b in px.items()})
    c = pd.DataFrame({s: b["close"] for s, b in px.items()})
    ev = lf["t"].dt.normalize()

    def basket(cols, hold):
        r = pd.concat({s: wt.event_returns(o[s].dropna(), c[s].dropna(), ev, hold) for s in cols + ["SPY"]}, axis=1)
        return r[cols].mean(axis=1) - r["SPY"] - COST

    report("W12 insurers - SPY, landfall +1..+5", basket(names, 5))
    report("W12 HD+LOW - SPY, landfall +1..+20", basket(["HD", "LOW"], 20))
    report("W12 UNG, landfall +1..+5", wt.event_returns(o["UNG"].dropna(), c["UNG"].dropna(), ev[ev > "2007-05"], 5))


if __name__ == "__main__":
    tests = {"actuals": actuals, "revisions": revisions, "grains": grains, "overseas": overseas,
             "kalshi": kalshi, "hurricanes": hurricanes}
    for name in sys.argv[1:] or ["actuals"]:
        tests[name]()
