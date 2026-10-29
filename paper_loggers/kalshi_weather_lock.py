"""Forward-test the Kalshi daily-high "already decided" rule on paper; never places orders.

    python paper_loggers/kalshi_weather_lock.py --once     # show today's locked brackets and their asks, then stop
    python paper_loggers/kalshi_weather_lock.py            # poll every --interval seconds until Ctrl-C
    python paper_loggers/kalshi_weather_lock.py --report   # score the paper trades that have settled

Run it on your own computer: Kalshi's API and aviationweather.gov are blocked
from the project's cloud sandbox.

The rule (thread "Find weather-based trading edges", Oct 2026): a day's high
can only go up. Once two METAR reports at the settlement station since local
standard midnight are at or above a bracket's threshold (+0.5 F margin for
rounding), "X or above" is settled Yes and every bracket / "X or below"
contract whose top is passed is settled No. Kalshi prints show these decided
contracts still selling below 97c on about 1% of city-days (mostly Chicago, on
cold-front days when the high was set just after midnight and traders still
price the forecast afternoon high). Backtest 2021 - Aug 2025 (ISD obs vs
Kalshi taker prints of the locked side at 60-97c): every print won, +11.3c per
print in-sample (43 city-days), +4.6c holdout (12 city-days), about $1k a year
if every printed contract had been ours. The two-report rule, the 0.5 F margin
and the 60c floor guard against a bad sensor reading (Central Park, Dec 29
2024: a 68 F report the climate summary threw out; the market priced it right).

Each poll reads the latest METARs, lists today's open daily-high markets of
the watched cities, and records one paper buy of --contracts at the ask of the
locked side when that ask is between --min-price and --max-price. Logs in
~/.cache/kitchensink/paper/kalshi_weather_lock/: trades.csv and locks.csv
(every locked market seen, with its ask, to measure how long cheap asks last).
"""
import argparse
import csv
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import _common
from backtester.data.live_books import kalshi_market, kalshi_series_markets, market_top, paper_pnl
from backtester.weather_trades import kalshi_bounds, locked_side

DIR = _common.log_dir("kalshi_weather_lock")
TRADES, LOCKS = DIR / "trades.csv", DIR / "locks.csv"
TRADE_FIELDS = ["entry_time", "series", "ticker", "subtitle", "side", "price", "contracts", "station", "obs_max2_f",
                "result", "net", "settled_time"]
LOCK_FIELDS = ["time", "ticker", "side", "ask", "obs_max2_f", "last_obs"]
# series -> (ICAO settlement station, UTC offset of local standard time)
SERIES = {
    "KXHIGHNY": ("KNYC", -5), "KXHIGHCHI": ("KMDW", -6), "KXHIGHMIA": ("KMIA", -5), "KXHIGHAUS": ("KAUS", -6),
    "KXHIGHPHIL": ("KPHL", -5), "KXHIGHDEN": ("KDEN", -7), "KXHIGHLAX": ("KLAX", -8),
}
METAR = "https://aviationweather.gov/api/data/metar"


def metars(stations) -> pd.DataFrame:
    """Recent METAR/SPECI temperatures (deg F, from the tenths-of-C remark group when present)."""
    r = requests.get(METAR, params={"ids": ",".join(stations), "format": "json", "hours": 30}, timeout=20)
    r.raise_for_status()
    rows = [(x.get("icaoId"), x.get("obsTime") or x.get("reportTime"), x.get("temp")) for x in r.json()]
    d = pd.DataFrame(rows, columns=["station", "time", "temp_c"]).dropna()
    t = pd.to_numeric(d["time"], errors="coerce")
    d["time"] = pd.to_datetime(t, unit="s", utc=True) if t.notna().all() else pd.to_datetime(d["time"], utc=True)
    d["temp_f"] = d["temp_c"].astype(float) * 9 / 5 + 32
    return d


def second_max_today(obs: pd.DataFrame, station: str, offset: int, now: datetime) -> tuple[float | None, str]:
    """Second-highest report since local-standard midnight (two reports guard against a bad reading)."""
    day_start = (pd.Timestamp(now) + pd.Timedelta(hours=offset)).normalize() - pd.Timedelta(hours=offset)
    x = obs[(obs["station"] == station) & (obs["time"] >= day_start)].sort_values("time")
    if len(x) < 2:
        return None, ""
    return float(np.sort(x["temp_f"].to_numpy())[-2]), x["time"].iloc[-1].isoformat()


def load(path, fields) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"result": str}) if path.exists() else pd.DataFrame(columns=fields)


def append(path, fields, rows) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def scan(contracts: int, min_price: float, max_price: float, margin: float, once: bool) -> None:
    now = datetime.now(timezone.utc)
    obs = metars([s for s, _ in SERIES.values()])
    trades = load(TRADES, TRADE_FIELDS)
    taken = set(trades["ticker"])
    new, locks = [], []
    for series, (station, off) in SERIES.items():
        m2, last = second_max_today(obs, station, off, now)
        if m2 is None:
            continue
        local = (now + timedelta(hours=off)).strftime("%y%b%d").upper()
        try:
            mk = [m for m in kalshi_series_markets(series) if f"-{local}-" in m.get("ticker", "")]
        except requests.RequestException as e:
            print(f"  {series}: {e}")
            continue
        if not mk:
            continue
        b = kalshi_bounds(pd.DataFrame({"yes_sub_title": [m.get("yes_sub_title", "") for m in mk]}))
        sides = locked_side(b["kind"], b["a"], b["b"], np.full(len(b), m2), margin)
        for m, side in zip(mk, sides):
            if side is None:
                continue
            bid, ask = market_top(m)
            price = ask if side == "yes" else (None if bid is None else round(1 - bid, 4))
            locks.append({"time": now.isoformat(), "ticker": m["ticker"], "side": side, "ask": price,
                          "obs_max2_f": round(m2, 1), "last_obs": last})
            if once:
                print(f"  {m['ticker']:28s} {m.get('yes_sub_title', ''):16s} locked {side:3s} ask {price} (2nd max {m2:.1f}F)")
            if price is not None and min_price <= price <= max_price and m["ticker"] not in taken:
                new.append({"entry_time": now.isoformat(), "series": series, "ticker": m["ticker"],
                            "subtitle": m.get("yes_sub_title", ""), "side": side, "price": price, "contracts": contracts,
                            "station": station, "obs_max2_f": round(m2, 1), "result": "", "net": "", "settled_time": ""})
                taken.add(m["ticker"])
    if locks:
        append(LOCKS, LOCK_FIELDS, locks)
    if new:
        append(TRADES, TRADE_FIELDS, new)
        for t in new:
            print(f"PAPER BUY {t['ticker']} {t['side']} at {t['price']:.2f} x{contracts}")


def settle() -> pd.DataFrame:
    trades = load(TRADES, TRADE_FIELDS)
    for i, t in trades[trades["result"].isna() | (trades["result"] == "")].iterrows():
        try:
            m = kalshi_market(t["ticker"])
        except requests.RequestException:
            continue
        res = m.get("result")
        if res in ("yes", "no"):
            trades.loc[i, "result"] = res
            trades.loc[i, "net"] = paper_pnl(t["side"], float(t["price"]), res, int(t["contracts"]))
            trades.loc[i, "settled_time"] = datetime.now(timezone.utc).isoformat()
    if len(trades):
        trades.to_csv(TRADES, index=False)
    return trades


def report() -> None:
    t = settle()
    done = t[t["result"].isin(["yes", "no"])].copy()
    if done.empty:
        print("no settled paper trades yet")
        return
    done["net"] = done["net"].astype(float)
    done["day"] = done["ticker"].str.extract(r"^([A-Z]+-\d\d[A-Z]{3}\d\d)")[0]
    by_day = done.groupby("day")["net"].mean()
    print(f"{len(done)} settled trades on {len(by_day)} city-days: {done['net'].mean() * 100:+.1f}c/contract, "
          f"win rate {(done['result'] == done['side']).mean():.3f}, day-clustered t {_common.tstat(by_day):.2f}")
    print("Backtest: every decided print won; half the holdout edge is +2.3c/contract (graduates after 30 city-days).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--interval", type=int, default=60, help="seconds between polls")
    ap.add_argument("--contracts", type=int, default=20)
    ap.add_argument("--min-price", type=float, default=0.60, help="below this the market probably knows of a bad reading")
    ap.add_argument("--max-price", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.5, help="deg F above the threshold before a bracket counts as decided")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.once:
        scan(a.contracts, a.min_price, a.max_price, a.margin, once=True)
    else:
        while True:
            try:
                scan(a.contracts, a.min_price, a.max_price, a.margin, once=False)
                settle()
            except requests.RequestException as e:
                print(f"{datetime.now(timezone.utc):%H:%M:%S} {e}")
            time.sleep(a.interval)
