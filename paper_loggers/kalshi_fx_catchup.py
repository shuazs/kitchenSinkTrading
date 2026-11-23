"""Paper-trade Kalshi hourly EUR/USD and USD/JPY contracts against live FX; logs only.

    python paper_loggers/kalshi_fx_catchup.py            # poll 09:00-17:00 ET on weekdays until Ctrl-C
    python paper_loggers/kalshi_fx_catchup.py --once     # show FX, sigma and the best quote per series now
    python paper_loggers/kalshi_fx_catchup.py --report   # score settled paper trades

Rule (examples/run_kalshi_fx_catchup.py): in the last 60 minutes before a
KXEURUSDH / KXUSDJPYH fixing, price each "above K" strike with a driftless
lognormal (`digital_fair_value`, sigma = trailing 60-min realized vol of
1-minute bars) and buy the side whose ask is more than 3c (after the 7%
p(1-p) taker fee) under fair value, 5-95c, once per market, $100 per trade.
Backtest 2022-24: +4.9c per contract (t 4.4, placebo with 30-min-old FX
+0.8c); the edge falls to ~2.5c with a 5-10 minute FX lag.

FX comes from Yahoo 1-minute bars (EURUSD=X, JPY=X), so each trade logs its
data lag. Fills are paper fills at the top-of-book ask (depth not checked).
Logs in ~/.cache/kitchensink/paper/kalshi_fx_catchup/trades.csv. Kalshi
market data needs no key. Graduates after 3 months at >= +2.4c per contract.
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
import requests

from _common import log_dir, read_log, tstat, write_log
from backtester.data.live_books import kalshi_market, kalshi_series_markets, market_top
from backtester.kalshi_catchup import digital_fair_value, taker_fee

PAIRS = {"KXEURUSDH": "EURUSD=X", "KXUSDJPYH": "JPY=X"}
MARGIN, RISK, WINDOW = 0.03, 100.0, 60
DIR = log_dir("kalshi_fx_catchup")
TRADES = DIR / "trades.csv"
FIELDS = ["time", "ticker", "K", "side", "price", "qty", "fee", "fair", "fx", "lag_s", "minutes_left", "result", "pnl"]


def fx_state(symbol: str):
    """(time of the last complete bar, close, per-minute sigma over 60 bars) from Yahoo 1-minute bars."""
    import yfinance as yf

    b = yf.download(symbol, period="1d", interval="1m", progress=False)
    b.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in b.columns]
    b.index = pd.DatetimeIndex(b.index).tz_convert("UTC")
    px = b["close"].dropna().iloc[:-1]  # the last bar is still forming
    sig = np.log(px).diff().tail(60).std()
    return px.index[-1] + pd.Timedelta(minutes=1), float(px.iloc[-1]), float(sig)


def strikes(series: str, now: pd.Timestamp) -> list[dict]:
    """Open "above K" markets of the next fixing within WINDOW minutes."""
    out = []
    for m in kalshi_series_markets(series):
        k = re.search(r"above ([\d.]+)", m.get("title", ""))
        close = pd.Timestamp(m["close_time"])
        if k and 1 <= (close - now).total_seconds() / 60 <= WINDOW:
            out.append(dict(m, K=float(k.group(1)), close=close))
    return out


def best_quote(ms: list[dict], fx: float, sig: float, now: pd.Timestamp) -> dict | None:
    best = None
    for m in ms:
        tau = (m["close"] - now).total_seconds() / 60
        F = float(digital_fair_value(fx, m["K"], sig, tau))
        bid, ask = market_top(m)
        for side, price, fair in (("yes", ask, F), ("no", None if bid is None else round(1 - bid, 4), 1 - F)):
            if price is None or not 0.05 <= price <= 0.95:
                continue
            edge = fair - price - 0.07 * price * (1 - price)
            if best is None or edge > best["edge"]:
                best = dict(ticker=m["ticker"], K=m["K"], side=side, price=price, fair=round(fair, 4),
                            edge=round(edge, 4), minutes_left=round(tau, 1))
    return best


def poll() -> None:
    now = pd.Timestamp.now(tz="UTC")
    t = read_log(TRADES, FIELDS)
    for series, sym in PAIRS.items():
        ms = [m for m in strikes(series, now) if m["ticker"] not in set(t["ticker"])]
        if not ms:
            continue
        bar_t, fx, sig = fx_state(sym)
        q = best_quote(ms, fx, sig, now)
        if q and q["edge"] > MARGIN:
            qty = int(RISK / q["price"])
            row = dict(time=now.isoformat(), ticker=q["ticker"], K=q["K"], side=q["side"], price=q["price"], qty=qty,
                       fee=taker_fee(qty, q["price"]), fair=q["fair"], fx=fx,
                       lag_s=round((now - bar_t).total_seconds()), minutes_left=q["minutes_left"])
            t = pd.concat([t, pd.DataFrame([row])], ignore_index=True)
            print(f"{now:%H:%M:%S} PAPER BUY {q} fx {fx} (lag {row['lag_s']}s)")
    write_log(t, TRADES)


def settle() -> None:
    t = read_log(TRADES, FIELDS)
    for i in t.index[t.result.isna()]:
        try:
            m = kalshi_market(t.at[i, "ticker"])
        except requests.RequestException:
            continue
        if m.get("result") in ("yes", "no"):
            side, price, qty = t.at[i, "side"], float(t.at[i, "price"]), int(float(t.at[i, "qty"]))
            t.at[i, "result"] = m["result"]
            t.at[i, "pnl"] = round(qty * (float(m["result"] == side) - price) - float(t.at[i, "fee"]), 2)
    write_log(t, TRADES)


def report() -> None:
    t = read_log(TRADES, FIELDS)
    s = t.dropna(subset=["pnl"])
    c = pd.to_numeric(s.pnl) / pd.to_numeric(s.qty) * 100
    lag = pd.to_numeric(t.lag_s, errors="coerce")
    print(f"{len(t)} paper trades, {len(s)} settled: {c.mean():+.2f}c per contract (t {tstat(c):.2f}), "
          f"${pd.to_numeric(s.pnl).sum():+.0f}; median FX data lag {lag.median():.0f}s")
    print("Backtest +4.9c per contract (2022-24, ~2.5c at a 5-10 min FX lag); graduates at >= +2.4c after 3 months.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.once:
        now = pd.Timestamp.now(tz="UTC")
        for series, sym in PAIRS.items():
            bar_t, fx, sig = fx_state(sym)
            ms = strikes(series, now)
            print(f"{series}: {sym} {fx} at {bar_t}, sigma/min {sig:.6f}, {len(ms)} strikes in window, "
                  f"best {best_quote(ms, fx, sig, now) if ms else None}")
        return
    while True:
        t0 = time.time()
        et = pd.Timestamp.now(tz="America/New_York")
        if et.weekday() < 5 and 8 <= et.hour < 17:
            try:
                poll()
                settle()
            except Exception as e:  # a failed poll must not stop the logger
                print(f"{et:%H:%M:%S} poll failed: {e}")
        elif int(t0) % 1800 < args.interval:
            settle()
        time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
