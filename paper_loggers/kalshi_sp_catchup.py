"""Paper-trade Kalshi S&P hourly contracts after a sharp 2-minute index move; logs only.

    python paper_loggers/kalshi_sp_catchup.py            # poll during market hours until Ctrl-C
    python paper_loggers/kalshi_sp_catchup.py --once     # show the current z-score and the open hourly ladder
    python paper_loggers/kalshi_sp_catchup.py --report   # score settled paper trades

Rule (backtester/kalshi_catchup.py, PR #17): when the S&P 500 moves more
than --z (2.5) per-minute sigmas over 2 minutes, find the KXINXU "above K"
ladder that settles 3-60 minutes later, price each strike with
`digital_fair_value` (EWMA minute vol x 1.1), and buy the side the move
favours where the ask is at least 3c under fair value, $250 per trade, with
the 3.5% x p(1-p) taker fee. The backtest made ~$3k over 14 months with
tiny fills (t 2.0, maybe luck).

The index comes from Yahoo 1-minute bars (^GSPC), so each trigger logs its
data lag; the backtest assumed a 5-second reaction, so a lag of a minute or
more is itself a finding. Fills are paper fills at the top-of-book ask
(depth is not checked). Logs in ~/.cache/kitchensink/paper/kalshi_sp_catchup/:
triggers.csv (every move, traded or not) and trades.csv (paper buys, the
bid 15 minutes later and the settled P&L). Kalshi market data needs no key.
Graduates after 3 months at half the backtest P&L rate.
"""
import argparse
import re
import time

import numpy as np
import pandas as pd
import requests

from _common import log_dir, read_log, tstat, write_log
from backtester.data.live_books import kalshi_market, kalshi_series_markets, market_top
from backtester.kalshi_catchup import digital_fair_value, known_prices, minute_vol, taker_fee

SERIES, K_MIN, FEE, VOL_MULT, MARGIN, RISK = "KXINXU", 2, 0.035, 1.1, 0.03, 250.0
DIR = log_dir("kalshi_sp_catchup")
TRIG, TRADES = DIR / "triggers.csv", DIR / "trades.csv"
TRIG_FIELDS = ["time", "bar_time", "lag_s", "spx", "z", "event", "minutes_left", "best_ticker", "fair", "ask", "edge", "traded"]
TRADE_FIELDS = ["time", "ticker", "K", "side", "price", "qty", "fee", "fair", "bid_15m", "pnl_15m", "result", "pnl"]


def spx_state():
    """(known time of the last bar, price, per-minute sigma, 2-minute z) from Yahoo 1-minute ^GSPC bars."""
    import yfinance as yf

    b = yf.download("^GSPC", period="5d", interval="1m", progress=False)
    b.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in b.columns]
    b.index = pd.DatetimeIndex(b.index).tz_convert("UTC")
    px = known_prices(b)
    sig = minute_vol(px)
    t = px.index[-1]
    prev = px.asof(t - pd.Timedelta(minutes=K_MIN))
    z = np.log(px.iloc[-1] / prev) / (sig.iloc[-1] * np.sqrt(K_MIN))
    return t, float(px.iloc[-1]), float(sig.iloc[-1]), float(z)


def ladder(now: pd.Timestamp) -> tuple[str, float, list[dict]]:
    """The open KXINXU "above K" markets of the event settling 3-60 minutes from now."""
    ms = [m for m in kalshi_series_markets(SERIES) if re.search(r"-T\d+(?:\.\d+)?$", m["ticker"])]
    for m in ms:
        m["close"] = pd.Timestamp(m["close_time"])
        m["K"] = float(re.search(r"-T(\d+(?:\.\d+)?)$", m["ticker"]).group(1))
    ms = [m for m in ms if 3 <= (m["close"] - now).total_seconds() / 60 <= 60]
    if not ms:
        return "", np.nan, []
    first = min(m["close"] for m in ms)
    ms = [m for m in ms if m["close"] == first]
    return ms[0].get("event_ticker", ""), (first - now).total_seconds() / 60, ms


def best_quote(ms: list[dict], spx: float, sig: float, tau: float, up: bool) -> dict | None:
    best = None
    for m in ms:
        F = float(digital_fair_value(spx, m["K"], sig, tau, VOL_MULT))
        if not 0.05 < F < 0.95:
            continue
        bid, ask = market_top(m)
        price = ask if up else (None if bid is None else round(1 - bid, 4))
        if price is None:
            continue
        fair = F if up else 1 - F
        if best is None or fair - price > best["edge"]:
            best = dict(ticker=m["ticker"], K=m["K"], side="yes" if up else "no", price=price, fair=round(fair, 4),
                        edge=round(fair - price, 4))
    return best


def poll(z_min: float, last_trigger: list) -> None:
    now = pd.Timestamp.now(tz="UTC")
    bar_t, spx, sig, z = spx_state()
    if abs(z) <= z_min or (last_trigger and bar_t - last_trigger[0] <= pd.Timedelta(minutes=K_MIN)):
        return
    last_trigger[:] = [bar_t]
    ev, tau, ms = ladder(now)
    q = best_quote(ms, spx, sig, tau, z > 0) if ms else None
    traded = bool(q and q["edge"] > MARGIN)
    trig = read_log(TRIG, TRIG_FIELDS)
    row = dict(time=now.isoformat(), bar_time=bar_t.isoformat(), lag_s=round((now - bar_t).total_seconds()), spx=spx,
               z=round(z, 2), event=ev, minutes_left=round(tau, 1) if ms else None, best_ticker=q and q["ticker"],
               fair=q and q["fair"], ask=q and q["price"], edge=q and q["edge"], traded=traded)
    write_log(pd.concat([trig, pd.DataFrame([row])], ignore_index=True), TRIG)
    print(f"{now:%H:%M:%S} z {z:+.1f} (bar lag {row['lag_s']}s) {ev} best {q}" + (" -> PAPER BUY" if traded else ""))
    if traded:
        qty = int(RISK / max(q["price"], 0.01))
        t = read_log(TRADES, TRADE_FIELDS)
        new = dict(time=now.isoformat(), ticker=q["ticker"], K=q["K"], side=q["side"], price=q["price"], qty=qty,
                   fee=taker_fee(qty, q["price"], FEE), fair=q["fair"])
        write_log(pd.concat([t, pd.DataFrame([new])], ignore_index=True), TRADES)


def settle() -> None:
    t = read_log(TRADES, TRADE_FIELDS)
    if t.empty:
        return
    now = pd.Timestamp.now(tz="UTC")
    for i in t.index[t.result.isna()]:
        try:
            m = kalshi_market(t.at[i, "ticker"])
        except requests.RequestException:
            continue
        side, price, qty, fee = t.at[i, "side"], float(t.at[i, "price"]), int(float(t.at[i, "qty"])), float(t.at[i, "fee"])
        if pd.isna(t.at[i, "bid_15m"]) and now >= pd.Timestamp(t.at[i, "time"]) + pd.Timedelta(minutes=15):
            bid, ask = market_top(m)
            exit_px = bid if side == "yes" else (None if ask is None else 1 - ask)
            if exit_px:
                t.at[i, "bid_15m"] = exit_px
                t.at[i, "pnl_15m"] = round(qty * (exit_px - price) - fee - taker_fee(qty, exit_px, FEE), 2)
        if m.get("result") in ("yes", "no"):
            t.at[i, "result"] = m["result"]
            t.at[i, "pnl"] = round(qty * (float(m["result"] == side) - price) - fee, 2)
    write_log(t, TRADES)


def report() -> None:
    trig, t = read_log(TRIG, TRIG_FIELDS), read_log(TRADES, TRADE_FIELDS)
    pnl = pd.to_numeric(t.pnl, errors="coerce").dropna()
    lag = pd.to_numeric(trig.lag_s, errors="coerce")
    print(f"{len(trig)} triggers (median data lag {lag.median():.0f}s), {len(t)} paper trades, {len(pnl)} settled: "
          f"${pnl.sum():+.0f} total, ${pnl.mean():+.1f}/trade, t {tstat(pnl):.2f}; "
          f"sold at the 15-min bid instead: ${pd.to_numeric(t.pnl_15m, errors='coerce').sum():+.0f}")
    print("Backtest: ~$3k per 14 months at $250 a trade; graduates at half that rate after 3 months.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=2.5)
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.once:
        bar_t, spx, sig, z = spx_state()
        ev, tau, ms = ladder(pd.Timestamp.now(tz="UTC"))
        print(f"S&P {spx:.2f} at {bar_t}, 2-min z {z:+.2f}, sigma/min {sig:.5f}; next {SERIES} event {ev or 'none'} "
              f"({len(ms)} strikes, {tau:.0f} min left)")
        return
    last: list = []
    while True:
        t0 = time.time()
        now = pd.Timestamp.now(tz="America/New_York")
        if now.weekday() < 5 and 9.5 <= now.hour + now.minute / 60 <= 16:
            try:
                poll(args.z, last)
                settle()
            except Exception as e:  # a failed poll must not stop the logger
                print(f"{now:%H:%M:%S} poll failed: {e}")
        elif int(t0) % 1800 < args.interval:
            settle()
        time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
