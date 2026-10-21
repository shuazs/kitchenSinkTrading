"""Paper-trade the Iran-to-oil rule live. Logs signals; never places orders.

    python examples/watch_polymarket_iran.py            # runs until Ctrl-C
    python examples/watch_polymarket_iran.py --once     # one snapshot, prints the markets it watches

Run it on your own computer: Polymarket's API is blocked from the project's
cloud sandbox. Every `--interval` seconds it snapshots the open Polymarket
Iran markets and compares them with the snapshot from about an hour earlier,
using the same signed, volume-weighted change as the backtest
(examples/run_iran_oil.py). When the change is `--threshold` (5 points) or
more and oil futures are trading, it logs a paper trade in micro WTI (MCL):
buy for escalation, sell for de-escalation, exit `--hold` hours later. It
fills both ends at the latest CL=F price from Yahoo, so the paper P&L
ignores slippage. Jumps while oil is closed are logged but not traded:
the backtest found they are already in the price when futures reopen.

Log: ~/.cache/kitchensink/polymarket/iran_signals.csv (one row per signal,
exit price and return filled in once the hold ends).
"""
import argparse
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from backtester.data.cache import cache_dir
from backtester.data.polymarket import iran_sign, live_markets, snapshot_change

ET = ZoneInfo("America/New_York")
MCL_BARRELS = 100


def oil_open(now: datetime) -> bool:
    """CME crude: Sunday 18:00 to Friday 17:00 New York time, with a daily 17:00-18:00 halt."""
    t = now.astimezone(ET)
    wd, hm = t.weekday(), t.hour + t.minute / 60  # Monday = 0
    if wd == 5 or (wd == 6 and hm < 18) or (wd == 4 and hm >= 17):
        return False
    return not (17 <= hm < 18)


def oil_price() -> float | None:
    import yfinance as yf

    try:
        px = yf.download("CL=F", period="1d", interval="1m", progress=False)["Close"].dropna()
        return float(px.iloc[-1].item() if hasattr(px.iloc[-1], "item") else px.iloc[-1])
    except Exception as e:  # a missed quote must not stop the watcher
        print(f"  price lookup failed: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--hold", type=float, default=3.0, help="hours")
    ap.add_argument("--lookback", type=float, default=60.0, help="minutes compared, as the backtest's 1h bars")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between snapshots")
    ap.add_argument("--cooldown", type=float, default=6.0, help="hours between signals")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    log_path = cache_dir("polymarket") / "iran_signals.csv"
    log = pd.read_csv(log_path, parse_dates=["time", "exit_due"]) if log_path.exists() else pd.DataFrame()
    snaps: deque[tuple[datetime, pd.DataFrame]] = deque()
    last_signal = log["time"].max().to_pydatetime() if len(log) else None

    while True:
        now = datetime.now(timezone.utc)
        try:
            snap = live_markets()
        except Exception as e:
            print(f"{now:%Y-%m-%d %H:%M} snapshot failed: {e}")
            time.sleep(args.interval)
            continue
        signed = snap[[iran_sign(q, e) != 0 for q, e in zip(snap["question"], snap["event_title"])]]
        if args.once:
            print(f"{len(signed)} signed Iran markets of {len(snap)} matched:")
            print(signed.sort_values("volume_1wk", ascending=False).head(25).to_string(index=False))
            return
        snaps.append((now, snap))
        while len(snaps) > 1 and now - snaps[1][0] >= timedelta(minutes=args.lookback):
            snaps.popleft()
        then, old = snaps[0]
        move = snapshot_change(old, snap) if now - then >= timedelta(minutes=args.lookback * 0.9) else 0.0
        print(f"{now:%Y-%m-%d %H:%M} UTC  {len(signed)} markets  change vs {then:%H:%M}: {move * 100:+.1f} pts")

        cooled = last_signal is None or now - last_signal.replace(tzinfo=timezone.utc) >= timedelta(hours=args.cooldown)
        if abs(move) >= args.threshold and cooled:
            is_open = oil_open(now)
            side = "BUY" if move > 0 else "SELL"
            row = {
                "time": now.replace(tzinfo=None), "change_pts": round(move * 100, 2), "oil_open": is_open,
                "action": f"{side} 1 MCL" if is_open else "none (oil closed)",
                "entry": oil_price() if is_open else None, "exit_due": (now + timedelta(hours=args.hold)).replace(tzinfo=None),
                "exit": None, "ret_pct": None, "pnl_usd": None,
            }
            print(f"  SIGNAL: {row['action']} at {row['entry']}, exit due {row['exit_due']:%H:%M} UTC")
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
            last_signal = now.replace(tzinfo=None)

        if len(log):
            due = log["entry"].notna() & log["exit"].isna() & (log["exit_due"] <= now.replace(tzinfo=None))
            for i in log.index[due]:
                px = oil_price()
                if px is None:
                    continue
                sign = 1 if log.at[i, "action"].startswith("BUY") else -1
                log.at[i, "exit"] = px
                log.at[i, "ret_pct"] = round(sign * (px / log.at[i, "entry"] - 1) * 100, 3)
                log.at[i, "pnl_usd"] = round(sign * (px - log.at[i, "entry"]) * MCL_BARRELS, 2)
                print(f"  EXIT {log.at[i, 'action']}: {log.at[i, 'ret_pct']:+.2f}% (${log.at[i, 'pnl_usd']:+.0f})")
            log.to_csv(log_path, index=False)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
