"""Log Kalshi and Polymarket US game books side by side; never places orders.

    python examples/watch_kalshi_polymarket_us.py --once     # list the paired games and their quotes, then stop
    python examples/watch_kalshi_polymarket_us.py            # log every --interval seconds until Ctrl-C
    python examples/watch_kalshi_polymarket_us.py --report   # replay the rule on the log, net of the 5% US fee

Run it on your own computer during NBA/NFL/NHL/MLB/soccer games: both APIs are
blocked from the project's cloud sandbox. The backtest (run_event_markets.py
--families) found Polymarket's international book trails Kalshi on live
games: when the two differ by 5c+, buying the Kalshi side on Polymarket 60s
later and selling 60 min later made +3.6c/share after the US taker fee.
Polymarket US is a separate book with no public history, so this logs both
books to see whether the lag exists there too.

Every poll writes the best bid and ask on both venues for each paired game to
~/.cache/kitchensink/kalshi_polyus/quotes.csv and prints a line when the mid
gap first reaches --threshold. `--report` replays the rule on that log
(event_markets.gap_followthrough). Pairs come from team codes in the Kalshi
ticker and the Polymarket slug; check the `--once` list, and put any fixes in
a CSV (kalshi_ticker,poly_slug,flip) passed with --pairs.
"""
import argparse
import csv
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from backtester.data.cache import cache_dir
from backtester.data.live_books import GAME_SERIES, kalshi_book, kalshi_games, poly_us_book, poly_us_markets
from backtester.event_markets import gap_followthrough, pair_games

LOG = cache_dir("kalshi_polyus") / "quotes.csv"
FIELDS = ["time", "kalshi_ticker", "poly_slug", "k_bid", "k_ask", "p_bid", "p_ask"]


def load_pairs(path: str | None) -> list[dict]:
    km, pm = kalshi_games(GAME_SERIES), poly_us_markets()
    pairs = {p["kalshi_ticker"]: p for p in pair_games(km, pm)}
    if path:
        for r in pd.read_csv(path).to_dict("records"):
            pairs[r["kalshi_ticker"]] = {"kalshi_ticker": r["kalshi_ticker"], "poly_slug": r["poly_slug"], "flip": bool(r.get("flip", False))}
    print(f"{len(km)} Kalshi game markets, {len(pm)} Polymarket US markets, {len(pairs)} pairs")
    return list(pairs.values())


def snapshot(p: dict) -> dict | None:
    try:
        kb, ka = kalshi_book(p["kalshi_ticker"])
        pb, pa = poly_us_book(p["poly_slug"])
    except requests.RequestException as e:
        print(f"  {p['kalshi_ticker']}: {e}")
        return None
    if p["flip"]:
        pb, pa = (None if pa is None else round(1 - pa, 4)), (None if pb is None else round(1 - pb, 4))
    return {"time": datetime.now(timezone.utc).isoformat(), "kalshi_ticker": p["kalshi_ticker"], "poly_slug": p["poly_slug"],
            "k_bid": kb, "k_ask": ka, "p_bid": pb, "p_ask": pa}


def mid(b, a):
    return None if b is None or a is None else (b + a) / 2


def report(threshold: float) -> None:
    q = pd.read_csv(LOG, parse_dates=["time"]).dropna(subset=["k_bid", "k_ask", "p_bid", "p_ask"])
    print(f"{len(q)} quotes, {q['kalshi_ticker'].nunique()} markets, {q['time'].min()} to {q['time'].max()}")
    for lat, hold in [("2s", "15min"), ("60s", "15min"), ("60s", "60min")]:
        r = gap_followthrough(q, threshold, lat, hold)
        if len(r):
            print(f"{lat} late, {hold} hold: {len(r)} trades on {r['kalshi_ticker'].nunique()} markets, "
                  f"net {100 * r['ret'].mean():+.2f}c/share, Polymarket mid moved {100 * r['toward'].mean():+.2f}c toward Kalshi")
        else:
            print(f"{lat} late, {hold} hold: no signals yet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    ap.add_argument("--refresh", type=float, default=30.0, help="minutes between re-pairing games")
    ap.add_argument("--pairs", help="CSV of kalshi_ticker,poly_slug,flip to add or override pairs")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.threshold)
        return

    pairs, paired_at, last_gap = load_pairs(args.pairs), time.time(), {}
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        while True:
            if time.time() - paired_at > 60 * args.refresh:
                pairs, paired_at = load_pairs(args.pairs), time.time()
            t0 = time.time()
            for p in pairs:
                s = snapshot(p)
                if s is None:
                    continue
                if args.once:
                    print(f"{s['kalshi_ticker']:<34} {s['poly_slug']:<36} flip={p['flip']!s:<5} "
                          f"Kalshi {s['k_bid']}/{s['k_ask']}  Poly US {s['p_bid']}/{s['p_ask']}")
                    continue
                w.writerow(s)
                km, pm = mid(s["k_bid"], s["k_ask"]), mid(s["p_bid"], s["p_ask"])
                if km is not None and pm is not None:
                    gap, prev = km - pm, last_gap.get(p["kalshi_ticker"], 0.0)
                    if abs(gap) >= args.threshold > abs(prev):
                        side = "buy" if gap > 0 else "sell"
                        print(f"{s['time'][:19]} {p['kalshi_ticker']}: Kalshi {km:.3f} vs Poly US {pm:.3f}, "
                              f"paper {side} on Polymarket US")
                    last_gap[p["kalshi_ticker"]] = gap
            if args.once:
                return
            f.flush()
            time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
