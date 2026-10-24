"""Forward-test two Kalshi rules on paper; never places orders.

    python paper_loggers/kalshi_near_misses.py --once     # list the markets each rule would buy now, then stop
    python paper_loggers/kalshi_near_misses.py            # poll every --interval minutes until Ctrl-C
    python paper_loggers/kalshi_near_misses.py --report   # score the paper trades that have settled

Run it on your own computer: Kalshi's API is blocked from the project's cloud
sandbox. The alpha loop (Oct 2026) found two rules that held up in backtest
but whose Kalshi history is too short for a clean out-of-sample test:

- mentions_no: on "mentions" markets (will X say a word), buy NO when it costs
  20-80c. Backtest +3.4c/contract after fees on Jul 2025+ data, t 5.7.
- atp_yes: on ATP men's match winners, buy YES when it costs 80-95c.
  Backtest +2.6c/contract on Jul 2025+, t 6.4, but a thin fit sample.

Each poll lists the open markets of the watched series, and the first time a
market's quote falls in a rule's band it records one paper buy of
--contracts at the ask (the price a taker pays), with Kalshi's taker fee. Later
polls check unsettled trades and fill in the result and net P&L once Kalshi
settles the market. After about three months, `--report` gives the net cents
per contract and an event-clustered t-stat; the rule counts as confirmed if it
keeps at least half its backtest edge (protocol.md P4).

Each paper buy also records the market's open time and how many minutes after
it the entry came: batches 11-12 found the mentions edge comes mostly from the
first hour of trading, and `--report` splits on it. Paper fills are at the ask
of the side bought (a print on the other side would understate the cost by
the spread).

Logs in ~/.cache/kitchensink/kalshi_near_misses/: trades.csv (one row per paper
buy) and quotes.csv (every in-band quote, for checking spreads later).
"""
import argparse
import csv
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import _common  # noqa: F401  (loads keys.env)
from backtester.data.cache import cache_dir
from backtester.data.live_books import (NEAR_MISS_RULES, kalshi_market, kalshi_mention_series, kalshi_series_markets,
                                        market_top, paper_entry, paper_pnl)

DIR = cache_dir("kalshi_near_misses")
TRADES, QUOTES = DIR / "trades.csv", DIR / "quotes.csv"
TRADE_FIELDS = ["entry_time", "rule", "series", "event_ticker", "ticker", "title", "side", "price", "contracts",
                "open_time", "mins_since_open", "close_time", "result", "net", "settled_time"]
QUOTE_FIELDS = ["time", "rule", "ticker", "yes_bid", "yes_ask", "volume"]
# Largest mentions series in the Kalshi history (2021 to Jan 2026); used when the /series lookup fails.
MENTION_SERIES = ["KXNFLMENTION", "KXTRUMPMENTION", "KXTRUMPMENTIONB", "KXNBAMENTION", "KXMAMDANIMENTION",
                  "KXSECPRESSMENTION", "KXTRUMPSAY", "KXNCAAMENTION", "KXLATENIGHTMENTION", "KXVANCEMENTION",
                  "KXSNFMENTION", "KXKIMMELMENTION", "KXTNFMENTION", "KXCOLBERTMENTION", "KXCONGRESSMENTION",
                  "KXBERNIEMENTION", "KXNEWSOMMENTION", "KXMRBEASTMENTION", "KXPOWELLMENTION", "KXSNLMENTION",
                  "KXTRUMPSAYMONTH", "KXFOXNEWSMENTION", "KXSTARMERMENTION"]


def watched_series() -> dict[str, list[str]]:
    mentions = kalshi_mention_series() or MENTION_SERIES
    return {"mentions_no": sorted(set(mentions)), "atp_yes": NEAR_MISS_RULES["atp_yes"]["series"]}


def load_trades() -> pd.DataFrame:
    if TRADES.exists():
        return pd.read_csv(TRADES, dtype={"result": str})
    return pd.DataFrame(columns=TRADE_FIELDS)


def scan(series: dict[str, list[str]], trades: pd.DataFrame, contracts: int, once: bool) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    taken = set(zip(trades["rule"], trades["ticker"]))
    new, quotes = [], []
    for rule, names in series.items():
        for s in names:
            try:
                markets = kalshi_series_markets(s)
            except requests.RequestException as e:
                print(f"  {s}: {e}")
                continue
            for m in markets:
                bid, ask = market_top(m)
                hit = paper_entry(rule, bid, ask)
                if hit is None:
                    continue
                quotes.append({"time": now, "rule": rule, "ticker": m["ticker"], "yes_bid": bid, "yes_ask": ask,
                               "volume": m.get("volume")})
                if (rule, m["ticker"]) in taken:
                    continue
                side, price = hit
                taken.add((rule, m["ticker"]))
                opened = pd.to_datetime(m.get("open_time"), utc=True, errors="coerce")
                age = round((pd.Timestamp(now) - opened).total_seconds() / 60) if pd.notna(opened) else np.nan
                new.append({"entry_time": now, "rule": rule, "series": s, "event_ticker": m.get("event_ticker"),
                            "ticker": m["ticker"], "title": m.get("title") or m.get("yes_sub_title"), "side": side,
                            "price": price, "contracts": contracts, "open_time": m.get("open_time"),
                            "mins_since_open": age, "close_time": m.get("close_time"),
                            "result": "", "net": np.nan, "settled_time": ""})
                print(f"{now[:19]} {rule:<11} paper buy {contracts} {side.upper()} {m['ticker']} at {price:.2f}")
    if not once and quotes:
        first = not QUOTES.exists()
        with open(QUOTES, "a", newline="") as f:
            w = csv.DictWriter(f, QUOTE_FIELDS)
            if first:
                w.writeheader()
            w.writerows(quotes)
    return pd.concat([trades, pd.DataFrame(new, columns=TRADE_FIELDS)], ignore_index=True) if new else trades


def settle(trades: pd.DataFrame) -> pd.DataFrame:
    """Fill result and net P&L for paper trades whose markets Kalshi has settled."""
    open_ = trades.index[trades["result"].fillna("").eq("")]
    for i in open_:
        try:
            m = kalshi_market(trades.at[i, "ticker"])
        except requests.RequestException:
            continue
        if m.get("result") in ("yes", "no"):
            trades.at[i, "result"] = m["result"]
            trades.at[i, "net"] = paper_pnl(trades.at[i, "side"], float(trades.at[i, "price"]), m["result"],
                                            int(trades.at[i, "contracts"]))
            trades.at[i, "settled_time"] = datetime.now(timezone.utc).isoformat()
    return trades


def report() -> None:
    t = load_trades()
    done = t[t["result"].isin(["yes", "no"])]
    print(f"{len(t)} paper trades, {len(done)} settled")
    bars = {"mentions_no": 3.4, "atp_yes": 2.6}
    for rule, g in done.groupby("rule"):
        ev = g.groupby("event_ticker")["net"].mean()
        tstat = ev.mean() / ev.std() * np.sqrt(len(ev)) if len(ev) > 1 and ev.std() > 0 else np.nan
        need = bars[rule] / 2
        print(f"{rule:<11} {len(g):>4} trades, {len(ev):>4} events, net {100 * g['net'].mean():+.2f}c/contract, "
              f"event t {tstat:.2f}; backtest {bars[rule]:+.1f}c, confirmed if >= {need:+.1f}c over 200+ events")
        age = pd.to_numeric(g["mins_since_open"], errors="coerce")
        for lab, m in (("  entered in the market's first hour", age <= 60), ("  entered later", age > 60)):
            if m.any():  # batch 11-12: the mentions edge sits mostly in the first hour of trading
                print(f"{lab:<38} {m.sum():>4} trades, net {100 * g.loc[m, 'net'].mean():+.2f}c, "
                      f"mean No/Yes price paid {g.loc[m, 'price'].astype(float).mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0, help="minutes between polls")
    ap.add_argument("--contracts", type=int, default=10, help="paper contracts per trade (fee rounds per order)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    series, listed_at = watched_series(), time.time()
    print(f"watching {len(series['mentions_no'])} mentions series and {len(series['atp_yes'])} ATP series")
    while True:
        t0 = time.time()
        if t0 - listed_at > 6 * 3600:  # pick up newly launched mentions series
            series, listed_at = watched_series(), t0
        trades = scan(series, load_trades() if not args.once else pd.DataFrame(columns=TRADE_FIELDS),
                      args.contracts, args.once)
        if args.once:
            return
        trades = settle(trades)
        trades.to_csv(TRADES, index=False)
        time.sleep(max(0.0, 60 * args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
