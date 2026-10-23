"""Paper-trade the WSB mention-spike fade live. Logs signals; never places orders.

    python examples/watch_wsb_spikes.py             # runs until Ctrl-C
    python examples/watch_wsb_spikes.py --once      # one poll + whatever is due now, then exit (for cron)
    python examples/watch_wsb_spikes.py --report    # print the paper results so far

Run it on your own computer: reddit.com is blocked from the project's cloud
sandbox. Every `--interval` seconds it pulls new posts from r/wallstreetbets,
r/stocks, r/investing, r/options and r/StockMarket and counts ticker mentions
in titles, as the backtest did (examples/run_news_sentiment.py):

- 09:25 ET: for today's session (posts since 09:25 ET on the previous session),
  flag up to 3 tickers mentioned >= 10x their average over the prior 20
  sessions and at least 3 times. These are the paper shorts.
- With Alpaca keys set (APCA_API_KEY_ID / APCA_API_SECRET_KEY, a free paper
  account), it also records whether each name is shortable and easy to borrow.
  Whether shares can be borrowed is the main open question for this trade.
- After 16:10 ET: fills each paper short at the day's open and covers at the
  close (Yahoo daily bar), net of the backtest's costs (7.5-40 bp per side by
  dollar volume, plus 0.3% borrow).

Counts need history, so signals start after 10 sessions of running (20 for the
full baseline). The watcher can only backfill the newest ~1,000 posts per
subreddit, so leave it running (or run --once from cron every 5-10 minutes).

Reddit often refuses anonymous requests. If you get 403/429 errors, create a
"script" app at https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET.

Logs in ~/.cache/kitchensink/wsb/: posts.csv (every post seen), counts.csv
(mentions per session), signals.csv (one row per paper short).
"""
import argparse
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from backtester.data.cache import cache_dir
from backtester.news_sentiment import cost_for, load_wsb_tickers, post_session, spike_candidates, ticker_mentions

ET = ZoneInfo("America/New_York")
SUBS = ["wallstreetbets", "stocks", "investing", "options", "StockMarket"]
UA = "kitchensink-wsb-watcher/0.1 (log-only research script)"
BORROW_PER_DAY = 0.003
DIR = cache_dir("wsb")
POSTS, COUNTS, SIGNALS, CLOSED = DIR / "posts.csv", DIR / "counts.csv", DIR / "signals.csv", DIR / "closed_days.txt"
SETTLED = DIR / "settled_days.txt"


class Reddit:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.base, self.expires = "https://www.reddit.com", 0.0

    def _auth(self):
        cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
        if not cid or time.time() < self.expires:
            return
        r = requests.post("https://www.reddit.com/api/v1/access_token", auth=(cid, secret),
                          data={"grant_type": "client_credentials"}, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        tok = r.json()
        self.s.headers["Authorization"] = f"bearer {tok['access_token']}"
        self.base, self.expires = "https://oauth.reddit.com", time.time() + tok.get("expires_in", 3600) - 60

    def new_posts(self, sub, stop_before=0, pages=10):
        """Newest posts of a subreddit, paging back until `stop_before` (unix time) or `pages` pages."""
        rows, after = [], None
        for _ in range(pages):
            self._auth()
            r = self.s.get(f"{self.base}/r/{sub}/new.json", params={"limit": 100, "after": after, "raw_json": 1}, timeout=30)
            r.raise_for_status()
            data = r.json()["data"]
            kids = [c["data"] for c in data["children"]]
            rows += [(k["id"], int(k["created_utc"]), sub.lower(), k.get("title") or "") for k in kids]
            after = data.get("after")
            if not kids or not after or kids[-1]["created_utc"] < stop_before:
                break
            time.sleep(1.0)
        return rows


def settled_days():
    return set(SETTLED.read_text().split()) if SETTLED.exists() else set()


def load_posts():
    if POSTS.exists():
        return pd.read_csv(POSTS, dtype={"id": str})
    return pd.DataFrame(columns=["id", "created_utc", "subreddit", "title"])


def poll(reddit, posts):
    last = posts.groupby("subreddit").created_utc.max().to_dict() if len(posts) else {}
    new = []
    for sub in SUBS:
        try:
            new += reddit.new_posts(sub, stop_before=last.get(sub.lower(), 0))
        except Exception as e:  # one failed subreddit must not stop the watcher
            print(f"  {sub}: {e}")
    new = pd.DataFrame(new, columns=["id", "created_utc", "subreddit", "title"])
    new = new[~new.id.isin(posts.id)].drop_duplicates("id")
    if len(new):
        new.to_csv(POSTS, mode="a", header=not POSTS.exists(), index=False)
    return pd.concat([posts, new], ignore_index=True), len(new)


def sessions_until(day, start):
    """Weekdays from `start` to `day`, minus days the market turned out to be closed."""
    closed = set(CLOSED.read_text().split()) if CLOSED.exists() else set()
    return pd.DatetimeIndex([d for d in pd.bdate_range(start, day) if d.strftime("%Y-%m-%d") not in closed])


def session_counts(posts, day, tickers):
    """Session x ticker mention counts, from the first session fully covered by the log up to `day`."""
    first = pd.Timestamp(posts.created_utc.min(), unit="s", tz="UTC").tz_convert(ET)
    # the first session whose 24h window starts after logging began
    start = (first + timedelta(days=1)).date() if first.hour * 60 + first.minute > 9 * 60 + 25 else first.date()
    start = pd.Timestamp(start) + pd.offsets.BDay(1)
    sessions = sessions_until(day, start)
    if len(sessions) == 0:
        return pd.DataFrame()
    p = posts.assign(T=post_session(posts.created_utc, sessions), tick=ticker_mentions(posts.title, tickers))
    p = p.dropna(subset=["T"]).explode("tick").dropna(subset=["tick"])
    p = p[p["T"] >= sessions[0]]
    c = p.groupby(["T", "tick"]).size().unstack(fill_value=0)
    return c.reindex(sessions, fill_value=0)


def borrow_status(symbol):
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key:
        return "", ""
    try:
        r = requests.get(f"https://paper-api.alpaca.markets/v2/assets/{symbol}", timeout=20,
                         headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        a = r.json()
        return a.get("shortable", ""), a.get("easy_to_borrow", "")
    except Exception as e:
        print(f"  Alpaca asset lookup failed for {symbol}: {e}")
        return "", ""


def load_signals():
    if SIGNALS.exists():
        return pd.read_csv(SIGNALS, parse_dates=["day"])
    return pd.DataFrame(columns=["day", "ticker", "mentions", "base", "ratio", "shortable", "easy_to_borrow",
                                 "open", "close", "dollar_volume", "ret", "net_short", "status"])


def make_signals(posts, day, tickers):
    sig = load_signals()
    if (sig.day == day).any():
        return
    counts = session_counts(posts, day, tickers)
    if len(counts):
        counts.loc[[day]].stack().rename("mentions").reset_index().to_csv(
            COUNTS, mode="a", header=not COUNTS.exists(), index=False)
    if len(counts) < 11:  # mention_spikes needs 10 prior sessions
        print(f"{day.date()}: warming up, {len(counts)} sessions logged (signals start after 10)")
        row = pd.DataFrame([{"day": day, "ticker": "", "status": f"warmup {len(counts)}"}])
    else:
        cand = spike_candidates(counts, day)
        if cand.empty:
            print(f"{day.date()}: no spikes")
            row = pd.DataFrame([{"day": day, "ticker": "", "status": "none"}])
        else:
            cand["shortable"], cand["easy_to_borrow"] = zip(*[borrow_status(t) for t in cand.ticker])
            row = cand.assign(day=day, status="open")
            print(f"{day.date()}: paper SHORT at the open -> " + ", ".join(
                f"{r.ticker} ({r.mentions} mentions, {r.ratio:.0f}x, shortable={r.shortable})" for r in row.itertuples()))
    pd.concat([sig, row], ignore_index=True).to_csv(SIGNALS, index=False)


def settle(day):
    """Fill today's open signals at the Yahoo open and close."""
    import yfinance as yf

    sig = load_signals()
    todo = sig[(sig.day == day) & (sig.status == "open")]
    spy = yf.download("SPY", start=day, end=day + timedelta(days=1), progress=False, auto_adjust=False)
    if spy.empty:
        CLOSED.open("a").write(day.strftime("%Y-%m-%d") + "\n")
        sig.loc[sig.day == day, "status"] = "market closed"
        sig.to_csv(SIGNALS, index=False)
        print(f"{day.date()}: market closed, day dropped from the counts")
        return
    for i, r in todo.iterrows():
        try:
            px = yf.download(r.ticker, start=day - timedelta(days=40), end=day + timedelta(days=1),
                             progress=False, auto_adjust=False)
            if isinstance(px.columns, pd.MultiIndex):
                px.columns = px.columns.get_level_values(0)
            bar = px.loc[px.index.normalize() == day]
            if bar.empty:
                sig.loc[i, "status"] = "no bar"
                continue
            o, c = float(bar.Open.iloc[0]), float(bar.Close.iloc[0])
            dv = float((px.Close * px.Volume).iloc[:-1].tail(20).median())
            ret = c / o - 1
            sig.loc[i, ["open", "close", "dollar_volume", "ret"]] = [o, c, dv, ret]
            sig.loc[i, "net_short"] = -ret - 2 * float(cost_for([dv])[0]) - BORROW_PER_DAY
            sig.loc[i, "status"] = "filled"
        except Exception as e:
            print(f"  price lookup failed for {r.ticker}: {e}")
    sig.to_csv(SIGNALS, index=False)
    f = sig[(sig.day == day) & (sig.status == "filled")]
    for r in f.itertuples():
        print(f"{day.date()}: {r.ticker} open {r.open:.2f} close {r.close:.2f}, paper short {100 * r.net_short:+.1f}% net")


def report():
    sig = load_signals()
    f = sig[sig.status == "filled"]
    print(f"{sig.day.nunique()} sessions logged, {len(f)} paper shorts filled")
    if len(f):
        print(f"net per trade {100 * f.net_short.mean():+.2f}% (median {100 * f.net_short.median():+.2f}%), "
              f"win {np.mean(f.net_short > 0):.0%}, worst {100 * f.net_short.min():+.1f}%  [backtest: +1.8-2.2%, 57%]")
        known = f.shortable.notna() & f.shortable.astype(str).ne("")
        if known.any():
            k = f[known].astype(str)
            print(f"shortable at Alpaca: {k.shortable.str.lower().eq('true').mean():.0%}, "
                  f"easy to borrow: {k.easy_to_borrow.str.lower().eq('true').mean():.0%}")
        print(f[["day", "ticker", "mentions", "ratio", "shortable", "open", "close", "net_short"]].tail(20).to_string(index=False))


def step(reddit, posts, tickers):
    posts, n = poll(reddit, posts)
    now = datetime.now(ET)
    day = pd.Timestamp(now.date())
    print(f"{now:%Y-%m-%d %H:%M} ET: +{n} posts ({len(posts)} logged)")
    if now.weekday() < 5 and len(posts):
        hm = now.hour * 60 + now.minute
        if 9 * 60 + 25 <= hm < 16 * 60:
            make_signals(posts, day, tickers)
        elif hm >= 16 * 60 + 10 and day.strftime("%Y-%m-%d") not in settled_days():
            sig = load_signals()
            if not (sig.day == day).any():
                make_signals(posts, day, tickers)  # started late: log the day's spikes anyway
            settle(day)
            SETTLED.open("a").write(day.strftime("%Y-%m-%d") + "\n")
    return posts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=300, help="seconds between polls")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    reddit, posts, tickers = Reddit(), load_posts(), load_wsb_tickers()
    print(f"Logging to {DIR}")
    while True:
        try:
            posts = step(reddit, posts, tickers)
        except Exception as e:  # keep watching through network errors
            print(f"error: {e}")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
