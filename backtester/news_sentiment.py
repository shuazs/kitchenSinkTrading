"""Trading off FinBERT news and Reddit sentiment.

Three tests, all with the text made tradable only after it was public:

* stock news: FinBERT score of a stock's headlines -> buy (or short) it at the
  next open, hold to that close or 5 days (`session_for`, `stock_events`,
  `bucket_table`, `spread_by_year`).
* Reddit buzz: ticker mentions in r/wallstreetbets & co. (`cashtag_whitelist`,
  `ticker_mentions`, `mention_spikes`), faded with a short at the open
  (`fade_book`).
* Kalshi: a daily mood signal decides whether to buy YES or NO of the
  S&P/BTC "closes above K" contract nearest the price at entry
  (`kalshi_directional`).

Scores come from `data.sentiment.score` (P(positive) - P(negative)).
Data loaders are in `data/news_mirrors.py`.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

# per-side cost (half spread + slippage) by 20-day median dollar volume
COST_TIERS = [(5e7, 7.5e-4), (1e7, 15e-4), (0, 40e-4)]
STOPWORDS = set("""YOU ME WE THEY DOW A I AM AN AND ARE AS AT BE BIG BUY BY CAN CEO CFO DD DO EPS ETF EV FOR GO HAS HE
IF IN IPO IS IT ITM OTM MY NO NOT NOW OF ON OR OUT PM SO TA THE TO UP US USA YOLO ATH IMO LOL FD FDS ALL ONE NEW
GOOD REAL LOVE FUN PLAY OPEN CASH HOLD SELL MOON APE APES GAIN GAINS LOSS RH TD WSB SEC FED CPI GDP ATM IV DTE EOD EOW
AH PT USD EUR UK EU AI OP EDIT TLDR OK DM PSA FYI LMAO WTF RIP HODL BTC ETH IRA ROTH LLC INC CO LTD NYSE NASDAQ OTC
SPAC FOMO MOASS YTD QE TV PC UI API ANY GOT HUGE HOT RUN DUE""".split())


def cost_for(dollar_volume):
    """Per-side cost fraction for each dollar-volume value."""
    dv = np.asarray(dollar_volume, dtype=float)
    out = np.full(dv.shape, COST_TIERS[-1][1])
    for floor, c in reversed(COST_TIERS[:-1]):
        out = np.where(dv > floor, c, out)
    return out


def session_for(ts, sessions, cutoff="09:25", tz="America/New_York"):
    """Map timestamps to the first session whose open can act on them.

    Text published before `cutoff` (local) on a session day trades that day's
    open; anything later (intraday, after the close, weekends) trades the next
    session's open. Returns (session dates, published-before-the-open flags).
    """
    t = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    h, m = map(int, cutoff.split(":"))
    pre = (t.dt.hour * 60 + t.dt.minute) < h * 60 + m
    day = t.dt.normalize()
    days = pd.DatetimeIndex(sessions).values.astype("datetime64[ns]")
    pos = np.searchsorted(days, day.values.astype("datetime64[ns]"))
    same = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == day.values.astype("datetime64[ns]"))
    pos = np.where(same & ~pre.values, pos + 1, pos)
    out = pd.Series(pd.NaT, index=range(len(t)), dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out.values, pre.values


def stock_events(news, opens, closes, volumes, bench="SPY", cutoff="09:25"):
    """One row per (session T, symbol): mean score `s`, count `n`, `pre` (any headline before the open),
    returns open->close `oc`, open T->close T+4 `o5`, overnight `gap`, market-excess `oc_x`/`o5_x`,
    20-day median dollar volume `dv` and per-side `cost`.

    `news` needs columns published, symbol, s (one row per headline and symbol).
    """
    sessions = closes[bench].dropna().index
    T, pre = session_for(news.published, sessions, cutoff)
    x = news.assign(T=T, pre=pre).dropna(subset=["T"])
    x = x[x.symbol.isin(closes.columns)]
    g = x.groupby(["T", "symbol"]).agg(s=("s", "mean"), n=("s", "size"), pre=("pre", "max")).reset_index()
    idx = pd.MultiIndex.from_frame(g[["T", "symbol"]])
    tabs = {"oc": closes / opens - 1, "o5": closes.shift(-4) / opens - 1, "gap": opens / closes.shift(1) - 1,
            "dv": (closes * volumes).rolling(20).median().shift(1), "open": opens}
    for k, tab in tabs.items():
        g[k] = tab.stack().reindex(idx).values
    for k in ("oc", "o5"):
        g[k + "_x"] = g[k] - tabs[k][bench].reindex(g["T"]).values
    g = g.dropna(subset=["oc", "dv"])
    g["cost"] = cost_for(g.dv)
    return g


def bucket_table(events, edges=(-1.01, -0.6, -0.2, 0.2, 0.6, 1.01), labels=("vneg", "neg", "neu", "pos", "vpos")):
    """Mean gap and market-excess returns (bp) by sentiment bucket."""
    b = pd.cut(events.s, list(edges), labels=list(labels))
    return events.groupby(b, observed=True).agg(
        n=("oc", "size"), gap_bp=("gap", lambda v: 1e4 * v.mean()),
        oc_x_bp=("oc_x", lambda v: 1e4 * v.mean()), o5_x_bp=("o5_x", lambda v: 1e4 * v.mean()))


def spread_by_year(events, col="oc_x", hi=0.6):
    """Daily (very positive mean) - (very negative mean) of `col`; returns (daily series, t-stat)."""
    p = events[events.s > hi].groupby("T")[col].mean()
    n = events[events.s < -hi].groupby("T")[col].mean()
    d = (p - n).dropna()
    return d, d.mean() / d.std() * np.sqrt(len(d))


def cashtag_whitelist(titles, min_count=30, stop=STOPWORDS):
    """Tickers written as $XYZ at least `min_count` times; bare XYZ is only trusted for these."""
    c = Counter(m for t in titles for m in re.findall(r"\$([A-Z]{1,5})\b", t or ""))
    return {k for k, v in c.items() if v >= min_count and k not in stop}


def ticker_mentions(titles, whitelist):
    """List of whitelisted tickers named in each title ($XYZ or bare XYZ)."""
    pat = re.compile(r"(?:\$|\b)([A-Z]{2,5})\b")
    return [sorted({m for m in pat.findall(t or "") if m in whitelist}) for t in titles]


def mention_spikes(counts, window=20, min_periods=10):
    """Abnormal buzz: (mentions + 1) / (trailing mean of the prior `window` sessions + 1). Date x ticker in, same out."""
    base = counts.rolling(window, min_periods=min_periods).mean().shift(1)
    return (counts + 1) / (base + 1)


def fade_book(events, min_ratio=10, min_mentions=3, min_gap=-1.0, hold=1, max_names=3, borrow_per_day=0.003):
    """Short the most-hyped names at the open: buzz ratio `ab` >= min_ratio and mentions `m` >= min_mentions.

    `events` rows: T, ab, m, gap, oc, o5, cost. Returns (trades with net `pnl`, daily mean pnl / hold).
    Short P&L = -return - round-trip cost - borrow fee per day held.
    """
    x = events[(events.ab >= min_ratio) & (events.m >= min_mentions) & (events.gap >= min_gap)]
    x = x.sort_values("ab", ascending=False).groupby("T").head(max_names)
    col = "oc" if hold == 1 else "o5"
    x = x.assign(pnl=-x[col] - 2 * x.cost - borrow_per_day * hold)
    return x, x.groupby("T").pnl.mean() / hold


def kalshi_fee(price, mult=0.07):
    """Taker fee per contract in dollars, rounded up to the cent."""
    return math.ceil(100 * mult * price * (1 - price) - 1e-9) / 100


def kalshi_directional(book, signal, threshold=1.0, slippage=0.01, fee_mult=0.07):
    """Buy YES when signal > threshold, NO when < -threshold, hold to settlement.

    `book` rows (indexed by day): yes (last YES price before entry, 0-1), result (1 if YES settled).
    Returns per-contract net P&L in dollars for each trade.
    """
    x = book.join(signal.rename("sig"), how="inner").dropna(subset=["sig"])
    out = []
    for day, r in x.iterrows():
        if r.sig > threshold:
            p = r.yes + slippage
            out.append((day, "yes", r.result - p - kalshi_fee(p, fee_mult)))
        elif r.sig < -threshold:
            p = 1 - r.yes + slippage
            out.append((day, "no", (1 - r.result) - p - kalshi_fee(p, fee_mult)))
    return pd.DataFrame(out, columns=["day", "side", "pnl"]).set_index("day")


def load_wsb_tickers(path=None):
    """The ticker whitelist the backtest used (`data/wsb_tickers.txt`)."""
    from pathlib import Path

    path = Path(path or Path(__file__).parent / "data" / "wsb_tickers.txt")
    return {t.strip() for t in path.read_text().splitlines() if t.strip() and not t.startswith("#")}


def post_session(created_utc, sessions, cutoff="09:25", tz="America/New_York"):
    """Session whose `cutoff` first follows each post (posts after 09:25 count toward the next session)."""
    t = pd.to_datetime(pd.Series(created_utc), unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    h, m = map(int, cutoff.split(":"))
    anchor = (t - pd.Timedelta(hours=h, minutes=m)).dt.normalize() + pd.Timedelta(days=1)
    days = pd.DatetimeIndex(sessions).values.astype("datetime64[ns]")
    pos = np.searchsorted(days, anchor.values.astype("datetime64[ns]"))
    out = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(days)
    out[ok] = days[pos[ok]]
    return out


def spike_candidates(counts, day, min_ratio=10, min_mentions=3, max_names=3, window=20, min_periods=10):
    """Live version of the fade filter: rank tickers on `day` by mentions vs their prior-session average.

    `counts` is session x ticker mention counts that includes `day` as its last row.
    Returns a frame (ticker, mentions, base, ratio) of at most `max_names` that pass.
    """
    ab = mention_spikes(counts, window, min_periods)
    base = counts.rolling(window, min_periods=min_periods).mean().shift(1)
    x = pd.DataFrame({"mentions": counts.loc[day], "base": base.loc[day], "ratio": ab.loc[day]}).dropna()
    x = x[(x.ratio >= min_ratio) & (x.mentions >= min_mentions)].sort_values("ratio", ascending=False)
    return x.head(max_names).rename_axis("ticker").reset_index()
