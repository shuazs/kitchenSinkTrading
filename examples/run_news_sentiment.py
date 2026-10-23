"""FinBERT news and Reddit sentiment: stock news, WSB buzz fade, and Kalshi S&P/BTC.

Scoring is the slow part (FinBERT runs ~40-180 headlines/s on 4 CPUs), so the
scored files are read from NEWS_SENTIMENT_DATA (default
/mnt/project-files/finbert-sentiment):

- benzinga_scores.parquet: published, symbol, s (one row per headline x symbol)
- reddit_posts.parquet: t, subreddit, title, tick (list of tickers), s (FinBERT, NaN if not scored)
- kalshi_book_spx.parquet / kalshi_book_btc.parquet: day, yes, result (nearest strike at entry)
- reddit_daily_mood.parquet: daily FinBERT mood and post counts per subreddit

To rebuild them: data/news_mirrors.load_benzinga + reddit_month, then data/sentiment.score.

    python examples/run_news_sentiment.py
"""
import os

import numpy as np
import pandas as pd

from backtester.data.yahoo_batch import prefetch
from backtester.data.yahoo import load_bars
from backtester.news_sentiment import (bucket_table, cost_for, fade_book, kalshi_directional, mention_spikes,
                                       spread_by_year, stock_events)

ROOT = os.environ.get("NEWS_SENTIMENT_DATA", "/mnt/project-files/finbert-sentiment")


def panel(symbols, start):
    have = prefetch(list(symbols) + ["SPY"])
    bars = {s: load_bars(s, start=start) for s in have}
    f = lambda k: pd.DataFrame({s: b[k] for s, b in bars.items()})
    return f("open"), f("close"), f("volume")


def stock_news():
    news = pd.read_parquet(f"{ROOT}/benzinga_scores.parquet")
    top = news.symbol.value_counts()
    O, C, V = panel(top[top >= 100].index, "2009-01-01")
    ev = stock_events(news, O, C, V)
    ev = ev[(ev.open > 3) & (ev.dv > 1e6)]
    for name, x in [("2009-2016", ev[ev["T"].dt.year <= 2016]), ("2017-2020 holdout", ev[ev["T"].dt.year > 2016])]:
        print(f"\nStock news, {name}: {len(x)} stock-days")
        print(bucket_table(x).round(1).to_string())
        d, t = spread_by_year(x[x.pre])
        print(f"pre-open very positive minus very negative, open->close excess: {1e4 * d.mean():+.1f} bp/day (t {t:.1f})")


def wsb_fade():
    r = pd.read_parquet(f"{ROOT}/reddit_posts.parquet")
    r = r[r.subreddit != "cryptocurrency"].explode("tick").dropna(subset=["tick"])
    O, C, V = panel(r.tick.value_counts().index[:600], "2020-06-01")
    sessions = C["SPY"].dropna().index
    local = r.t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    anchor = (local - pd.Timedelta(hours=9, minutes=25)).dt.normalize() + pd.Timedelta(days=1)
    pos = np.searchsorted(sessions.values, anchor.values.astype(sessions.values.dtype))
    r = r.assign(T=sessions.values[np.minimum(pos, len(sessions) - 1)])[pos < len(sessions) - 5]
    g = r[r.tick.isin(C.columns)].groupby(["T", "tick"]).size().rename("m").reset_index()
    counts = g.pivot(index="T", columns="tick", values="m").reindex(sessions).fillna(0)
    idx = pd.MultiIndex.from_frame(g[["T", "tick"]])
    look = lambda tab: tab.stack().reindex(idx).values
    g["ab"] = look(mention_spikes(counts))
    g["oc"], g["o5"], g["gap"] = look(C / O - 1), look(C.shift(-4) / O - 1), look(O / C.shift(1) - 1)
    g["dv"] = look((C * V).rolling(20).median().shift(1))
    g = g.dropna(subset=["oc", "ab", "dv"])
    g = g[g.dv > 2e6].assign(cost=lambda x: cost_for(x.dv))
    for name, x in [("2021-2022", g[g["T"].dt.year <= 2022]), ("2023-2025 holdout", g[g["T"].dt.year > 2022])]:
        tr, _ = fade_book(x, min_ratio=10, min_mentions=3, hold=1)
        print(f"\nFade WSB spikes (>=10x usual mentions), short open->close, {name}: {len(tr)} trades, "
              f"{1e4 * tr.pnl.mean():+.0f} bp/trade net, win {np.mean(tr.pnl > 0):.0%}, worst {tr.pnl.min():+.0%}, "
              f"t {tr.pnl.mean() / tr.pnl.std() * np.sqrt(len(tr)):.1f} (per trade)")


def kalshi():
    mood = pd.read_parquet(f"{ROOT}/reddit_daily_mood.parquet")
    z = lambda s: (s - s.rolling(60, min_periods=20).mean()) / s.rolling(60, min_periods=20).std()
    sig = {"spx": z(mood[["mood_wallstreetbets", "mood_stocks", "mood_investing"]].mean(axis=1)),
           "btc": z(mood["mood_cryptocurrency"])}
    for name, fee in [("spx", 0.035), ("btc", 0.07)]:
        book = pd.read_parquet(f"{ROOT}/kalshi_book_{name}.parquet").set_index("day")
        book = book[(book.yes > 0.1) & (book.yes < 0.9)]
        for thr in (0.5, 1.0):
            tr = kalshi_directional(book, sig[name], threshold=thr, fee_mult=fee)
            print(f"Kalshi {name.upper()} nearest-strike, Reddit mood |z|>{thr}: {len(tr)} trades, "
                  f"{100 * tr.pnl.mean():+.1f} c/contract net, t {tr.pnl.mean() / tr.pnl.std() * np.sqrt(len(tr)):.1f}")


if __name__ == "__main__":
    stock_news()
    wsb_fade()
    kalshi()
