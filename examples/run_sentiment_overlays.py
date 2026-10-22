"""Market-wide sentiment as a forecaster and as an overlay on the keepers.

    python examples/run_sentiment_overlays.py [out_dir]

1. Does sentiment predict S&P returns? Forward 5/21/63-day SPY returns by
   quintile of a Fear & Greed style mood index (and its parts, VIX, VVIX,
   SKEW, Michigan sentiment), design 2007-18 vs holdout 2019-26.
2. Overlays fixed in advance from the literature, on UPRO vol target, SPY and
   multi-asset trend: buy extreme fear, trim extreme greed, flat in VIX
   backwardation, halve when credit weakens, lever up when Michigan
   sentiment is low. Plus a walk-forward pick of the mood thresholds.
3. Crypto: BTC/ETH perpetual funding (crowded longs / shorts paying) on
   BTC+ETH breakout, ensemble trend and BTC hold, design May 2021-2023 vs
   holdout 2024-Feb 2026 (funding data ends there).
4. ETF breakout trades (the meta-label keeper's primary) by mood at entry,
   before the locked July 2024 holdout only.

Sentiment surveys (AAII, Baker-Wurgler, CNN and crypto Fear & Greed, Cboe
put/call) are blocked from the cloud, so the mood index is rebuilt from
Yahoo prices. See backtester/data/sentiment_indices.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data import sentiment_indices as si
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.metrics import drawdown
from backtester.ml.metalabel import event_table
from backtester.research import walk_forward
from backtester.sentiment_overlay import SentimentOverlay, bucket_returns
from backtester.strategies import Breakout, BuyAndHold, EnsembleTrend, MultiAssetTrend, Sleeves, VolTarget
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
KEY = ["cagr", "volatility", "sharpe", "max_drawdown", "annual_turnover"]
DESIGN, HOLDOUT = ("2007-06-01", "2018-12-31"), ("2019-01-01", None)


def curve_stats(eq: pd.Series) -> pd.Series:
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return pd.Series({"start": eq.index[0].date(), "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1,
                      "sharpe": r.mean() / r.std() * np.sqrt(252), "max_drawdown": drawdown(eq).min()})


def save(df: pd.DataFrame, name: str) -> None:
    print(df.to_string(), "\n")
    if OUT:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"{name}.csv")


def run_table(runs: dict, data, bt, periods: dict, bench) -> pd.DataFrame:
    rows = {}
    for name, strat in runs.items():
        for label, (a, b) in periods.items():
            rows[(name, label)] = bt.run(strat, data, start=a, end=b).stats(benchmark=bench)[KEY]
    return pd.DataFrame(rows).T


def equity_rules(base, symbols, max_gross=1.0) -> dict:
    def O(**k):
        return SentimentOverlay(base=base, symbols=symbols, max_gross=max_gross, **k)

    return {
        "base": base,
        "fear x2 (mood<=10)": O(signal_key="mood", low=10, low_mult=2.0),
        "greed x0.5 (mood>=90)": O(signal_key="mood", high=90, high_mult=0.5),
        "fear x2 + greed x0.5": O(signal_key="mood", low=10, low_mult=2.0, high=90, high_mult=0.5),
        "fear x0 (mood<=10)": O(signal_key="mood", low=10, low_mult=0.0),
        "VIX backwardation -> 0": O(signal_key="term", high=1.0, high_mult=0.0),
        "credit below 50d -> 0.5": O(signal_key="credit", low=0.0, low_mult=0.5),
        "Michigan bottom 20% -> 1.5": O(signal_key="umich", low=0.2, low_mult=1.5),
    }


# ---------------------------------------------------------------- equities
syms = sorted(set(si.MOOD_SYMBOLS + ["^VVIX", "^SKEW", "UPRO", "SHY"] + TREND_UNIVERSE + list(INVERSE.values())))
data = load_market(syms, start="2003-01-01")
# BTC trades on weekends; keep it out of the equity runs and the mood index so
# volatility and trailing windows count trading days only.
eqdata = MarketData({s: b for s, b in data.bars.items() if s != "BTC-USD"}, data.extra)
close = eqdata.field("close")
spy = close["SPY"]
mood = si.mood_index(close)
hyg_ief = close["HYG"] / close["IEF"]
data.extra.update({
    "mood": mood,
    "term": close["^VIX"] / close["^VIX3M"],
    "credit": hyg_ief / hyg_ief.rolling(50).mean() - 1,
    "umich": si.consumer_sentiment().rolling(120, min_periods=60).rank(pct=True),
})
if OUT:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([mood, si.mood_components(close)], axis=1).to_csv(OUT / "mood_index.csv")

signals = {"mood": mood, **si.mood_components(close), "low VIX": -close["^VIX"], "low VVIX": -close["^VVIX"],
           "low SKEW": -close["^SKEW"], "Michigan": si.consumer_sentiment(),
           "Michigan 12m change": si.consumer_sentiment().diff(12)}
rows = []
for period, (a, b) in {"design 2007-18": DESIGN, "holdout 2019-26": HOLDOUT}.items():
    for h in (5, 21, 63):
        for name, sig in signals.items():
            t = bucket_returns(sig, spy, h, a, b)
            rows.append({"period": period, "horizon": h, "signal (high = greed)": name,
                         "fear_q1_ret_%": t["mean_pct"].iloc[0], "greed_q5_ret_%": t["mean_pct"].iloc[-1],
                         "fear_q1_ret/vol": t["ret_per_vol"].iloc[0], "greed_q5_ret/vol": t["ret_per_vol"].iloc[-1],
                         "fear_t": t["t"].iloc[0], "greed_t": t["t"].iloc[-1]})
save(pd.DataFrame(rows), "spy_forward_returns_by_sentiment")
save(pd.concat({p: bucket_returns(mood, spy, 21, a, b, n=10) for p, (a, b) in
                {"design": DESIGN, "holdout": HOLDOUT}.items()}), "mood_deciles_21d")

rf = tbill_rate()
costs = {s: COSTS["etf"] for s in syms} | {"BTC-USD": COSTS["crypto"]}
bt = Backtester(5_000, costs=costs, cash_rate=rf)
upro = VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20)
print("UPRO vol target 20% (keeper)")
save(run_table(equity_rules(upro, ["UPRO"]), eqdata, bt,
               {"design 2009-18": ("2009-07-01", DESIGN[1]), "holdout 2019-26": HOLDOUT}, spy), "overlay_upro")
p_spy = {"design 2008-18": ("2008-01-01", DESIGN[1]), "holdout 2019-26": HOLDOUT}
print("SPY buy and hold")
save(run_table(equity_rules(BuyAndHold(symbols=["SPY"]), ["SPY"]), eqdata, bt, p_spy, spy), "overlay_spy")
print("Multi-asset trend, equity longs scaled (keeper)")
bt15 = Backtester(5_000, costs=costs, cash_rate=rf, max_leverage=1.5)
save(run_table(equity_rules(MultiAssetTrend(), ["SPY", "QQQ", "IWM", "EFA", "EEM"], max_gross=1.5),
               data, bt15, p_spy, spy), "overlay_multi_asset_trend")

print("Walk-forward mood thresholds on UPRO vol target (5y train, 1y test)")
grid = {"low": [5, 10, 20], "low_mult": [1.0, 1.5, 2.0], "high": [80, 90, 101], "high_mult": [0.5]}
wf_eq, picks = walk_forward(SentimentOverlay, grid, eqdata.slice("2009-07-01", None), bt, train_years=5,
                            test_years=1, base=upro, symbols=["UPRO"], signal_key="mood")
base_eq = bt.run(upro, eqdata, start=str(picks["test_start"].iloc[0])).equity
save(picks, "walk_forward_mood_picks")
save(pd.DataFrame({
    "walk-forward mood rule": curve_stats(wf_eq),
    "UPRO vol target alone": curve_stats(base_eq),
}), "walk_forward_mood_summary")

# ------------------------------------------------------------------ crypto
COINS = ("BTC ETH XRP LTC BCH ADA DOGE LINK XLM TRX EOS XMR ETC DASH ZEC NEO XEM WAVES QTUM OMG BAT ZRX "
         "LSK ICX VET XTZ ATOM ALGO DOT SOL AVAX AAVE FIL NEAR BNB SHIB HBAR ICP EGLD THETA MKR SNX SAND "
         "MANA AXS KSM DCR SC DGB RVN ONT ZIL ENJ CHZ IOTA").split()
cdata = load_market([f"{c}-USD" for c in COINS], start="2017-11-09")
fund7 = ((si.funding_rates("BTC") + si.funding_rates("ETH")) / 2).rolling(7).mean()
cdata.extra["fund7"] = fund7
btc = cdata.bars["BTC-USD"]["close"]
rows = []
for period, (a, b) in {"design 2021-23": ("2021-05-08", "2023-12-31"), "holdout 2024-26": ("2024-01-01", "2026-02-28")}.items():
    for h in (7, 30):
        t = bucket_returns(fund7, btc, h, a, b)
        rows.append({"period": period, "horizon": h, **{f"q{i + 1}_ret_%": t["mean_pct"].iloc[i] for i in range(5)},
                     "q1_t": t["t"].iloc[0], "q5_t": t["t"].iloc[-1]})
print("BTC forward returns by 7-day funding quintile (q1 = shorts paying, q5 = crowded longs)")
save(pd.DataFrame(rows), "btc_forward_returns_by_funding")


def crypto_rules(base, symbols) -> dict:
    def O(**k):
        return SentimentOverlay(base=base, symbols=symbols, signal_key="fund7", **k)

    return {
        "base": base,
        "crowded longs x0.5 (7d funding >= 0.1%/day)": O(high=0.001, high_mult=0.5),
        "crowded longs x0": O(high=0.001, high_mult=0.0),
        "shorts paying x1.5 (7d funding <= 0)": O(low=0.0, low_mult=1.5),
    }


cbt = Backtester(5_000, costs=COSTS["crypto"], cash_rate=rf, min_trade_value=10)
p_c = {"design 2021-05..2023": ("2021-05-08", "2023-12-31"), "holdout 2024..2026-02": ("2024-01-01", "2026-02-28")}
for name, base, symbols in [
    ("BTC+ETH breakout 20/10", Sleeves(sleeves=[(Breakout(entry=20, exit=10), ["BTC-USD", "ETH-USD"], 1.0)]), ["BTC-USD", "ETH-USD"]),
    ("Ensemble trend, 50% vol", EnsembleTrend(target_vol=0.5), None),
    ("BTC buy and hold", BuyAndHold(symbols=["BTC-USD"]), ["BTC-USD"]),
]:
    print(name)
    save(run_table(crypto_rules(base, symbols), cdata, cbt, p_c, btc), f"overlay_crypto_{name.split()[0].lower().replace('+', '_')}")

# --------------------------------------------- ETF breakout trades by mood
ETFS = ("SPY QQQ IWM EFA EEM TLT IEF GLD SLV DBC USO UUP FXE FXY XLK XLF XLE XLV XLI XLY XLP XLU XLB VNQ "
        "EWJ EWZ EWG FXI HYG LQD TIP DBA").split()
edata = load_market(sorted(set(ETFS + si.MOOD_SYMBOLS)), start="2004-01-01")
trades = event_table(edata, Breakout(entry=20, exit=10), ETFS, cost=0.0008)
trades["mood"] = si.mood_index(edata.field("close")).reindex(trades["entry"]).to_numpy()
trades = trades[(trades["entry"] < "2024-07-01") & trades["mood"].notna()]
equity_etfs = {"SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
               "VNQ", "EWJ", "EWZ", "EWG", "FXI"}
rows = []
for group, t in [("equity ETFs", trades[trades["symbol"].isin(equity_etfs)]), ("other ETFs", trades[~trades["symbol"].isin(equity_etfs)])]:
    t = t.assign(q=pd.qcut(t["mood"], 5, labels=False))
    for era, part in [("2005-14", t[t["entry"] < "2015-01-01"]), ("2015-24H1", t[t["entry"] >= "2015-01-01"]), ("all", t)]:
        g = part.groupby("q")["ret"]
        rows.append({"group": group, "era": era, **{f"q{i + 1}_ret_%": g.mean().iloc[i] * 100 for i in range(5)},
                     "trades": len(part)})
print("ETF 20/10 breakout trade return by mood quintile at entry (q1 = fear)")
save(pd.DataFrame(rows), "etf_breakout_trades_by_mood")

# ------------------------------------- FinBERT market news mood (Benzinga)
# Daily mean FinBERT score of all Benzinga headlines, 2009 - Jun 2020, from the
# FinBERT news thread. Each row is keyed by the day whose open first sees the
# text; the engine trades the open after, so the test is one day late (safe).
NEWS = Path("/mnt/project-files/finbert-sentiment/benzinga_daily_mood.parquet")
if NEWS.exists():
    news = pd.read_parquet(NEWS)
    news = news.loc[news["n"] >= 30, "z"].rolling(5, min_periods=3).mean()
    eqdata.extra["news"] = news
    p_news = {"2010-15": ("2010-01-01", "2015-12-31"), "2016-Jun 2020": ("2016-01-01", "2020-06-12")}
    for name, base, symbols in [("UPRO vol target", upro, ["UPRO"]), ("SPY", BuyAndHold(symbols=["SPY"]), ["SPY"])]:
        runs = {
            "base": base,
            "news gloom (5d z <= -1) x0.5": SentimentOverlay(base=base, symbols=symbols, signal_key="news", low=-1, low_mult=0.5),
            "news gloom x2 (contrarian)": SentimentOverlay(base=base, symbols=symbols, signal_key="news", low=-1, low_mult=2.0),
            "news cheer (5d z >= 1) x0.5": SentimentOverlay(base=base, symbols=symbols, signal_key="news", high=1, high_mult=0.5),
        }
        print(f"{name} with Benzinga news mood")
        save(run_table(runs, eqdata, bt, p_news, spy), f"overlay_news_{name.split()[0].lower()}")

# ------------------------------------------- FinBERT Reddit mood (2021-2025)
# Daily FinBERT mood per subreddit, keyed by the day whose 09:00 ET first sees
# the post; traded the open after. Each subreddit is z-scored on its own
# trailing year, then averaged: WSB + stocks + investing for stocks,
# r/CryptoCurrency for crypto.
REDDIT = Path("/mnt/project-files/finbert-sentiment/reddit_daily_mood.parquet")
if REDDIT.exists():
    rd = pd.read_parquet(REDDIT)

    def z(col: str) -> pd.Series:
        s = rd[col].dropna()
        return (s - s.rolling(252, min_periods=60).mean()) / s.rolling(252, min_periods=60).std()

    eqdata.extra["reddit"] = pd.concat([z(f"mood_{s}") for s in ("wallstreetbets", "stocks", "investing")], axis=1).mean(axis=1).rolling(5, min_periods=3).mean()
    cdata.extra["reddit"] = z("mood_cryptocurrency").rolling(7, min_periods=4).mean()

    def reddit_rules(base, symbols) -> dict:
        def O(**k):
            return SentimentOverlay(base=base, symbols=symbols, signal_key="reddit", **k)

        return {"base": base, "Reddit gloom (z <= -1) x0.5": O(low=-1, low_mult=0.5),
                "Reddit gloom x2 (contrarian)": O(low=-1, low_mult=2.0), "Reddit cheer (z >= 1) x0.5": O(high=1, high_mult=0.5)}

    p_r = {"2021-23": ("2021-04-01", "2023-12-31"), "2024-25": ("2024-01-01", "2025-12-31")}
    for name, base, symbols, d, b, bench in [
        ("UPRO vol target", upro, ["UPRO"], eqdata, bt, spy),
        ("SPY", BuyAndHold(symbols=["SPY"]), ["SPY"], eqdata, bt, spy),
        ("BTC+ETH breakout", Sleeves(sleeves=[(Breakout(entry=20, exit=10), ["BTC-USD", "ETH-USD"], 1.0)]), ["BTC-USD", "ETH-USD"], cdata, cbt, btc),
        ("Ensemble trend", EnsembleTrend(target_vol=0.5), None, cdata, cbt, btc),
    ]:
        print(f"{name} with Reddit mood")
        save(run_table(reddit_rules(base, symbols), d, b, p_r, bench), f"overlay_reddit_{name.split()[0].lower().replace('+', '_')}")
