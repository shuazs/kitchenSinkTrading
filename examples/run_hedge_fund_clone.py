"""Cloning concentrated hedge funds' best ideas from 13F filings.

Needs SEC EDGAR and Yahoo access and SEC_USER_AGENT set:
    SEC_USER_AGENT="name email" python examples/run_hedge_fund_clone.py

First run downloads every SEC 13F data set since 2013 (~3 GB, reduced to
~150 MB of cache) and looks up a few thousand CUSIPs on Yahoo.
"""
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data.thirteenf import FUND_WORDS, cusip_tickers, load_holdings
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch
from backtester.strategies import BuyAndHold, HedgeFundClone, best_ideas

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

holdings = load_holdings()
print(f"13F filings: {holdings['accession'].nunique():,}, managers: {holdings['manager_cik'].nunique():,}, "
      f"{holdings['period'].min():%Y-%m} to {holdings['period'].max():%Y-%m}")

variants = {
    "top 3 ideas per manager": dict(top_k=3),
    "top idea per manager": dict(top_k=1),
    "top 5, looser filter": dict(top_k=5, max_positions=100, min_top10=0.4),
}
ideas = {k: best_ideas(holdings, **v) for k, v in variants.items()}
names = pd.concat(ideas.values()).drop_duplicates("cusip")
print(f"CUSIPs to map: {len(names)}")
tick = cusip_tickers(names["cusip"], names["issuer"])
for k in ideas:
    ideas[k] = ideas[k].assign(symbol=ideas[k]["cusip"].map(tick))
base = ideas["top 3 ideas per manager"]
print("managers kept per quarter:", base.groupby("period")["managers"].first().describe()[["min", "50%", "max"]].to_dict())
top15 = base[(base["rank"] <= 15) & ~base["issuer"].str.upper().str.contains(FUND_WORDS)]
print(f"top-15 stocks (ETFs excluded) with a current ticker: {top15['symbol'].notna().mean():.0%}; the rest were "
      "delisted, acquired or renamed (survivorship bias)")

symbols = prefetch(sorted(pd.concat(ideas.values())["symbol"].dropna().unique()) + ["SPY", "QQQ", "RSP", "GURU"], verbose=True)
bars = {s: df for s in symbols if len(df := load_bars(s, start="2013-06-01")) > 20}
data = MarketData(bars, extra={f"ideas_{i}": v for i, v in enumerate(ideas.values())})
spy = data.bars["SPY"]["close"]
bt = Backtester(5_000, costs=COSTS["large_cap"], min_trade_value=25)
start = "2013-11-15"

latest = base[base["date"] == base["date"].max()]
print("\nLatest quarter's top 15:", ", ".join(latest.dropna(subset=["symbol"])["symbol"].head(15)))

runs = {
    "SPY": BuyAndHold(symbols=["SPY"]),
    "RSP (equal-weight S&P)": BuyAndHold(symbols=["RSP"]),
    "QQQ": BuyAndHold(symbols=["QQQ"]),
    "GURU ETF": BuyAndHold(symbols=["GURU"]),
    "Clone: top 3 ideas, 15 stocks": HedgeFundClone(ideas_key="ideas_0"),
    "Clone: top 3 ideas, 10 stocks": HedgeFundClone(ideas_key="ideas_0", n_stocks=10),
    "Clone: top 3 ideas, 25 stocks": HedgeFundClone(ideas_key="ideas_0", n_stocks=25),
    "Clone: top idea, 15 stocks": HedgeFundClone(ideas_key="ideas_1"),
    "Clone: top 5 looser, 15 stocks": HedgeFundClone(ideas_key="ideas_2"),
}
print("\nSince", start)
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}))

print("\nCalendar-year returns")
yr = {}
for n in ["SPY", "RSP (equal-weight S&P)", "Clone: top 3 ideas, 15 stocks", "Clone: top idea, 15 stocks"]:
    eq = bt.run(runs[n], data, start=start).equity
    ye = pd.concat([eq.iloc[:1], eq.resample("YE").last()])
    yr[n] = ye.pct_change().iloc[1:].set_axis(ye.index[1:].year)
print(pd.DataFrame(yr))
