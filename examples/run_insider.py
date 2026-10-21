"""Insider cluster buying in small caps vs buy and hold, on real data.

Needs network access to SEC EDGAR and Yahoo Finance, and SEC_USER_AGENT set:
    SEC_USER_AGENT="name email" python examples/run_insider.py

First run downloads ~36 quarterly SEC insider data sets (~15 MB each, reduced
to a small cache), 36 XBRL share-count frames, and daily bars for ~1,400
tickers; later runs read the cache (KS_CACHE_DIR, default ~/.cache/kitchensink).

Universe: every company with a qualifying cluster purchase filed since 2018
whose market cap at the time (shares outstanding x insider trade price) was
$50M-$2B and which Yahoo still has prices for. The signal scan covers all
SEC filers, so the universe is not hand picked, but names that were delisted
since are lost (survivorship bias; counted below).
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data.insider_bulk import add_market_cap, open_market_trades
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch
from backtester.strategies import BuyAndHold
from backtester.strategies.insider import InsiderCluster, cluster_signals

START, CAPITAL = "2018-01-01", 5_000
RULE = dict(min_insiders=2, window=30, min_value=100_000, min_mcap=50e6, max_mcap=2e9)

trades = add_market_cap(open_market_trades("2017-10-01"))  # warm-up for the 30-day window
events_all = cluster_signals(trades.assign(symbol=trades["symbol"].fillna("CIK" + trades["issuer_cik"].astype(str))),
                             **RULE)
events = cluster_signals(trades, **RULE)
lost = events_all["symbol"].str.startswith("CIK").sum()
print(f"cluster events {START}..: {len(events_all)}, of which {lost} ({lost / len(events_all):.0%}) "
      f"are companies with no current SEC ticker (delisted/acquired, dropped)")

symbols = prefetch(sorted(events["symbol"].unique()), verbose=True)
bars = {}
for s in symbols + ["IWM"]:
    df = load_bars(s, start=START)
    if len(df) > 20:
        bars[s] = df
no_px = events[~events["symbol"].isin(bars)]
print(f"tickers with events: {events['symbol'].nunique()}, with Yahoo prices: {len(bars) - 1}; "
      f"events lost for missing prices: {len(no_px)}")
events = events[events["symbol"].isin(bars) & (events["filed"] >= START)]
print(f"events used: {len(events)} across {events['symbol'].nunique()} tickers")

data = MarketData(bars, extra={"insider_events": events})
universe = [s for s in data.symbols if s != "IWM"]
iwm = data.bars["IWM"]["close"]
# min_trade_value skips tiny drift trades on held names; whole slots are $500.
bt = Backtester(initial_capital=CAPITAL, costs=COSTS["small_cap"], min_trade_value=25)

# --- event study: every event, no slot limit, entry next open, vs IWM ---
opens = data.field("open")
idx = data.index
rows = []
for e in events.itertuples():
    i = idx.searchsorted(e.filed) + 1  # next trading day's open
    for h in (20, 60, 120):
        if i + h < len(idx):
            p0, p1 = opens[e.symbol].iloc[i], opens[e.symbol].iloc[i + h]
            b0, b1 = opens["IWM"].iloc[i], opens["IWM"].iloc[i + h]
            if p0 > 0 and np.isfinite(p1):
                rows.append((h, p1 / p0 - 1, p1 / p0 - b1 / b0))
es = pd.DataFrame(rows, columns=["h", "ret", "excess"])
print("\nEvent study (all events, gross of costs, vs IWM):")
print(es.groupby("h").agg(n=("ret", "size"), mean_ret=("ret", "mean"), median_ret=("ret", "median"),
                          mean_excess=("excess", "mean"), median_excess=("excess", "median"),
                          hit_rate=("excess", lambda x: (x > 0).mean())).round(4))

runs = {
    "Insider cluster (hold 60d, 10 slots)": InsiderCluster(hold_days=60, max_positions=10),
    "  variant: hold 20d": InsiderCluster(hold_days=20, max_positions=10),
    "  variant: hold 120d": InsiderCluster(hold_days=120, max_positions=10),
    "  variant: 5 slots": InsiderCluster(hold_days=60, max_positions=5),
    "  variant: 20 slots": InsiderCluster(hold_days=60, max_positions=20),
    "  variant: officer required": InsiderCluster(hold_days=60, max_positions=10, require_officer=True),
    "IWM buy and hold": BuyAndHold(symbols=["IWM"]),
}
table, results = {}, {}
for name, strat in runs.items():
    res = bt.run(strat, data, start=START)
    results[name] = (strat, res)
    table[name] = res.stats(benchmark=iwm)

# Equal-weight buy and hold of the universe, among names already trading at the
# start (the engine's default rebalance="on_change" lets weights drift).
first_close = data.field("close").loc[START:].iloc[0]
alive = [s for s in universe if np.isfinite(first_close.get(s, np.nan))]
res = Backtester(CAPITAL, costs=COSTS["small_cap"], min_trade_value=1).run(BuyAndHold(symbols=alive), data, start=START)
table[f"EW buy and hold, {len(alive)} universe names"] = res.stats(benchmark=iwm)

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)
print("\n", pd.DataFrame(table).drop(["start"]).to_string())

strat, res = results["Insider cluster (hold 60d, 10 slots)"]
print(f"\nbase run: {len(strat.taken)} positions opened from {len(events)} eligible events; "
      f"{len(res.trades)} fills; costs ${res.costs.sum():,.0f}")
yearly = res.equity.resample("YE").last().pct_change()
yearly.iloc[0] = res.equity.resample("YE").last().iloc[0] / CAPITAL - 1
iwm_y = iwm.loc[START:].resample("YE").last()
iwm_yr = iwm_y.pct_change()
iwm_yr.iloc[0] = iwm_y.iloc[0] / iwm.loc[START:].iloc[0] - 1
print(pd.DataFrame({"strategy": yearly, "IWM": iwm_yr}).set_index(yearly.index.year).to_string())
