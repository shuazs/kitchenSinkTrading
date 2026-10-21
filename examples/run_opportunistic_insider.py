"""Opportunistic vs routine insider purchases (Cohen, Malloy and Pomorski
2012), as an event study and as a filter on the insider strategies.

    SEC_USER_AGENT="name email" python examples/run_opportunistic_insider.py

Insiders are classified from their purchases AND sales in the three prior
calendar years, so trades load from 2013 and signals start in 2016. Small
and mid caps only ($50M-$2B at the time). Same survivorship caveat as
run_insider.py: companies without a current SEC ticker or Yahoo prices are
dropped, and counted below.
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester, MarketData
from backtester.data.insider_bulk import add_market_cap, open_market_trades
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import prefetch
from backtester.strategies import BuyAndHold
from backtester.strategies.insider import InsiderCluster, classify_insiders, cluster_signals

START, CAPITAL = "2016-01-01", 5_000
BAND = dict(min_mcap=50e6, max_mcap=2e9)

trades = classify_insiders(open_market_trades("2013-01-01", codes=("P", "S")))
buys = add_market_cap(trades[(trades["code"] == "P") & (trades["filed"] >= "2015-10-01")])
print("purchases since 2016 by trader type:")
print(buys[buys["filed"] >= START]["trader_type"].value_counts().to_string())

# One event per (company, filing date, trader type): solo buys are allowed here.
solo = {kind: cluster_signals(buys, min_insiders=1, min_value=25_000, trader_types=(kind,), **BAND)
        for kind in ("routine", "opportunistic", "unclassified")}
clusters = {
    "all": cluster_signals(buys, min_insiders=2, min_value=100_000, **BAND),
    "no routine": cluster_signals(buys, min_insiders=2, min_value=100_000,
                                  trader_types=("opportunistic", "unclassified"), **BAND),
}
every = pd.concat([*solo.values(), *clusters.values()])
symbols = prefetch(sorted(every["symbol"].unique()), verbose=False)
bars = {}
for s in symbols + ["IWM"]:
    df = load_bars(s, start=START)
    if len(df) > 20:
        bars[s] = df
print(f"\ntickers with events: {every['symbol'].nunique()}, with Yahoo prices: {len(bars) - 1}")

data = MarketData(bars)
opens, idx = data.field("open"), data.index


def event_study(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["symbol"].isin(bars) & (ev["filed"] >= START)]
    rows = []
    for e in ev.itertuples():
        i = idx.searchsorted(e.filed) + 1
        for h in (20, 60, 120):
            if i + h < len(idx):
                p0, p1 = opens[e.symbol].iloc[i], opens[e.symbol].iloc[i + h]
                b0, b1 = opens["IWM"].iloc[i], opens["IWM"].iloc[i + h]
                if p0 > 0 and np.isfinite(p1):
                    rows.append((h, p1 / p0 - b1 / b0))
    es = pd.DataFrame(rows, columns=["h", "excess"])
    return es.groupby("h")["excess"].agg(n="size", mean="mean", median="median", hit=lambda x: (x > 0).mean())


pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
print("\n== Event study: excess return vs IWM from the next open, gross of costs")
for name, ev in {**{f"solo {k}": v for k, v in solo.items()}, **{f"cluster {k}": v for k, v in clusters.items()}}.items():
    print(f"\n{name}")
    print(event_study(ev).to_string())

iwm = data.bars["IWM"]["close"]
bt = Backtester(initial_capital=CAPITAL, costs=COSTS["small_cap"], min_trade_value=25)
runs = {
    "Cluster, all insiders": (InsiderCluster(hold_days=60, max_positions=10), clusters["all"]),
    "Cluster, no routine": (InsiderCluster(hold_days=60, max_positions=10), clusters["no routine"]),
    "Solo opportunistic, 60d": (InsiderCluster(min_insiders=1, min_value=25_000, hold_days=60, max_positions=10), solo["opportunistic"]),
    "Solo opportunistic, 120d": (InsiderCluster(min_insiders=1, min_value=25_000, hold_days=120, max_positions=10), solo["opportunistic"]),
    "Solo opportunistic, $100k+": (InsiderCluster(min_insiders=1, min_value=100_000, hold_days=60, max_positions=10), solo["opportunistic"]),
    "Solo routine, 60d (control)": (InsiderCluster(min_insiders=1, min_value=25_000, hold_days=60, max_positions=10), solo["routine"]),
}
table = {}
for name, (strat, ev) in runs.items():
    d = MarketData(bars, extra={"insider_events": ev[ev["filed"] >= START]})
    table[name] = bt.run(strat, d, start=START).stats(benchmark=iwm)
table["IWM buy and hold"] = bt.run(BuyAndHold(symbols=["IWM"]), data, start=START).stats(benchmark=iwm)
print("\n", pd.DataFrame(table).drop(["start"]).to_string())
