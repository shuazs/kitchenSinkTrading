"""Cut leverage after a burst of Trump posts on tariffs or markets.

    python examples/run_trump_overlay.py

Posts: first-term tweets (2017-2021) and Truth Social (2022 on). Results are
only meaningful while the archive has posts, so runs end at the archive's
last post and the Jan 2021 - Feb 2022 gap counts as "no burst".
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.trump_posts import MARKET, TARIFF, TRADE, covered, daily_counts, load_posts
from backtester.data.yahoo import load_market
from backtester.research import sweep
from backtester.strategies import BuyAndHold, PostRiskOverlay, VolTarget

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

data = load_market(["SPY", "UPRO", "QQQ", "TQQQ", "SHY"], start="2009-01-01")
posts = load_posts()
idx = data.index
topics = {"tariff": TARIFF, "trade": TRADE, "market": MARKET}
for name, pat in topics.items():
    data.extra[name] = daily_counts(posts, idx, pat)
cov = covered(posts, idx)
start = "2017-01-20"
end = posts["time"].max().tz_convert(None).normalize()
print(f"posts {len(posts):,}; testing {start} to {end.date()}; sessions with post data {cov[start:end].sum()}")

# 1. Does a burst predict the next week's volatility? (Not returns, just risk.)
spy = data.bars["SPY"]["close"]
r = spy.pct_change()
next_vol = r[::-1].rolling(5).std()[::-1].shift(-1) * np.sqrt(252)
next_ret = spy.shift(-5) / spy - 1
rows = []
for name in topics:
    for k in (1, 3, 5):
        on = PostRiskOverlay(counts_key=name, window=5, min_posts=k, hold=1).triggered(data)
        m = cov & (idx >= start) & (idx <= end)
        rows.append({
            "topic": name, "min_posts_5d": k, "share_of_days": on[m].mean(),
            "next5d_vol_burst": next_vol[m & on].mean(), "next5d_vol_other": next_vol[m & ~on].mean(),
            "next5d_ret_burst": next_ret[m & on].mean(), "next5d_ret_other": next_ret[m & ~on].mean(),
        })
print("\nSPY over the next 5 sessions, after a burst vs other covered days")
print(pd.DataFrame(rows))

# 2. The overlay on leveraged strategies.
bt = Backtester(initial_capital=5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
bases = {
    "UPRO vol target 20%": VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20),
    "UPRO buy and hold": BuyAndHold(symbols=["UPRO"]),
    "TQQQ buy and hold": BuyAndHold(symbols=["TQQQ"]),
}
table = {"SPY buy and hold": bt.run(BuyAndHold(symbols=["SPY"]), data, start, end).stats(benchmark=spy)}
for name, base in bases.items():
    table[name] = bt.run(base, data, start, end).stats(benchmark=spy)
    table[name + " + tariff overlay"] = bt.run(
        PostRiskOverlay(base=base, counts_key="tariff", min_posts=3, scale=0.5, risk_off="SHY"), data, start, end
    ).stats(benchmark=spy)
print("\n", pd.DataFrame(table).T[["cagr", "volatility", "sharpe", "max_drawdown", "calmar", "annual_turnover"]])

grid = sweep(
    PostRiskOverlay,
    {"counts_key": list(topics), "min_posts": [1, 3, 5], "scale": [0.0, 0.5]},
    data, bt, start=start, end=end, base=bases["UPRO vol target 20%"], risk_off="SHY", window=5, hold=5,
)
print(f"\nsweep on UPRO vol target 20% ({grid.attrs['n_variants']} variants)")
print(grid[["counts_key", "min_posts", "scale", "cagr", "sharpe", "max_drawdown", "annual_turnover"]])
