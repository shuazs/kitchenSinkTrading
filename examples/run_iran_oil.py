"""Polymarket Iran-war odds as a leading signal for crude oil.

    python examples/run_iran_oil.py

Builds the hourly Iran escalation index from Polymarket's on-chain fills
(Hugging Face snapshot, first run downloads ~5GB of columns in ~20 min),
then trades WTI futures (CL=F, hourly from yfinance) in the direction of
each jump of 5 points or more that happened while futures were trading:
enter at the next bar's open, exit 3 bars later, 6 bp round-trip cost.

Prints the base rule, a threshold x hold grid (one good cell among bad ones
would be luck), the effect of entering one or two hours late, and what
happened when the jump came while oil was closed.
"""
import pandas as pd

from backtester.data.polymarket import escalation_index, hourly_prices, iran_markets, load_fills, load_markets
from backtester.data.yahoo import load_hourly
from backtester.jump_trades import grid, jump_events, jump_trades, summarize

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.width", 200)

markets = iran_markets(load_markets())
fills = load_fills(markets["id"], since="2024-10-01", name="iran_fills")
prices, volume = hourly_prices(fills)
signs = markets.set_index(markets["id"].astype(str))["sign"]
index = escalation_index(prices, volume, signs)
print(f"{len(markets)} Iran markets, {len(fills):,} fills, index {index.index[0]:%Y-%m-%d} to {index.index[-1]:%Y-%m-%d}")

oil = load_hourly("CL=F").loc[index.index[0]:]
index = index.loc[: oil.index[-1]]

events = jump_events(index, 0.05)
live = events[events["time"].isin(oil.index)]
trades = jump_trades(oil, live, hold=3)
print("\nBase rule: jumps >= 5 pts while futures trade, enter next bar, hold 3h")
print(summarize(trades).to_string())
print("\nBy year and direction (mean net return, %):")
trades["year"] = trades["time"].dt.year
trades["side"] = trades["direction"].map({1: "escalation", -1: "de-escalation"})
print((trades.groupby("year")["ret"].agg(["count", "mean"]).assign(mean=lambda d: d["mean"] * 100)).to_string())
print((trades.groupby("side")["ret"].agg(["count", "mean"]).assign(mean=lambda d: d["mean"] * 100)).to_string())

g = grid(oil, index, [0.03, 0.04, 0.05, 0.07, 0.10], [1, 2, 3, 6, 12, 24])
print("\nMean net bp per trade, threshold (rows) x hold in hours (columns):")
print(g.pivot(index="threshold", columns="hold", values="mean_bp").round(0).to_string())
print("\nt-stats:")
print(g.pivot(index="threshold", columns="hold", values="t_stat").to_string())

print("\nEntering late (5 pts, 3h hold):")
for d in (0, 1, 2):
    s = summarize(jump_trades(oil, live, hold=3, delay=d))
    print(f"  {d}h late: {s['mean_bp']:.0f} bp, t {s['t_stat']:.2f}, hit {s['hit_rate']:.0%}")

closed = events[~events["time"].isin(oil.index)]
ct = jump_trades(oil, closed, hold=3)
print(f"\nJumps while oil was closed: {len(ct)}; reopening gap {ct['gap'].mean() * 100:.2f}% in the jump's direction, "
      f"then {ct['ret'].mean() * 100:.2f}% over the next 3h")
print("\nRecent events (the watcher should have caught these):")
print(events.tail(8).to_string(index=False))
