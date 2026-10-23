"""Copy Congress members' disclosed stock purchases (STOCK Act PTRs, 2014-2026).

Events are keyed on the filing date (the day the trade became public), bought
at the open two days later and held 63 trading days, 10 slots, $5,000,
large-cap costs. Subsets from the literature: all purchases, big ones
(>= $50k), cluster buys (3+ members in 60 days), by party, party leaders
(mostly Nancy Pelosi), fast filers (<= 10 days). Periods: design 2014-2019,
validation 2020 - Jun 2024, holdout Jul 2024+. Also compares the NANC and
KRUZ Congress-tracking ETFs with SPY.

    python examples/run_congress_trades.py [out_dir]
"""
import os
import sys

import pandas as pd

from backtester import COSTS, Backtester
from backtester.data import congress
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.data.yahoo_batch import prefetch
from backtester.research import event_study, summarize_events
from backtester.strategies import BuyAndHold
from backtester.strategies.events import EventHold

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

PERIODS = {"design 2014-19": ("2014-02-01", "2019-12-31"), "valid 2020-24H1": ("2020-01-01", "2024-06-30"), "holdout 2024H2+": ("2024-07-01", None)}
out_dir = sys.argv[1] if len(sys.argv) > 1 else None

trades = congress.load_trades()
events = congress.buy_events(trades)
sells = (trades[trades["side"] == -1].assign(date=lambda x: x["filing_date"] + pd.Timedelta(days=1))
         .groupby(["ticker", "date"]).size().rename("n").reset_index().rename(columns={"ticker": "symbol"}))
syms = prefetch(sorted(set(events["symbol"]) | set(sells["symbol"]) | {"SPY"}))
data = load_market(syms, start="2013-06-01")
print(f"{len(trades):,} stock trades, {len(events):,} buy events; prices for {len(syms):,} of "
      f"{len(set(events['symbol']) | set(sells['symbol'])):,} tickers (missing ones are mostly delisted)")

subsets = {
    "all buys": events,
    "big (>= $50k)": events[events["amount_low"] >= 50_001],
    "cluster (3+ members/60d)": events[events["members_60d"] >= 3],
    "Republican": events[(events["n_members"] == 1) & (events["party"] == "Republican")],
    "Democrat": events[(events["n_members"] == 1) & (events["party"] == "Democrat")],
    "leaders": events[events["leader"]],
    "fast filers (<= 10d)": events[events["lag_days"] <= 10],
}

print("\nEvent study: return minus SPY from the open after the public day")
rows = []
study = event_study(data, events, horizons=(5, 21, 63, 126), benchmark="SPY").merge(events, on=["date", "symbol"])
for name, ev in subsets.items():
    s = study.merge(ev[["date", "symbol"]], on=["date", "symbol"])
    for per, (a, b) in {**PERIODS, "all": ("2014", None)}.items():
        part = s[(s["date"] >= a) & (s["date"] <= (b or "2100"))]
        summ = summarize_events(part)
        rows.append({"subset": name, "period": per, "n": int(summ.loc["x63", "n"]),
                     **{f"{h} mean": summ.loc[h, "mean"] for h in ("x21", "x63", "x126")},
                     "x63 t": summ.loc["x63", "t"]})
sell_study = summarize_events(event_study(data, sells, horizons=(21, 63, 126), benchmark="SPY"))
rows.append({"subset": "sales (all)", "period": "all", "n": int(sell_study.loc["x63", "n"]),
             **{f"{h} mean": sell_study.loc[h, "mean"] for h in ("x21", "x63", "x126")}, "x63 t": sell_study.loc["x63", "t"]})
es = pd.DataFrame(rows)
print(es.to_string(index=False))
print("t-stats treat events as independent; overlapping windows overstate them.")

print("\nCopy portfolios: 10 slots, hold 63 days, $5,000, large-cap costs (Sharpe over T-bills)")
cash = tbill_rate()
bt = Backtester(5_000, costs=COSTS["large_cap"], cash_rate=cash)
spy = data.bars["SPY"]["close"]
rows = []
for name, ev in {**subsets, "SPY": None}.items():
    if ev is None:
        strat = BuyAndHold(symbols=["SPY"])
    else:
        data.extra["ev"] = ev.assign(priority=ev["amount_low"])
        strat = EventHold(events_key="ev", hold_days=63, max_positions=10)
    for per, (a, b) in PERIODS.items():
        st = bt.run(strat, data, a, b).stats(benchmark=spy, rf=cash)
        rows.append({"portfolio": name, "period": per, "cagr": st["cagr"], "sharpe": st["sharpe"],
                     "max_dd": st["max_drawdown"], "excess_vs_spy": st["cagr"] - st["bench_cagr"],
                     "beta": st["beta"], "cost_drag": st["cost_drag_pct_per_year"]})
pf = pd.DataFrame(rows)
print(pf.to_string(index=False))

if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    es.to_csv(f"{out_dir}/event_study.csv", index=False)
    pf.to_csv(f"{out_dir}/portfolios.csv", index=False)

print("\nCongress-tracking ETFs since launch (Feb 2023; skips any Yahoo no longer lists):")
etfs = load_market(prefetch(["NANC", "KRUZ", "SPY"]), start="2023-02-07")
for sym, bars in etfs.bars.items():
    c = bars["close"].dropna()
    if len(c) < 2:
        continue
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    print(f"{sym}: {c.index[0].date()}..{c.index[-1].date()} CAGR {(c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1:.3f}")
