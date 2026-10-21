"""Front-running pension funds' month-end stock/bond rebalancing.

    python examples/run_pension_rebalance.py
"""
import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.strategies import BuyAndHold, PensionRebalance, TurnOfMonth

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

data = load_market(["SPY", "TLT", "SSO", "SH"], start="2002-08-01")  # TLT lists July 2002
idx = data.index
opens, closes = data.bars["SPY"]["open"].reindex(idx), data.bars["SPY"]["close"].reindex(idx)

# The signal on its own: SPY's return over the window, by what the gap said.
print("SPY return from the window's first open to the month's last close, by signal")
for days in (2, 4, 6):
    st = PensionRebalance(days=days)
    g = st.gaps(data)["gap"]
    hi = g.expanding(36).quantile(2 / 3).shift()
    lo = g.expanding(36).quantile(1 / 3).shift()
    rows = []
    for d in g.index[hi.notna()]:
        i = idx.get_loc(d)
        if i + days < len(idx):
            side = "stocks led" if g[d] > hi[d] else "bonds led" if g[d] < lo[d] else "neither"
            rows.append((d, side, closes.iloc[i + days] / opens.iloc[i + 1] - 1))
    df = pd.DataFrame(rows, columns=["date", "signal", "r"])
    t = df.groupby("signal")["r"].agg(["mean", "std", "count"])
    t["t"] = t["mean"] / (t["std"] / np.sqrt(t["count"]))
    t["2005-15"] = df[df["date"] < "2016"].groupby("signal")["r"].mean()
    t["2016-26"] = df[df["date"] >= "2016"].groupby("signal")["r"].mean()
    print(f"\nlast {days} trading days"), print(t)

bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=tbill_rate())
spy = data.bars["SPY"]["close"]
start = "2006-01-01"  # after the 36 months the cutoffs need
runs = {
    "SPY buy and hold": BuyAndHold(symbols=["SPY"]),
    "SPY, SSO when bonds led, cash when stocks led": PensionRebalance(),
    "SPY, SSO when bonds led, SH when stocks led": PensionRebalance(when_stocks_lead="SH"),
    "SPY, SSO when bonds led only": PensionRebalance(when_stocks_lead="SPY"),
    "SPY, cash when stocks led only": PensionRebalance(when_bonds_lead="SPY"),
    "SPY, SSO in the last 4 days every month (control)": PensionRebalance(quantile=0.5, when_stocks_lead="SSO"),
    "Only the window: SSO when bonds led, else cash": PensionRebalance(base=None),
    "Turn of month (SPY last day to day 3)": TurnOfMonth(),
}
print("\nPortfolios since", start)
print(pd.DataFrame({n: bt.run(s, data, start=start).stats(benchmark=spy) for n, s in runs.items()}).T[
    ["cagr", "volatility", "sharpe", "max_drawdown", "exposure", "annual_turnover", "cost_drag_pct_per_year"]
])
