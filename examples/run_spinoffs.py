"""Spin-offs after index-fund dumping (alpha hunt #4).

Form 10-12B registrants 2005+, buy the child 21 trading days after its first
bar, hold 12 months, excess vs IWM; fit < 2018, holdout 2018+. Then adds the
dead children (filed periodic reports, no ticker today) at 0% (acquired),
-100% (bankrupt) and -50% (other) excess. Kill rule: holdout excess under
4%/yr or t under 2.

    SEC_USER_AGENT="name email" python examples/run_spinoffs.py
"""
import json

import pandas as pd
import yfinance as yf

from backtester.data.edgar import _get
from backtester.data.edgar_forms import filer_status, form_index
from backtester.spinoffs import child_events, dead_fate, survivorship_bound


def main():
    f = form_index(("10-12B", "10-12B/A"))
    reg = f.groupby("cik").agg(first=("date", "min")).reset_index()
    tick = json.loads(_get("https://www.sec.gov/files/company_tickers_exchange.json").text)
    tick = pd.DataFrame(tick["data"], columns=tick["fields"]).drop_duplicates("cik")
    reg = reg.merge(tick[["cik", "ticker"]], on="cik", how="left")
    alive = reg.dropna(subset=["ticker"])
    px = yf.download(alive.ticker.tolist() + ["IWM"], start="2005-01-01", auto_adjust=True, progress=False)
    ev = child_events(alive.set_index("ticker")["first"], px["Open"], px["Close"])
    status = pd.DataFrame([filer_status(c, d) for c, d in reg.loc[reg.ticker.isna(), ["cik", "first"]].itertuples(index=False)])
    dead = status[status.n_periodic >= 2].copy()
    dead["start"] = pd.to_datetime(dead.first_periodic)
    dead["fate"] = [dead_fate(a, b) for a, b in zip(dead.acquired, dead.bankrupt)]
    print(f"registrants {len(reg)}, alive with prices {ev.ticker.nunique()}, dead {len(dead)}")
    for (lag, hold), g in ev.groupby(["lag", "hold"]):
        for name, lo, hi in (("fit", "2005", "2018"), ("holdout", "2018", "2100")):
            a = g[(g.start >= lo) & (g.start < hi)].excess
            d = dead[(dead.start >= lo) & (dead.start < hi)].fate
            alive_only = survivorship_bound(a, d.iloc[:0])
            bound = survivorship_bound(a, d)
            print(f"lag {lag:2d} hold {hold} {name:7s} alive n={alive_only['n_alive']:3d} excess {alive_only['mean_excess']:+.3f} "
                  f"t {alive_only['t']:.2f} | +{bound['n_dead']} dead: {bound['mean_excess']:+.3f} t {bound['t']:.2f}")


if __name__ == "__main__":
    main()
