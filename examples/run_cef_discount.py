"""Closed-end fund discount reversion (alpha hunt #2).

Universe: EDGAR-listed fund-like names (NYSE/Nasdaq/Cboe) with a Yahoo NAV
ticker (X<ticker>X) and 500+ days of history, ~260 funds still trading.
Grid of 12 CEFDiscount variants, 40 bp spread + 10 bp slippage per fill,
alpha vs the equal-weight CEF total return; fit 2010-18, holdout 2019+,
plus a yearly walk-forward re-pick. Kill rule: holdout alpha under 3%/yr.

    SEC_USER_AGENT="name email" python examples/run_cef_discount.py
"""
import itertools
import json

import numpy as np
import pandas as pd

from backtester import Backtester, CostModel
from backtester.data.cef import load_cef_market, universe
from backtester.data.edgar import _get
from backtester.strategies import CEFDiscount

FUNDISH = r"FUND|INCOME|MUNI|TRUST|OPPORTUNIT|DIVIDEND|CREDIT|YIELD"


def main():
    j = json.loads(_get("https://www.sec.gov/files/company_tickers_exchange.json").text)
    t = pd.DataFrame(j["data"], columns=j["fields"])
    t = t[t.name.str.contains(FUNDISH, case=False) & (t.exchange != "OTC")]
    cands = [x for x in t.ticker.astype(str) if x.isalpha() and len(x) <= 4]
    data = load_cef_market(universe(cands))
    print(f"{len(data.symbols)} funds")
    ew = data.field("close").pct_change(fill_method=None).mean(axis=1)
    bt = Backtester(5000, costs={s: CostModel(spread_bps=40, slippage_bps=10) for s in data.symbols})
    rows, excess = [], {}
    for lb, ze, zx in itertools.product([126, 252], [1.5, 2.0, 2.5], [0.0, 0.5]):
        r = bt.run(CEFDiscount(lookback=lb, entry_z=ze, exit_z=zx), data, start="2006-01-01").equity.pct_change().dropna()
        ex = r - ew.reindex(r.index)
        excess[(lb, ze, zx)] = ex
        for name, lo, hi in (("fit", "2010", "2018"), ("holdout", "2019", "2026")):
            x, e = r[lo:hi], ex[lo:hi]
            rows.append(dict(lookback=lb, entry_z=ze, exit_z=zx, period=name, cagr=(1 + x).prod() ** (252 / len(x)) - 1,
                             ew_cagr=(1 + ew[lo:hi]).prod() ** (252 / len(x)) - 1, alpha=e.mean() * 252,
                             t=e.mean() / e.std() * np.sqrt(len(e)), sharpe=x.mean() / x.std() * np.sqrt(252)))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string())
    X = pd.DataFrame({str(k): v for k, v in excess.items()})
    wf = pd.concat([X.loc[str(y), X.loc["2010":str(y - 1)].mean().idxmax()] for y in range(2013, 2027)])
    h = wf["2019":]
    print(f"walk-forward 2019+: alpha {h.mean()*252:.3f}/yr, t {h.mean()/h.std()*np.sqrt(len(h)):.2f}")


if __name__ == "__main__":
    main()
