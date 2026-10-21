"""Wheel (cash-secured puts, then covered calls) vs buy and hold, $5,000 account.

A $5k account cannot secure one SPY put (~$77k of strike), so the wheel runs on
liquid ETFs whose strikes have mostly fit under $5,000: XLF (default), XLU,
EWZ and SLV. When a strike no longer fits, the account sits in T-bills.

First it checks the synthetic pricer: a cost-free replica of CBOE's PUT index
(monthly ATM SPX puts on T-bill collateral) against the real ^PUT history.

Needs network access to Yahoo Finance and FRED:
    python examples/run_wheel.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

from backtester import COSTS, Backtester
from backtester.data.fred import load_series, tbill_rate
from backtester.data.yahoo import load_market
from backtester.strategies import BuyAndHold
from backtester.strategies.wheel import Wheel, bs_price, load_wheel_inputs, monthly_expiries

CAPITAL = 5_000
rf = tbill_rate()
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 10)


def put_index_check(iv_mults=(0.85, 0.88, 1.0), q=0.018):
    """CAGR of a synthetic PUT-index replica at several IV = mult x VIX."""
    spx = yf.download("^GSPC", start="1996-01-01", progress=False)["Close"].squeeze()
    put = yf.download("^PUT", start="1996-01-01", progress=False)["Close"].squeeze()
    idx = spx.index[spx.index >= "1996-09-01"]
    vix = load_series("VIXCLS").reindex(idx).ffill() / 100
    r = rf.reindex(idx).ffill()
    exps = monthly_expiries(idx)
    years = (idx[-1] - idx[0]).days / 365.25
    cagr = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    out = {"^PUT actual": cagr(put.reindex(idx).ffill()), "S&P 500 price": cagr(spx.reindex(idx))}
    for mult in iv_mults:
        cash, K, eq = 1.0, None, []
        for i, d in enumerate(idx):
            S = spx[d]
            if i:
                cash *= (1 + r.iloc[i - 1]) ** ((d - idx[i - 1]).days / 365)
            if K is not None and d >= exp:
                cash -= max(K - S, 0) * n
                K = None
            if K is None:
                exp = exps[exps.searchsorted(d + pd.Timedelta(days=20))]
                K, n = np.floor(S / 5) * 5, cash / (np.floor(S / 5) * 5)
                cash += bs_price(S, K, (exp - d).days / 365, r.iloc[i], q, vix.iloc[i] * mult, False) * n
            eq.append(cash - bs_price(S, K, (exp - d).days / 365, r.iloc[i], q, vix.iloc[i] * mult, False) * n)
        out[f"replica, ATM IV = {mult} x VIX"] = cagr(pd.Series(eq, index=idx))
    return pd.Series(out, name="CAGR")


print("Pricer check vs CBOE PUT index (no costs), 1996-09 onward:")
print(put_index_check().map("{:.2%}".format).to_string())

bt = Backtester(initial_capital=CAPITAL, costs=COSTS["etf"], cash_rate=rf)
for sym in ["XLF", "XLU", "EWZ", "SLV"]:
    inputs = load_wheel_inputs(sym)
    start = inputs.dropna(subset=["iv"]).index[0]
    runs = {
        "wheel": Wheel(symbol=sym).run(inputs, CAPITAL),
        "wheel, no skew credit": Wheel(symbol=sym, put_iv_mult=0.88, call_iv_mult=0.75).run(inputs, CAPITAL),
        "wheel, 30d put": Wheel(symbol=sym, put_delta=0.30).run(inputs, CAPITAL),
        "wheel, calls >= basis": Wheel(symbol=sym, call_above_basis=True).run(inputs, CAPITAL),
        "buy and hold": bt.run(BuyAndHold(symbols=[sym]), load_market([sym], start=start)),
    }
    table = pd.DataFrame({k: r.stats(benchmark=inputs["adj_close"], rf=rf) for k, r in runs.items()})
    w = runs["wheel"]
    idle = (w.positions.abs().sum(axis=1) == 0).mean()
    assigned = (w.trades["symbol"] == sym).sum()
    print(f"\n=== {sym}: default wheel idle in cash {idle:.1%} of days, {assigned} assignments + call-aways ===")
    print(table.drop(index=["bench_cagr", "bench_max_drawdown", "bench_sharpe", "exposure"]))
    yearly = pd.DataFrame({k: r.equity.resample("YE").last().pct_change() for k, r in runs.items()})
    yearly.index = yearly.index.year
    print("worst buy-and-hold calendar years:")
    print(yearly.sort_values("buy and hold").head(3))
