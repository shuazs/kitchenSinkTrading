"""Fit the spread pricer's skew to today's SPY option chain.

Backs out Black-Scholes IV for out-of-the-money SPY options 12 to 75 days
out, expresses each as a ratio to VIX against standardized moneyness
z = ln(K/S) / (VIX * sqrt(T)), and fits IV/VIX = atm * (1 - slope*z + curve*z^2)
with separate curvature below and above the money. The defaults in
CreditSpreads came from the 2026-10-20 close: atm 0.79, slope 0.344, put curve
0.016, call curve 0.164.

Run it during market hours, when bids and asks are live; outside them it
falls back to last trade prices from the final hour of the previous session.
Re-running on different days shows how much the skew moves, which is what
`skew_scale` is for.

    python examples/fit_spy_skew.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

from backtester.strategies.wheel import bs_price


def implied_vol(price, S, K, T, r, q, call):
    lo, hi = 0.01, 3.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, q, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def chain_ivs(r=0.04, q=0.012):
    spy = yf.Ticker("SPY")
    hist = spy.history(period="5d")
    S = float(hist["Close"].iloc[-1])
    last_session = hist.index[-1].tz_convert("America/New_York").normalize()
    close_time = last_session + pd.Timedelta(hours=16)
    vix = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]) / 100
    rows = []
    for exp in spy.options:
        T = ((pd.Timestamp(exp).tz_localize("America/New_York") + pd.Timedelta(hours=16)) - close_time).total_seconds() / 86400 / 365
        if not 12 / 365 <= T <= 75 / 365:
            continue
        chain = spy.option_chain(exp)
        for call, df in ((True, chain.calls), (False, chain.puts)):
            live = (df["bid"] > 0) & (df["ask"] > 0)
            if live.mean() > 0.5:
                df = df[live].assign(px=(df["bid"] + df["ask"]) / 2)
            else:  # market closed: last trades from the final hour
                df = df[df["lastTradeDate"] >= close_time - pd.Timedelta(hours=1)].assign(px=df["lastPrice"])
            df = df[(df["px"] >= 0.05) & ((df["strike"] > S) if call else (df["strike"] < S))]
            for K, px in zip(df["strike"], df["px"]):
                rows.append({"expiry": exp, "T": T, "K": K, "call": call, "px": px,
                             "iv": implied_vol(px, S, K, T, r, q, call)})
    out = pd.DataFrame(rows)
    out["z"] = np.log(out["K"] / S) / (vix * np.sqrt(out["T"]))
    out["ratio"] = out["iv"] / vix
    return out[(out["z"] > -4) & (out["z"] < 2.2)], S, vix


def fit(df):
    z, y = df["z"].to_numpy(), df["ratio"].to_numpy()
    X = np.c_[np.ones_like(z), -z, np.where(z < 0, z**2, 0), np.where(z >= 0, z**2, 0)]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rmse = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    atm = coef[0]
    return {"atm_over_vix": atm, "skew_slope": coef[1] / atm, "put_curve": coef[2] / atm,
            "call_curve": coef[3] / atm, "rmse": rmse, "n": len(df)}


if __name__ == "__main__":
    df, S, vix = chain_ivs()
    print(f"SPY {S:.2f}, VIX {vix:.2%}, {len(df)} options")
    bins = pd.cut(df["z"], np.arange(-4, 2.5, 0.5))
    print(df.groupby(bins, observed=True)["ratio"].agg(["size", "median"]).round(3).to_string())
    print(pd.Series(fit(df)).round(3).to_string())
