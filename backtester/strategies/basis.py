"""Crypto cash-and-carry: long spot, short the front-month future, roll monthly.

A hedged position earns the futures premium (the basis) as the future
converges to spot at expiry, whatever the coin does. Desks run this at size
on CME and offshore perpetuals; at $5,000 the pieces are a spot ETF or coins
plus Coinbase's nano futures (0.01 BTC) or CME micros (0.1 BTC).

There is no free history of contract-by-contract futures prices, so the
basis comes from Yahoo's continuous front-month series (BTC=F, ETH=F)
against spot (BTC-USD, ETH-USD). The two settle hours apart (CME around
20:00 UTC, Yahoo spot at midnight UTC), so each day's basis carries a few
percent of noise (worst before 2021), so the simulator uses medians over
`smooth` days.
It assumes the next contract's annualised basis equals the current front
month's, and that the locked-in basis accrues evenly over the month. That
hides mark-to-market swings and margin calls on the short future while the
coin rallies, which are real and need spare cash at the futures broker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Result
from ..data.yahoo import load_bars


def last_fridays(start, end) -> pd.DatetimeIndex:
    """CME bitcoin and ether futures expire on the last Friday of each month."""
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(1), freq="M")
    out = []
    for m in months:
        d = m.end_time.normalize()
        out.append(d - pd.Timedelta(days=(d.weekday() - 4) % 7))
    return pd.DatetimeIndex(out)


def load_basis_inputs(coin: str = "BTC", start="2017-12-01", end=None) -> pd.DataFrame:
    """Daily futures and spot closes, raw basis, days to expiry and annualised basis."""
    fut = load_bars(f"{coin}=F", start, end)["close"]
    spot = load_bars(f"{coin}-USD", start, end)["close"]
    df = pd.concat({"future": fut, "spot": spot}, axis=1, sort=True).dropna()
    expiries = last_fridays(df.index[0], df.index[-1])
    nxt = expiries[np.searchsorted(expiries.values, df.index.values, side="left")]
    df["expiry"] = nxt
    df["dte"] = (df["expiry"] - df.index).dt.days
    df["basis"] = df["future"] / df["spot"] - 1
    # Close to expiry the basis is tiny and dividing by a few days magnifies the noise.
    df["ann_basis"] = (df["basis"] * 365 / df["dte"]).where(df["dte"] >= 7)
    return df


class BasisCarry:
    """Monthly cash-and-carry with an optional hurdle over T-bills.

    margin: fraction of the hedged notional kept as cash at the futures
        broker, so `1 / (1 + margin)` of equity buys spot.
    hurdle: hold the trade only when the annualised basis beats the T-bill
        rate by this much; otherwise the whole account sits in T-bills.
        None holds it every month.
    smooth: days of annualised basis to take the median of, both for the
        decision before a roll and for the basis earned after it.
    roll_cost_bps: cost of rolling the short future each month, per side
        of notional (exchange fee, spread, slippage).
    spot_cost_bps: cost of buying or selling the spot leg, charged only
        when the trade is switched on or off.
    spot_fee: annual fee on the spot leg (0.0025 for a bitcoin ETF).
    margin_yield: fraction of the T-bill rate earned on margin cash.
    """

    name = "basis_carry"
    params = {
        "margin": 0.35,
        "hurdle": None,
        "smooth": 10,
        "roll_cost_bps": 3.0,
        "spot_cost_bps": 10.0,
        "spot_fee": 0.0025,
        "margin_yield": 0.0,
    }

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"BasisCarry got unknown params {sorted(unknown)}")
        self.p = {**self.params, **params}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.p.items())
        return f"BasisCarry({args})"

    def monthly(self, inputs: pd.DataFrame, cash_rate: pd.Series) -> pd.DataFrame:
        """One row per contract month: the basis locked in and whether it was held."""
        est = inputs["ann_basis"].rolling(self.p["smooth"], min_periods=3).median()
        rows = []
        expiries = inputs["expiry"].drop_duplicates().tolist()
        for prev, exp in zip(expiries[:-1], expiries[1:]):
            # Decide on the last close before the old contract expires.
            known = est.loc[: prev - pd.Timedelta(days=1)].dropna()
            if known.empty:
                continue
            # What the trade earns is the new contract's basis, measured on the days
            # after the roll. Booking the estimate used to decide would let the noise
            # that triggered an entry also pay for it.
            after = inputs["ann_basis"].loc[prev + pd.Timedelta(days=1): exp].dropna().iloc[: self.p["smooth"]]
            if after.empty:
                continue
            rf = float(cash_rate.reindex(inputs.index).ffill().loc[:prev].iloc[-1])
            signal = float(known.iloc[-1])
            hold = self.p["hurdle"] is None or signal > rf + self.p["hurdle"]
            rows.append({"start": prev, "end": exp, "signal": signal, "ann_basis": float(after.median()),
                         "tbill": rf, "hold": hold})
        return pd.DataFrame(rows)

    def run(self, inputs: pd.DataFrame, cash_rate: float | pd.Series = 0.0, initial_capital: float = 5_000.0,
            start=None, end=None) -> Result:
        if not isinstance(cash_rate, pd.Series):
            cash_rate = pd.Series(cash_rate, index=inputs.index)
        months = self.monthly(inputs, cash_rate)
        if start is not None:
            months = months[months["start"] >= pd.Timestamp(start)]
        if end is not None:
            months = months[months["end"] <= pd.Timestamp(end)]
        p = self.p
        hedged = 1 / (1 + p["margin"])
        last = min(months["end"].iloc[-1], inputs.index[-1])
        days = pd.date_range(months["start"].iloc[0], last, freq="D")
        daily = pd.Series(0.0, index=days)
        cost = pd.Series(0.0, index=days)
        weight = pd.Series(0.0, index=days)
        was_on = False
        for m in months.itertuples():
            span = daily.loc[m.start + pd.Timedelta(days=1): m.end].index
            if span.empty:
                continue
            if m.hold:
                carry = hedged * (m.ann_basis - p["spot_fee"]) + (1 - hedged) * p["margin_yield"] * m.tbill
                daily.loc[span] = carry / 365
                fee = hedged * 2 * p["roll_cost_bps"] / 1e4
                if not was_on:
                    fee += hedged * p["spot_cost_bps"] / 1e4
                weight.loc[span] = hedged
            else:
                daily.loc[span] = (1 + m.tbill) ** (1 / 365) - 1
                fee = hedged * p["spot_cost_bps"] / 1e4 if was_on else 0.0
            cost.loc[span[0]] = fee
            daily.loc[span[0]] -= fee
            was_on = m.hold

        equity = initial_capital * (1 + daily).cumprod()
        dollars = cost * equity.shift(1).fillna(initial_capital)
        return Result(
            equity=equity.rename("equity"),
            positions=weight.to_frame("hedged_weight"),
            weights=weight.to_frame("hedged_weight"),
            trades=pd.DataFrame(columns=["date", "symbol", "qty", "ref_price", "fill_price", "fees", "slippage"]),
            costs=dollars.rename("costs"),
            initial_capital=initial_capital,
            strategy=repr(self),
        )
