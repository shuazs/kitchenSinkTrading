"""The wheel: sell cash-secured puts, and if assigned sell covered calls.

Option selling earns the variance risk premium: implied vol has on average
exceeded realized vol (Carr and Wu 2009, "Variance Risk Premiums"; Bondarenko
2014 on put prices), and CBOE's PUT index of monthly cash-secured SPX puts has
roughly matched the S&P 500 with lower volatility since 1986 (Whaley 2002;
Israelov and Nielsen 2015, "Covered Calls Uncovered", note the gain is mostly
short-vol beta, paid back in crashes).

There is no free option history, so options are priced with Black-Scholes.
Implied vol is a proxy: VIX (FRED VIXCLS) scaled by the underlying's realized
vol relative to SPY's. That carries over the IV-over-RV premium that VIX
already embeds and adds no edge of its own. Options are American and settle
on the real (split-adjusted, not dividend-adjusted) price, so this module uses
raw closes, credits dividends in cash and sizes contracts at 100 shares.

The engine's Strategy interface is target weights, which cannot express short
options, so `Wheel.run` is a standalone daily simulator that returns the same
`Result` type and works with `Result.stats()`.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..core import Result

_N = NormalDist()


def bs_price(S, K, T, r, q, sigma, call: bool) -> float:
    """Black-Scholes-Merton price with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / v
    d2 = d1 - v
    if call:
        return S * math.exp(-q * T) * _N.cdf(d1) - K * math.exp(-r * T) * _N.cdf(d2)
    return K * math.exp(-r * T) * _N.cdf(-d2) - S * math.exp(-q * T) * _N.cdf(-d1)


def bs_delta(S, K, T, r, q, sigma, call: bool) -> float:
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / v
    return math.exp(-q * T) * (_N.cdf(d1) if call else _N.cdf(d1) - 1)


def strike_for_delta(S, T, r, q, sigma, delta: float, call: bool, step: float) -> float:
    """Listed strike (multiple of `step`) whose |delta| is closest to `delta`."""
    z = _N.inv_cdf(delta * math.exp(q * T)) if call else -_N.inv_cdf(delta * math.exp(q * T))
    exact = S * math.exp((r - q + 0.5 * sigma**2) * T - z * sigma * math.sqrt(T))
    lo = math.floor(exact / step) * step
    cands = [k for k in (lo, lo + step) if k > 0]
    return min(cands, key=lambda k: abs(abs(bs_delta(S, k, T, r, q, sigma, call)) - delta))


def strike_step(S: float) -> float:
    """Typical listed strike spacing for ETF monthlies."""
    return 0.5 if S < 25 else 1.0 if S < 200 else 5.0


def monthly_expiries(index: pd.DatetimeIndex, months_ahead: int = 3) -> pd.DatetimeIndex:
    """Third Fridays, moved back to the last trading day when that is a holiday."""
    start = index[0].to_period("M").to_timestamp()
    end = index[-1] + pd.DateOffset(months=months_ahead)
    third_fri = pd.date_range(start, end, freq="WOM-3FRI")
    pos = index.searchsorted(third_fri, side="right") - 1
    out = [index[p] if 0 <= p and t <= index[-1] else t for t, p in zip(third_fri, pos)]
    return pd.DatetimeIndex(sorted(set(out)))


def implied_vol_proxy(close: pd.Series, spy_close: pd.Series, vix: pd.Series, window: int = 63,
                      clip=(0.5, 4.0)) -> pd.Series:
    """VIX scaled by trailing realized vol of the underlying over SPY's.

    For SPY itself this is just VIX. Uses only data up to each day.
    """
    rv = lambda s: np.log(s).diff().rolling(window).std()
    idx = close.index
    ratio = (rv(close) / rv(spy_close.reindex(idx).ffill())).clip(*clip)
    return (vix.reindex(idx).ffill() / 100 * ratio).rename("iv")


def load_wheel_inputs(symbol: str, start="1990-01-01", end=None, window: int = 63, refresh=False) -> pd.DataFrame:
    """Daily close (raw, split-adjusted), adj_close, dividend, iv and rate for `symbol`.

    Needs Yahoo Finance and FRED (VIXCLS, DTB3).
    """
    import yfinance as yf

    from ..data.cache import cache_dir
    from ..data.fred import load_series, tbill_rate
    from ..data.yahoo import load_bars

    path = cache_dir("yahoo") / f"{symbol}_1d_raw.csv"
    if path.exists() and not refresh:
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raw = yf.download(symbol, start="1990-01-01", auto_adjust=False, actions=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.to_csv(path)
    df = pd.DataFrame({
        "close": raw["Close"],
        "adj_close": raw["Adj Close"],
        "dividend": raw.get("Dividends", 0.0),
    }).dropna(subset=["close"])
    spy = load_bars("SPY", start="1990-01-01")["close"]
    df["iv"] = implied_vol_proxy(df["adj_close"], spy, load_series("VIXCLS"), window)
    df["rate"] = tbill_rate().reindex(df.index).ffill()
    return df.loc[start:end].fillna({"dividend": 0.0})


class Wheel:
    """Cash-secured put writing that turns into covered calls after assignment.

    Each cycle sells the monthly (third Friday) expiry at least `min_dte` days
    out, at the listed strike nearest `put_delta`, using as many contracts as
    idle cash fully secures. Options are held to expiry and settle on the
    close: an ITM put buys 100 shares per contract at the strike, an ITM call
    delivers them. While shares are held, calls at `call_delta` are sold
    against every full 100 shares (strike floored at the assignment price if
    `call_above_basis`). Cash, including collateral, earns the T-bill rate.

    put_iv_mult / call_iv_mult map the VIX-based proxy to the strike's IV.
    VIX is a 30-day variance swap rate that includes the put skew, so it sits
    above at-the-money IV: replicating CBOE's PUT index (monthly ATM SPX puts)
    with this pricer matches its 1996-2026 CAGR at about 0.88 x VIX, while
    1.0 x VIX overstates it by ~3%/yr (examples/run_wheel.py checks this).
    25-delta puts trade a few vol points over ATM (0.95), OTM calls a few
    under (0.80). Raising these is how a backtest invents edge.
    Costs: `commission` per contract on every sale (no fee on expiry or
    assignment unless `assignment_fee`), and half of the bid-ask spread, the
    larger of `spread_pct` of mid and `min_spread`, on every sale. A sale whose
    bid is below `min_premium` is skipped and retried the next day.
    """

    name = "wheel"
    params = {
        "symbol": "XLF",
        "put_delta": 0.25,
        "call_delta": 0.30,
        "min_dte": 21,
        "put_iv_mult": 0.95,
        "call_iv_mult": 0.80,
        "call_above_basis": False,
        "commission": 0.65,
        "assignment_fee": 0.0,
        "spread_pct": 0.05,
        "min_spread": 0.03,
        "min_premium": 0.05,
        "multiplier": 100,
    }

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"Wheel got unknown params {sorted(unknown)}")
        self.p = {**self.params, **params}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.p.items())
        return f"Wheel({args})"

    def run(self, inputs: pd.DataFrame, initial_capital: float = 5_000.0, start=None, end=None) -> Result:
        """Simulate on a frame with columns close, iv, and optional dividend, rate."""
        p, m, sym = self.p, self.p["multiplier"], self.p["symbol"]
        df = inputs.loc[start:end].dropna(subset=["close", "iv"])
        idx = df.index
        S_a = df["close"].to_numpy(float)
        iv_a = df["iv"].to_numpy(float)
        div_a = df.get("dividend", pd.Series(0.0, index=idx)).fillna(0.0).to_numpy(float)
        r_a = df.get("rate", pd.Series(0.0, index=idx)).ffill().fillna(0.0).to_numpy(float)
        # Trailing 12-month dividend yield for pricing (known on each day).
        q_a = (pd.Series(div_a, index=idx).rolling("365D").sum() / df["close"]).fillna(0.0).to_numpy(float)
        expiries = monthly_expiries(idx)

        cash, shares, basis, opt = float(initial_capital), 0, np.nan, None
        n = len(idx)
        eq, pos, w, costs = np.empty(n), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
        trades = []

        for i, d in enumerate(idx):
            S, r, q = S_a[i], r_a[i], q_a[i]
            if i > 0:
                days = (d - idx[i - 1]).days
                cash *= (1 + r_a[i - 1]) ** (days / 365.0)
                cash += div_a[i] * shares

            # Settle at expiry on the close.
            if opt is not None and d >= opt["expiry"]:
                K, k = opt["K"], opt["n"]
                if opt["call"] and S > K:
                    cash += K * m * k - p["assignment_fee"]
                    shares -= m * k
                    trades.append((d, sym, -m * k, S, K, p["assignment_fee"], 0.0))
                    costs[i] += p["assignment_fee"]
                elif not opt["call"] and S < K:
                    cash -= K * m * k + p["assignment_fee"]
                    shares += m * k
                    basis = K
                    trades.append((d, sym, m * k, S, K, p["assignment_fee"], 0.0))
                    costs[i] += p["assignment_fee"]
                opt = None

            # Open the next leg.
            if opt is None:
                exp = expiries[expiries.searchsorted(d + pd.Timedelta(days=p["min_dte"]))]
                T = (exp - d).days / 365.0
                call = shares >= m
                sigma = iv_a[i] * (p["call_iv_mult"] if call else p["put_iv_mult"])
                delta = p["call_delta"] if call else p["put_delta"]
                K = strike_for_delta(S, T, r, q, sigma, delta, call, strike_step(S))
                if call and p["call_above_basis"] and K < basis:
                    K = math.ceil(basis / strike_step(S)) * strike_step(S)
                k = shares // m if call else int(cash // (K * m))
                mid = bs_price(S, K, T, r, q, sigma, call)
                half = max(p["spread_pct"] * mid, p["min_spread"]) / 2
                fill = mid - half
                if k > 0 and fill >= p["min_premium"]:
                    fee = p["commission"] * k
                    cash += fill * m * k - fee
                    costs[i] += fee + half * m * k
                    label = f"{sym} {exp.date()} {K:g}{'C' if call else 'P'}"
                    trades.append((d, label, -m * k, mid, fill, fee, half * m * k))
                    opt = {"call": call, "K": K, "expiry": exp, "n": k, "mult": sigma / iv_a[i]}

            # Mark to market at model mid.
            liab = 0.0
            if opt is not None:
                T = (opt["expiry"] - d).days / 365.0
                liab = bs_price(S, opt["K"], T, r, q, iv_a[i] * opt["mult"], opt["call"]) * m * opt["n"]
            equity = cash + shares * S - liab
            eq[i] = equity
            short_put = -opt["n"] if opt is not None and not opt["call"] else 0
            short_call = -opt["n"] if opt is not None and opt["call"] else 0
            pos[i] = (shares, short_put, short_call)
            # Exposure as notional: shares, and strikes the short puts commit to buy.
            K_now = opt["K"] if opt is not None else 0.0
            w[i] = (shares * S / equity, short_put * m * K_now / equity, short_call * m * K_now / equity)

        cols = [sym, f"{sym}_short_put", f"{sym}_short_call"]
        return Result(
            equity=pd.Series(eq, index=idx, name="equity"),
            positions=pd.DataFrame(pos, index=idx, columns=cols),
            weights=pd.DataFrame(w, index=idx, columns=cols),
            trades=pd.DataFrame(trades, columns=["date", "symbol", "qty", "ref_price", "fill_price", "fees", "slippage"]),
            costs=pd.Series(costs, index=idx, name="costs"),
            initial_capital=initial_capital,
            strategy=repr(self),
        )
