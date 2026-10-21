"""Defined-risk credit spreads: bull put spreads, bear call spreads, iron condors.

Selling a spread earns part of the variance risk premium (implied vol above
the realized vol that follows) and pays part of it back by buying the wing.
Index puts carry a steep skew, so the long wing of a put spread is the
richest option on the chain; the strategy keeps less premium per dollar of
risk than a naked put (Cboe's CNDR iron condor index has had lower returns
than its PUT index since 1986, at lower volatility).

There is no free option history, so every leg is priced with Black-Scholes
off a modeled volatility surface:

* Level: VIX (or the wheel's VIX-based proxy for other underlyings) times
  `atm_mult`. VIX is a variance swap rate that includes the skew, so it sits
  above at-the-money IV. 0.88 is the multiple at which a cost-free replica of
  Cboe's PUT index matches its 1996-2026 CAGR (examples/run_wheel.py).
* Term: VIX9D, VIX, VIX3M and VIX6M interpolated in total variance to each
  leg's expiry, as a ratio to VIX. Days without them (before 2006-2011) use a
  flat curve.
* Skew: with z = ln(K/S) / (sigma_atm * sqrt(T)), IV = sigma_atm *
  (1 - skew_slope*z + curve*z^2), curve = put_curve below the money and
  call_curve above. The defaults are a least-squares fit to 745 late-day SPY
  option trades on 2026-10-20 with 17 to 71 days to expiry (RMSE 0.033 in
  IV/VIX): a 25-delta put trades ~1.24x ATM IV, a 5-delta put ~1.75x, a
  20-delta call ~0.84x. One day's skew is not every day's skew, so
  `skew_scale` scales slope and curves for sensitivity runs. A flatter skew
  makes the long put wing look cheap and flatters put spreads.

Trades settle as European, cash-settled options (XSP-style): no early
assignment, no pin risk, and a position held to expiry settles at intrinsic
on that day's close. Exits are checked on daily closes only, so intraday
stops and gaps are not modeled. Costs: `commission` per contract per leg on
every open and close (none at expiry), and half the bid-ask spread on every
leg traded, the larger of `spread_pct` of the leg's mid and `min_spread`,
widened by VIX/20 when VIX is above 20 if `stress_spreads`.

Old prices make dollar costs meaningless: SPY was $140 in 2000, so a
"$5 wide" spread then was 3.5% of spot and $0.65 a leg was a far bigger
share of the premium. With `rebase` (default) each position is opened and
managed as if the underlying were at the frame's last close: strikes, fills
and dollar P&L use spot times (last close / spot at entry), while the moves
are the real ones. Results then answer "what would this trade have done at
today's price level".

Like `Wheel`, this is a standalone daily simulator returning `Result`, since
target weights cannot express multi-leg option positions.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..core import Result
from .wheel import bs_delta, bs_price, monthly_expiries

_N = NormalDist()

# Volatility-index tenors in calendar days, used for the term structure.
TERM_COLUMNS = {"vix9d": 9, "vix": 30, "vix3m": 91, "vix6m": 182}


def term_factor(row_vols: dict[int, float], days: float) -> float:
    """IV at `days` over 30-day IV, interpolating total variance between tenors.

    row_vols maps tenor (days) to that day's index level; missing tenors are
    skipped, and with only the 30-day level the curve is flat.
    """
    pts = sorted((d, v) for d, v in row_vols.items() if v is not None and np.isfinite(v) and v > 0)
    v30 = row_vols.get(30)
    if v30 is None or not np.isfinite(v30) or v30 <= 0 or len(pts) < 2:
        return 1.0
    ds = np.array([d for d, _ in pts], float)
    var = np.array([v * v * d for d, v in pts])
    if days <= ds[0]:
        tv = var[0] * days / ds[0]
    elif days >= ds[-1]:
        tv = var[-1] * days / ds[-1]
    else:
        tv = np.interp(days, ds, var)
    return math.sqrt(tv / days) / v30


def smile_iv(S, K, T, atm, slope, put_curve, call_curve, floor=0.5) -> float:
    """Quadratic-in-standardized-moneyness smile around at-the-money IV `atm`."""
    if T <= 0:
        return atm
    z = math.log(K / S) / (atm * math.sqrt(T))
    curve = put_curve if z < 0 else call_curve
    return atm * max(1 - slope * z + curve * z * z, floor)


def strike_for_delta_smile(S, T, r, q, delta, call, step, vol_at) -> float:
    """Listed strike whose |delta|, at its own smile IV, is closest to `delta`."""
    sigma = vol_at(S)
    for _ in range(4):
        z = _N.inv_cdf(delta * math.exp(q * T)) if call else -_N.inv_cdf(delta * math.exp(q * T))
        K = S * math.exp((r - q + 0.5 * sigma**2) * T - z * sigma * math.sqrt(T))
        sigma = vol_at(K)
    lo = math.floor(K / step) * step
    cands = [k for k in (lo - step, lo, lo + step, lo + 2 * step) if k > 0]
    return min(cands, key=lambda k: abs(abs(bs_delta(S, k, T, r, q, vol_at(k), call)) - delta))


def load_spread_inputs(symbol: str = "SPY", start="1996-01-01", end=None, refresh=False) -> pd.DataFrame:
    """Wheel inputs (raw close, dividend, VIX-based iv, T-bill rate) plus VIX term indexes.

    Needs Yahoo Finance and FRED.
    """
    from ..data.yahoo import load_bars
    from .wheel import load_wheel_inputs

    df = load_wheel_inputs(symbol, start="1990-01-01", refresh=refresh)
    for col, ticker in (("vix9d", "^VIX9D"), ("vix", "^VIX"), ("vix3m", "^VIX3M"), ("vix6m", "^VIX6M")):
        try:
            df[col] = load_bars(ticker, start="1990-01-01")["close"].reindex(df.index).ffill(limit=5) / 100
        except Exception:  # an index Yahoo no longer serves just flattens the curve
            df[col] = np.nan
    return df.loc[start:end]


class CreditSpreads:
    """Short vertical spreads or iron condors with take-profit, time and stop exits.

    Every `entry_spacing` trading days, while fewer than `max_positions` are
    open, opens a position in the first monthly expiry at least `min_dte`
    days out: short strikes nearest `put_delta` / `call_delta` (at their smile
    IV), long wings `width_pct` of spot further out (at least one strike).
    Contracts are sized so the position's max loss is at most
    `risk_per_trade` of equity, or one contract if that rounds to zero and
    `allow_one`. Cash always covers every open position's full loss (the
    margin a broker holds for spreads), so equity cannot go below zero. A position closes when its mid value
    has fallen to (1 - take_profit) of the credit, when its loss reaches
    `stop_loss` times the credit, at `exit_dte` days left, or at expiry.

    Entry filters: `vix_min` (VIX level), `vrp_filter` (VIX above the
    underlying's trailing 21-day realized vol) and `trend_filter` (close above
    its 200-day average; puts and condors only).
    """

    name = "credit_spreads"
    params = {
        "symbol": "SPY",
        "structure": "put_spread",  # put_spread | call_spread | iron_condor
        "put_delta": 0.25,
        "call_delta": 0.16,
        "width_pct": 0.0065,
        "step": 1.0,
        "min_dte": 30,
        "take_profit": 0.5,
        "exit_dte": 21,
        "stop_loss": None,
        "max_positions": 1,
        "entry_spacing": 1,
        "risk_per_trade": 0.10,
        "allow_one": True,
        "vix_min": 0.0,
        "vrp_filter": False,
        "trend_filter": False,
        "atm_mult": 0.88,
        "skew_slope": 0.344,
        "put_curve": 0.016,
        "call_curve": 0.164,
        "skew_scale": 1.0,
        "commission": 0.65,
        "spread_pct": 0.005,
        "min_spread": 0.02,
        "stress_spreads": True,
        "multiplier": 100,
        "rebase": True,
    }

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"CreditSpreads got unknown params {sorted(unknown)}")
        self.p = {**self.params, **params}
        if self.p["structure"] not in ("put_spread", "call_spread", "iron_condor"):
            raise ValueError(f"unknown structure {self.p['structure']!r}")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.p.items() if v != self.params[k])
        return f"CreditSpreads({args})"

    # -- pricing -----------------------------------------------------------
    def _vol_fn(self, S, T, iv30, terms):
        p = self.p
        atm = iv30 * p["atm_mult"] * term_factor(terms, T * 365)
        k = p["skew_scale"]
        return lambda K: smile_iv(S, K, T, atm, p["skew_slope"] * k, p["put_curve"] * k, p["call_curve"] * k)

    def _value(self, legs, S, T, r, q, vol_at) -> float:
        """Mid value per share of the position (what it costs to buy back)."""
        return sum(sign * bs_price(S, K, T, r, q, vol_at(K), call) for K, call, sign in legs)

    @staticmethod
    def _max_payout(legs) -> float:
        """Most the position can be worth at expiry: the wider wing."""
        width = 0.0
        for call in (False, True):
            ks = sorted(K for K, c, _ in legs if c == call)
            if ks:
                width = max(width, ks[-1] - ks[0])
        return width

    def _half_spread(self, mid, vix) -> float:
        p = self.p
        half = max(p["spread_pct"] * mid, p["min_spread"]) / 2
        if p["stress_spreads"] and vix > 0.20:
            half *= vix / 0.20
        return half

    def _legs(self, S, T, r, q, vol_at):
        """(strike, is_call, +1 short / -1 long) for the configured structure."""
        p, step = self.p, self.p["step"]
        width = max(round(S * p["width_pct"] / step), 1) * step
        legs = []
        if p["structure"] in ("put_spread", "iron_condor"):
            Kp = strike_for_delta_smile(S, T, r, q, p["put_delta"], False, step, vol_at)
            legs += [(Kp, False, 1), (Kp - width, False, -1)]
        if p["structure"] in ("call_spread", "iron_condor"):
            Kc = strike_for_delta_smile(S, T, r, q, p["call_delta"], True, step, vol_at)
            legs += [(Kc, True, 1), (Kc + width, True, -1)]
        return legs, width

    # -- simulation ----------------------------------------------------------
    def run(self, inputs: pd.DataFrame, initial_capital: float = 5_000.0, start=None, end=None) -> Result:
        """Simulate on a frame with close, iv and optional dividend, rate, vix9d/vix/vix3m/vix6m."""
        p, m, sym = self.p, self.p["multiplier"], self.p["symbol"]
        df = inputs.dropna(subset=["close", "iv"])
        # Filters need history before `start`, so compute them on the full frame first.
        rv21 = np.log(df["close"]).diff().rolling(21).std() * math.sqrt(252)
        sma200 = df["close"].rolling(200).mean()
        q_full = (df.get("dividend", pd.Series(0.0, index=df.index)).fillna(0.0).rolling("365D").sum() / df["close"]).fillna(0.0)
        df = df.loc[start:end]
        idx = df.index
        S_a = df["close"].to_numpy(float)
        iv_a = df["iv"].to_numpy(float)
        r_a = df.get("rate", pd.Series(0.0, index=idx)).ffill().fillna(0.0).to_numpy(float)
        q_a = q_full.reindex(idx).to_numpy(float)
        rv_a = rv21.reindex(idx).to_numpy(float)
        sma_a = sma200.reindex(idx).to_numpy(float)
        term_cols = {d: df[c].to_numpy(float) for c, d in TERM_COLUMNS.items() if c in df}
        vix_a = term_cols.get(30, iv_a)
        expiries = monthly_expiries(idx)

        cash, open_pos, last_entry = float(initial_capital), [], -10**9
        n = len(idx)
        eq, pos, w, costs = np.empty(n), np.zeros((n, 1)), np.zeros((n, 1)), np.zeros(n)
        trades, closed = [], []

        def terms_at(i):
            t = {d: a[i] for d, a in term_cols.items()}
            t.setdefault(30, iv_a[i])
            return t

        def leg_cost(legs, S, T, r, q, vol_at, k, vix, closing):
            """Commission + half-spread dollars for trading every leg, and the fill per share."""
            mid = self._value(legs, S, T, r, q, vol_at)
            slip = sum(self._half_spread(bs_price(S, K, T, r, q, vol_at(K), c), vix) for K, c, _ in legs)
            fee = p["commission"] * len(legs) * k
            # Opening sells at mid minus slippage; closing buys back at mid plus
            # slippage, but never above the width (holding to expiry costs no more).
            if closing:
                fill = min(mid + slip, self._max_payout(legs))
                slip = fill - mid
            else:
                fill = mid - slip
            return fee, slip * m * k, fill

        for i, d in enumerate(idx):
            S, r, q, vix = S_a[i], r_a[i], q_a[i], vix_a[i]
            if i > 0:
                cash *= (1 + r_a[i - 1]) ** ((d - idx[i - 1]).days / 365.0)

            # Manage open positions on today's close.
            still_open = []
            for ps in open_pos:
                T = max((ps["expiry"] - d).days, 0) / 365.0
                k, S = ps["n"], S_a[i] * ps["f"]
                if T <= 0:
                    value = sum(sign * (max(S - K, 0) if c else max(K - S, 0)) for K, c, sign in ps["legs"])
                    cash -= value * m * k
                    trades.append((d, ps["label"] + " settle", k, value, value, 0.0, 0.0))
                    closed.append({**ps, "exit": d, "exit_value": value, "reason": "expiry"})
                    continue
                vol_at = self._vol_fn(S, T, iv_a[i], terms_at(i))
                value = self._value(ps["legs"], S, T, r, q, vol_at)
                dte = (ps["expiry"] - d).days
                reason = None
                if value <= ps["credit"] * (1 - p["take_profit"]):
                    reason = "take_profit"
                elif p["stop_loss"] is not None and value - ps["credit"] >= p["stop_loss"] * ps["credit"]:
                    reason = "stop_loss"
                elif p["exit_dte"] is not None and dte <= p["exit_dte"]:
                    reason = "time"
                if reason:
                    fee, slip, fill = leg_cost(ps["legs"], S, T, r, q, vol_at, k, vix, closing=True)
                    cash -= fill * m * k + fee
                    costs[i] += fee + slip
                    trades.append((d, ps["label"] + " close", k, value, fill, fee, slip))
                    closed.append({**ps, "exit": d, "exit_value": fill, "reason": reason})
                else:
                    ps["value"] = value
                    still_open.append(ps)
            open_pos = still_open
            S = S_a[i]

            # Open a new position.
            can_enter = (
                len(open_pos) < p["max_positions"]
                and i - last_entry >= p["entry_spacing"]
                and vix >= p["vix_min"]
                and (not p["vrp_filter"] or (np.isfinite(rv_a[i]) and vix > rv_a[i]))
                and (not p["trend_filter"] or p["structure"] == "call_spread" or (np.isfinite(sma_a[i]) and S > sma_a[i]))
            )
            if can_enter:
                exp = expiries[expiries.searchsorted(d + pd.Timedelta(days=p["min_dte"]))]
                T = (exp - d).days / 365.0
                f = S_a[-1] / S if p["rebase"] else 1.0
                S = S * f
                vol_at = self._vol_fn(S, T, iv_a[i], terms_at(i))
                legs, width = self._legs(S, T, r, q, vol_at)
                _, _, fill = leg_cost(legs, S, T, r, q, vol_at, 1, vix, closing=False)
                fee1 = p["commission"] * len(legs)
                max_loss = (width - fill) * m + 2 * fee1  # per contract, fees both ways
                # Cash already holds the open positions' credits, so what it must
                # cover is each open position's full width, not width minus credit.
                free = cash - sum((ps["width"] * m + p["commission"] * len(ps["legs"])) * ps["n"] for ps in open_pos)
                equity_now = cash - sum(ps.get("value", ps["credit"]) * m * ps["n"] for ps in open_pos)
                k = int(min(p["risk_per_trade"] * equity_now, free) // max_loss) if max_loss > 0 else 0
                if k == 0 and p["allow_one"] and 0 < max_loss <= free:
                    k = 1
                if fill > 0 and k > 0:
                    fee, slip, fill = leg_cost(legs, S, T, r, q, vol_at, k, vix, closing=False)
                    cash += fill * m * k - fee
                    costs[i] += fee + slip
                    desc = "/".join(f"{K:g}{'C' if c else 'P'}" for K, c, _ in legs)
                    label = f"{sym} {exp.date()} {desc}"
                    mid = self._value(legs, S, T, r, q, vol_at)
                    trades.append((d, label + " open", -k, mid, fill, fee, slip))
                    open_pos.append({"legs": legs, "expiry": exp, "n": k, "credit": fill, "value": mid,
                                     "max_loss": max_loss, "width": width, "label": label, "entry": d, "spot": S, "f": f})
                    last_entry = i

            liab = sum(ps["value"] * m * ps["n"] for ps in open_pos)
            eq[i] = cash - liab
            pos[i, 0] = sum(ps["n"] for ps in open_pos)
            w[i, 0] = sum(ps["max_loss"] * ps["n"] for ps in open_pos) / eq[i] if eq[i] > 0 else 0.0

        self.closed_ = pd.DataFrame(
            [{"entry": c["entry"], "exit": c["exit"], "expiry": c["expiry"], "label": c["label"], "contracts": c["n"],
              "credit": c["credit"], "exit_value": c["exit_value"], "max_loss": c["max_loss"],
              "pnl": (c["credit"] - c["exit_value"]) * m * c["n"], "reason": c["reason"]} for c in closed]
        )
        col = f"{sym}_{p['structure']}"
        return Result(
            equity=pd.Series(eq, index=idx, name="equity"),
            positions=pd.DataFrame(pos, index=idx, columns=[col]),
            weights=pd.DataFrame(w, index=idx, columns=[col]),
            trades=pd.DataFrame(trades, columns=["date", "symbol", "qty", "ref_price", "fill_price", "fees", "slippage"]),
            costs=pd.Series(costs, index=idx, name="costs"),
            initial_capital=initial_capital,
            strategy=repr(self),
        )
