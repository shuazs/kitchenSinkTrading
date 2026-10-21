"""Intraday strategy bench: many day-trading rules on one minute-grid engine.

`intraday.py` holds two paper-faithful simulators (first half hour, noise
area). This module tests families of intraday rules with parameter grids
under one set of assumptions, so they can be compared and walk-forward
selected:

- Every rule returns an exposure array (days x minutes): the fraction of
  equity held long (+) or short (-) after the price known at minute m.
- `simulate` fills each change one minute later (lag=1), at that minute's
  close, charges `cost_bps` per unit of exposure traded (half spread +
  slippage + fees), and flattens at the session close. Exposure is a
  fraction of equity, so $5,000 with fractional shares compounds daily.
- `walk_forward` picks, at the start of each test year, the variant with the
  best net Sharpe on all earlier days and trades it for that year.

Families and sources:

- `momentum_window`: Gao, Han, Li and Zhou (2018) and Baltussen, Da, Lammers
  and Martens (2021): the return up to some time of day predicts the last
  half hour or hour. Wen et al. (2022) find the same in bitcoin's UTC day.
- `noise_area`: Zarattini, Aziz and Barbon (2024), as in `intraday.py`.
- `opening_range`: opening range breakout (Crabel 1990; Zarattini and Aziz
  2023, "Can Day Trading Really Be Profitable?", 5-minute ORB on QQQ).
- `vwap_trend`: Zarattini and Aziz (2023), long above VWAP, short below.
- `vwap_reversion`: practitioner fade of stretches away from VWAP.
- `gap`: fade or follow the overnight gap (Plastun et al. 2019 and
  practitioner "gap and go" / "gap fill").
- `hour_of_day`: crypto intraday seasonality (Eross et al. 2019, Baur et al.
  2019): hold only the UTC hours with the best trailing mean return.
- `session_split`: overnight versus regular-session returns (Cliff, Cooper
  and Gulen 2008; Lou, Polk and Skouras 2019).
"""
from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from .intraday import minute_grid


class Grid:
    """Day x minute arrays for one instrument and session.

    `px[:, 0]` is the session's opening price and `px[:, m]` the close known
    m minutes later, so `px` has length + 1 columns. `hi`, `lo`, `vol` and
    `vwap` line up with `px[:, 1:]` (minute 1 to length) and are padded with
    the open at column 0.
    """

    def __init__(self, bars: pd.DataFrame, start: int = 570, length: int = 390):
        g = minute_grid(bars, start=start, length=length)
        close = g["close"]
        keep = close.notna().sum(axis=1) >= 0.8 * close.shape[1]
        close = close[keep]
        opens = g["open"][keep].bfill(axis=1).iloc[:, 0]
        self.days = close.index
        self.length = length
        c = close.bfill(axis=1).to_numpy(float)
        o = opens.to_numpy(float)
        o = np.where(np.isfinite(o), o, c[:, 0])
        self.px = np.column_stack([o, c])
        self.hi = np.column_stack([o, g["high"][keep].ffill(axis=1).bfill(axis=1).to_numpy(float)])
        self.lo = np.column_stack([o, g["low"][keep].ffill(axis=1).bfill(axis=1).to_numpy(float)])
        v = g["volume"][keep].to_numpy(float)
        typical = (self.hi[:, 1:] + self.lo[:, 1:] + c) / 3
        cv = np.cumsum(v, axis=1)
        vwap = np.where(cv > 0, np.cumsum(typical * v, axis=1) / np.where(cv > 0, cv, 1), np.nan)
        # No volume (or none yet): fall back to the running mean of typical price.
        twap = np.cumsum(typical, axis=1) / np.arange(1, length + 1)
        self.vwap = np.column_stack([o, np.where(np.isfinite(vwap), vwap, twap)])
        # The previous session's close, only when it is the previous trading day's.
        prev = np.r_[np.nan, self.px[:-1, -1]]
        gap_days = np.r_[np.inf, np.diff(self.days.values).astype("timedelta64[D]").astype(float)]
        prev[gap_days > 5] = np.nan
        # A >20% overnight jump is a source splice or bad print, not a gap.
        prev[np.abs(self.px[:, 0] / prev - 1) > 0.2] = np.nan
        self.prev_close = prev
        daily = self.px[:, -1] / prev - 1
        self.daily_vol = pd.Series(daily).rolling(14, min_periods=10).std().shift(1).to_numpy()
        rng = np.maximum(self.hi.max(axis=1), np.nan_to_num(prev, nan=-np.inf)) - \
            np.minimum(self.lo.min(axis=1), np.nan_to_num(prev, nan=np.inf))
        self.atr = pd.Series(rng / self.px[:, -1]).rolling(14, min_periods=10).mean().shift(1).to_numpy()

    def __len__(self) -> int:
        return len(self.days)


def simulate(grid: Grid, pos: np.ndarray, cost_bps: float, lag: int = 1) -> pd.DataFrame:
    """Daily net returns of an exposure array.

    pos[:, m] (m = 0 .. length-1) is the exposure wanted after the price at
    minute m. It is filled `lag` minutes later and held minute by minute;
    whatever is open is closed at the session's last price.
    """
    n, L = pos.shape
    pos = np.nan_to_num(pos)
    held = np.zeros_like(pos)
    held[:, lag:] = pos[:, : L - lag]
    held[:, -1] = 0.0
    r = grid.px[:, 1:] / grid.px[:, :-1] - 1
    gross = np.nansum(held * r, axis=1)
    turnover = np.abs(np.diff(np.column_stack([np.zeros(n), held, np.zeros(n)]), axis=1)).sum(axis=1)
    ret = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"ret": ret, "gross": gross, "turnover": turnover, "n_trades": (turnover > 0).astype(int)},
                        index=grid.days)


def _hold(n: int, L: int, side: np.ndarray, start: int, end: int | None = None) -> np.ndarray:
    pos = np.zeros((n, L))
    pos[:, start: end or L] = np.nan_to_num(side)[:, None]
    return pos


# ---- rule families -------------------------------------------------------------------------------

def momentum_window(g: Grid, signal_end: int = 30, hold: int = 30, ref: str = "prev_close",
                    long_only: bool = False) -> np.ndarray:
    """Sign of the return from `ref` (previous close or today's open) to
    minute `signal_end`, held for the session's last `hold` minutes."""
    L = g.length
    start = L - hold
    base = g.prev_close if ref == "prev_close" else g.px[:, 0]
    side = np.sign(g.px[:, min(signal_end, start)] / base - 1)
    if long_only:
        side = np.clip(side, 0, None)
    return _hold(len(g), L, side, start)


def noise_area(g: Grid, lookback: int = 14, band_mult: float = 1.0, check_every: int = 30,
               long_only: bool = False, target_vol: float = 0.02, max_leverage: float = 1.0) -> np.ndarray:
    """Zarattini et al.: enter on a close outside the average-move band at a
    check mark, exit on a close back inside max(band, VWAP) (min for shorts)."""
    n, L = len(g), g.length
    o = g.px[:, 0]
    move = np.abs(g.px[:, 1:] / o[:, None] - 1)
    sigma = pd.DataFrame(move).rolling(lookback).mean().shift(1).to_numpy()
    up = (1 + band_mult * sigma) * np.fmax(o, g.prev_close)[:, None]
    dn = (1 - band_mult * sigma) * np.fmin(o, g.prev_close)[:, None]
    lev = np.clip(target_vol / g.daily_vol, 0, max_leverage)
    lev = np.where(np.isfinite(lev), lev, 0.0)
    pos = np.zeros((n, L))
    state = np.zeros(n)
    for m in range(1, L):
        if m >= 30 and m % check_every == 0:
            px, ub, lb, vw = g.px[:, m], up[:, m - 1], dn[:, m - 1], g.vwap[:, m]
            new = state.copy()
            new[(state == 1) & (px < np.fmax(ub, vw))] = 0
            new[(state == -1) & (px > np.fmin(lb, vw))] = 0
            new[px > ub] = 1
            new[px < lb] = 0 if long_only else -1
            state = np.where(np.isfinite(ub), new, 0)
        pos[:, m] = state
    return pos * lev[:, None]


def opening_range(g: Grid, minutes: int = 30, mode: str = "breakout", stop: str = "range",
                  long_only: bool = False) -> np.ndarray:
    """Opening range breakout, one trade a day.

    mode "breakout": enter when a close leaves the first `minutes`' high-low
    range. mode "candle": enter at the end of the range in the direction of
    the range's open-to-close move (Zarattini and Aziz). stop "range": exit on
    a close beyond the other side of the range; "atr": exit on a close 10%
    of the 14-day average range against the entry; "none": hold to the close.
    """
    n, L = len(g), g.length
    or_hi = g.hi[:, 1: minutes + 1].max(axis=1)
    or_lo = g.lo[:, 1: minutes + 1].min(axis=1)
    pos = np.zeros((n, L))
    state = np.zeros(n)
    done = np.zeros(n, bool)
    entry = np.full(n, np.nan)
    if mode == "candle":
        side = np.sign(g.px[:, minutes] - g.px[:, 0])
        if long_only:
            side = np.clip(side, 0, None)
        state, entry, done = side.copy(), g.px[:, minutes].copy(), side != 0
    for m in range(minutes, L):
        px = g.px[:, m]
        if mode == "breakout" and m > minutes:
            go_up = ~done & (px > or_hi)
            go_dn = ~done & (px < or_lo) & (not long_only)
            state[go_up], state[go_dn] = 1, -1
            entry[go_up | go_dn] = px[go_up | go_dn]
            done |= go_up | go_dn
        if stop == "range":
            hit = ((state == 1) & (px < or_lo)) | ((state == -1) & (px > or_hi))
        elif stop == "atr":
            d = 0.1 * g.atr * entry
            hit = ((state == 1) & (px < entry - d)) | ((state == -1) & (px > entry + d))
        else:
            hit = np.zeros(n, bool)
        state[hit] = 0
        pos[:, m] = state
    return pos


def vwap_trend(g: Grid, check_every: int = 5, first: int = 5, band: float = 0.0,
               long_only: bool = False) -> np.ndarray:
    """Long when the price is above VWAP by `band`, short (or flat) when below
    by `band`, rechecked every `check_every` minutes from minute `first`."""
    n, L = len(g), g.length
    pos = np.zeros((n, L))
    state = np.zeros(n)
    for m in range(first, L):
        if (m - first) % check_every == 0:
            dev = g.px[:, m] / g.vwap[:, m] - 1
            state = np.where(dev > band, 1.0, np.where(dev < -band, 0.0 if long_only else -1.0, state))
        pos[:, m] = state
    return pos


def vwap_reversion(g: Grid, k: float = 0.5, first: int = 30, stop: bool = False,
                   long_only: bool = False) -> np.ndarray:
    """Fade a stretch of more than k daily vols away from VWAP; exit when the
    price is back at VWAP, at the close, or (stop=True) at 2k away."""
    n, L = len(g), g.length
    pos = np.zeros((n, L))
    state = np.zeros(n)
    for m in range(first, L):
        dev = (g.px[:, m] / g.vwap[:, m] - 1) / g.daily_vol
        dev = np.where(np.isfinite(dev), dev, 0.0)
        enter_long = (state == 0) & (dev < -k)
        enter_short = (state == 0) & (dev > k) & (not long_only)
        state[enter_long], state[enter_short] = 1, -1
        back = ((state == 1) & (dev >= 0)) | ((state == -1) & (dev <= 0))
        if stop:
            back |= ((state == 1) & (dev < -2 * k)) | ((state == -1) & (dev > 2 * k))
        state[back] = 0
        pos[:, m] = state
    return pos


def gap(g: Grid, threshold: float = 0.5, direction: str = "fade", entry: int = 1, exit: int | None = None,
        long_only: bool = False) -> np.ndarray:
    """Trade gaps bigger than `threshold` daily vols: against the gap (fade)
    or with it (follow), from minute `entry` to `exit` (None = close)."""
    z = (g.px[:, 0] / g.prev_close - 1) / g.daily_vol
    side = np.where(np.abs(z) > threshold, np.sign(z), 0.0)
    if direction == "fade":
        side = -side
    if long_only:
        side = np.clip(side, 0, None)
    return _hold(len(g), g.length, side, entry, exit)


def hour_of_day(g: Grid, lookback: int = 365, rule: str = "top4", refit_every: int = 21) -> np.ndarray:
    """Long only in the clock hours whose trailing `lookback`-day mean return
    is best: "topK" = the K best hours, "tX" = hours with t-stat above X.
    Refit every `refit_every` days on data up to the previous day."""
    n, L = len(g), g.length
    hours = L // 60
    hr = g.px[:, 60::60][:, :hours] / g.px[:, 0:L:60][:, :hours] - 1
    pos = np.zeros((n, L))
    pick = np.zeros(hours, bool)
    for d in range(n):
        if d % refit_every == 0:
            past = hr[max(0, d - lookback): d]
            pick = np.zeros(hours, bool)
            if len(past) >= min(lookback, 120):
                mu, se = np.nanmean(past, axis=0), np.nanstd(past, axis=0) / np.sqrt(len(past))
                if rule.startswith("top"):
                    best = np.argsort(-mu)[: int(rule[3:])]
                    pick[best[mu[best] > 0]] = True
                else:
                    pick = mu / se > float(rule[1:])
        pos[d] = np.repeat(pick.astype(float), 60)[:L]
    return pos


def session_split(g: Grid, cost_bps: float) -> pd.DataFrame:
    """Daily returns of holding only overnight (close to open), only the
    session (open to close), and both, each overnight trade paying two fills,
    plus overnight only after a down session (a reversal variant)."""
    on = g.px[:, 0] / g.prev_close - 1
    day = g.px[:, -1] / g.px[:, 0] - 1
    c = 2 * cost_bps / 1e4
    prev_day = np.r_[np.nan, day[:-1]]
    out = pd.DataFrame({
        "overnight only": np.nan_to_num(on) - c * np.isfinite(on),
        "session only": day - c,
        "overnight after down session": np.where(prev_day < 0, np.nan_to_num(on) - c, 0.0),
        "buy and hold": np.nan_to_num((1 + on) * (1 + day) - 1),
    }, index=g.days)
    return out


def high_vol_days(g: Grid, q: float, window: int = 252) -> np.ndarray:
    """True on days whose trailing daily volatility (known at the open) is
    above its `q` quantile over the previous `window` days. Trend rules earn
    a few basis points per trade on average while costs are fixed per trade,
    so the trade is worth more on volatile days. q=0 keeps every day."""
    if q <= 0:
        return np.ones(len(g), bool)
    v = pd.Series(g.daily_vol)
    thr = v.rolling(window, min_periods=120).quantile(q).shift(1)
    return (v > thr).to_numpy()


# ---- grids, evaluation, walk-forward -------------------------------------------------------------

FAMILIES: dict[str, tuple[Callable, dict]] = {
    "end-of-day momentum": (momentum_window, {
        "signal_end": [30, 60, 120, 240, 330, 360], "hold": [30, 60],
        "ref": ["prev_close", "open"], "long_only": [False, True]}),
    "noise area": (noise_area, {
        "lookback": [10, 14, 20], "band_mult": [0.8, 1.0, 1.2], "check_every": [15, 30, 60],
        "long_only": [False, True]}),
    "opening range": (opening_range, {
        "minutes": [5, 15, 30, 60], "mode": ["breakout", "candle"], "stop": ["range", "atr", "none"],
        "long_only": [False, True]}),
    "VWAP trend": (vwap_trend, {
        "check_every": [1, 5, 15, 30], "first": [5, 30], "band": [0.0, 0.0005, 0.001],
        "long_only": [False, True]}),
    "VWAP reversion": (vwap_reversion, {
        "k": [0.25, 0.5, 0.75, 1.0], "first": [30, 60], "stop": [False, True], "long_only": [False, True]}),
    "gap": (gap, {
        "threshold": [0.25, 0.5, 1.0], "direction": ["fade", "follow"], "entry": [1, 5],
        "exit": [30, 120, None], "long_only": [False, True]}),
}

# Families also run with `high_vol_days` filters at these quantiles.
VOL_FILTERED = ("end-of-day momentum", "noise area", "opening range", "VWAP trend")
VOL_QUANTILES = [0.0, 0.5, 0.75]

CRYPTO_FAMILIES: dict[str, tuple[Callable, dict]] = {
    "hour of day": (hour_of_day, {"lookback": [180, 365, 730], "rule": ["top2", "top4", "top6", "t1", "t2"]}),
}


def variants(params: dict) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, v)) for v in product(*(params[k] for k in keys))]


def label(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def family_positions(g: Grid, name: str, fn: Callable, params: dict):
    """Yield (label, exposure) for every variant of a family, including the
    volatility-filtered copies for the families in VOL_FILTERED."""
    qs = VOL_QUANTILES if name in VOL_FILTERED else [0.0]
    masks = {q: high_vol_days(g, q) for q in qs}
    for p in variants(params):
        pos = fn(g, **p)
        for q in qs:
            yield label({**p, "vol_q": q}) if len(qs) > 1 else label(p), pos * masks[q][:, None]


def run_family(g: Grid, name: str, cost_bps: float, families: dict | None = None) -> pd.DataFrame:
    """Net daily returns of every variant, one column per variant label."""
    fn, params = (families or FAMILIES)[name]
    return pd.DataFrame({k: simulate(g, x, cost_bps)["ret"] for k, x in family_positions(g, name, fn, params)})


def stats(r: pd.Series) -> pd.Series:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 252
    sd = r.std()
    return pd.Series({
        "days": len(r),
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 and len(r) else np.nan,
        "vol": sd * np.sqrt(252),
        "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "t": r.mean() / sd * np.sqrt(len(r)) if sd > 0 else np.nan,
        "max_dd": (eq / eq.cummax() - 1).min() if len(r) else np.nan,
        "active": (r != 0).mean(),
    })


def sharpe(r: pd.DataFrame | pd.Series):
    sd = r.std()
    return r.mean() / sd.where(sd > 0) * np.sqrt(252)


def walk_forward(rets: pd.DataFrame, test_years: list[int], min_days: int = 200) -> tuple[pd.Series, list]:
    """Trade, in each test year, the column with the best Sharpe on all
    earlier days. Returns the stitched out-of-sample returns and the picks."""
    out, picks = [], []
    years = rets.index.year
    for y in test_years:
        train = rets[years < y]
        test = rets[years == y]
        if len(train) < min_days or test.empty:
            continue
        sh = sharpe(train)
        if sh.isna().all():
            continue
        best = sh.idxmax()
        picks.append((y, best))
        out.append(test[best])
    return (pd.concat(out) if out else pd.Series(dtype=float)), picks
