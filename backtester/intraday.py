"""Intraday momentum on SPY or QQQ: day-trading simulators on minute bars.

The daily engine trades at the next open, so intraday rules get their own
small simulators here. Each takes regular-session bars indexed by bar START
time (New York, tz-naive), as `data.alpaca.load_minute_bars` returns, and
returns one row per day with the position taken and the net return.

Two rules:

- `first_half_hour`: Gao, Han, Li and Zhou (2018), "Market Intraday
  Momentum". The return from yesterday's close to 10:00 predicts the last
  half hour, so trade 15:30 -> 16:00 in its direction.
- `noise_area`: Zarattini, Aziz and Barbon (2024), "Beat the Market: An
  Effective Intraday Momentum Strategy for S&P500 ETF (SPY)". Around each
  day's open, a band of the average absolute move from the open at that time
  of day over the last `lookback` days ("noise area"). A close outside the
  band at a half-hour mark opens a trade in that direction; a trailing stop
  at the band or VWAP, whichever is tighter, or the close ends it. Sized to
  `target_vol` daily volatility. They report 19.6%/yr, Sharpe 1.33, net,
  2007 to 2024.

Retail caveats: in a US margin account under $25,000, more than three day
trades in five business days flags the account as a pattern day trader.
A cash account avoids that rule but cannot short and must wait for sales
to settle (T+1) before reusing the cash, so `long_only=True` with one trade
a day is the realistic $5,000 setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import CostModel

SESSION_MINUTES = 390


def minute_grid(bars: pd.DataFrame, start: int = 570, length: int = SESSION_MINUTES) -> dict[str, pd.DataFrame]:
    """Pivot bars into day x minute frames keyed by field.

    Columns are minutes after the session start (`start` minutes after
    midnight, 570 = 09:30) at the bar's END (1 = 09:31 for a 1-minute bar,
    390 = 16:00), so column m holds the price known at m. Missing bars are
    forward-filled within the day. A 24-hour crypto day is start=0,
    length=1440 on UTC bars.
    """
    t = bars.index
    step = pd.Series(t).diff().dt.total_seconds().div(60)
    step = int(step[(step > 0) & (step < 60)].median())
    end = (t.hour * 60 + t.minute - start) + step
    keep = (end > 0) & (end <= length)
    b = bars[keep]
    day = pd.DatetimeIndex(b.index.normalize())
    minute = end[keep]
    cols = np.arange(step, length + 1, step)
    out = {}
    for f in ("open", "high", "low", "close", "volume"):
        g = pd.DataFrame({"day": day, "m": minute, "v": b[f].to_numpy()}).pivot_table(
            index="day", columns="m", values="v", aggfunc="last"
        )
        g = g.reindex(columns=cols)
        out[f] = g.fillna(0.0) if f == "volume" else g.ffill(axis=1)
    return out


def _price_at(close: pd.DataFrame, minute: int) -> pd.Series:
    cols = close.columns[close.columns <= minute]
    return close[cols[-1]]


def _cost_return(cost: CostModel, price: float, shares: float, equity: float) -> float:
    """Round-trip cost of a trade, as a fraction of equity."""
    if shares == 0:
        return 0.0
    adverse = (cost.spread_bps / 2 + cost.slippage_bps) / 1e4 * price * shares * 2
    return (adverse + cost.fees(price, shares) * 2) / equity


def first_half_hour(bars: pd.DataFrame, long_only: bool = True, threshold: float = 0.0,
                    capital: float = 5_000.0, cost: CostModel | None = None) -> pd.DataFrame:
    """Gao et al.: trade the last half hour in the direction of the return
    from the previous close to 10:00 (when it beats `threshold`)."""
    cost = cost or CostModel(spread_bps=1, slippage_bps=1)
    g = minute_grid(bars)
    close = g["close"]
    prev = close[SESSION_MINUTES].shift(1)
    r1 = _price_at(close, 30) / prev - 1
    entry, exit_ = _price_at(close, 360), close[SESSION_MINUTES]
    side = np.sign(r1).where(r1.abs() > threshold, 0.0)
    if long_only:
        side = side.clip(lower=0)
    out = pd.DataFrame({"signal": r1, "side": side.fillna(0.0)})
    gross = out["side"] * (exit_ / entry - 1)
    equity, rets = capital, []
    for d in out.index:
        shares = np.floor(equity / entry[d]) if out.at[d, "side"] and entry[d] > 0 else 0.0
        frac = shares * entry[d] / equity if equity > 0 else 0.0
        r = frac * gross[d] - _cost_return(cost, entry[d], shares, equity) if shares else 0.0
        rets.append(r)
        equity *= 1 + r
    out["ret"] = rets
    out["equity"] = capital * (1 + out["ret"]).cumprod()
    return out.iloc[1:]


def noise_area(bars: pd.DataFrame, lookback: int = 14, band_mult: float = 1.0, check_every: int = 30,
               long_only: bool = False, target_vol: float = 0.02, max_leverage: float = 1.0,
               capital: float = 5_000.0, cost: CostModel | None = None) -> pd.DataFrame:
    """Zarattini et al.'s noise-area breakout with a VWAP trailing stop.

    Decisions are made every `check_every` minutes from 10:00 on the close
    of that bar, and fills are at that price plus costs. All positions are
    closed at 16:00. Shares = equity x min(max_leverage, target_vol / last
    `lookback`-day daily vol) / open, rounded down.
    """
    cost = cost or CostModel(per_share=0.0035, spread_bps=1, slippage_bps=1)
    g = minute_grid(bars)
    close, vol = g["close"], g["volume"]
    day_open = g["open"].bfill(axis=1).iloc[:, 0]
    prev_close = close[SESSION_MINUTES].shift(1)

    move = close.div(day_open, axis=0).sub(1).abs()
    sigma = move.rolling(lookback).mean().shift(1)
    upper = sigma.mul(band_mult).add(1).mul(np.maximum(day_open, prev_close), axis=0)
    lower = (1 - sigma.mul(band_mult)).mul(np.minimum(day_open, prev_close), axis=0)
    typical = (g["high"] + g["low"] + close) / 3
    vwap = (typical * vol).cumsum(axis=1) / vol.cumsum(axis=1).replace(0, np.nan)

    daily_ret = close[SESSION_MINUTES].pct_change()
    daily_vol = daily_ret.rolling(lookback).std().shift(1)
    lev = (target_vol / daily_vol).clip(upper=max_leverage)

    marks = [m for m in close.columns if m >= 30 and m % check_every == 0 and m < SESSION_MINUTES]
    rows, equity = [], capital
    for d in close.index:
        if not np.isfinite(sigma.loc[d].iloc[-1]) or not np.isfinite(lev.get(d, np.nan)) or not np.isfinite(prev_close[d]):
            continue
        shares = np.floor(equity * lev[d] / day_open[d])
        pos, entry_px, pnl, fees, n_trades = 0, 0.0, 0.0, 0.0, 0

        def fill(new_pos, px):
            nonlocal pos, entry_px, pnl, fees, n_trades
            if pos:
                pnl += pos * shares * (px - entry_px)
            traded = abs(new_pos - pos) * shares
            if traded:
                fees += (cost.spread_bps / 2 + cost.slippage_bps) / 1e4 * px * traded + cost.fees(px, traded)
                n_trades += 1
            pos, entry_px = new_pos, px

        for m in marks:
            px, ub, lb, vw = close.at[d, m], upper.at[d, m], lower.at[d, m], vwap.at[d, m]
            if not np.isfinite(px):
                continue
            want = pos
            if px > ub:
                want = 1
            elif px < lb:
                want = 0 if long_only else -1
            elif pos == 1 and px < max(ub, vw):
                want = 0
            elif pos == -1 and px > min(lb, vw):
                want = 0
            if want != pos:
                fill(want, px)
        if pos:
            fill(0, close.at[d, SESSION_MINUTES])
        r = (pnl - fees) / equity
        rows.append((d, shares, n_trades, r))
        equity *= 1 + r
    out = pd.DataFrame(rows, columns=["day", "shares", "n_trades", "ret"]).set_index("day")
    out["equity"] = capital * (1 + out["ret"]).cumprod()
    return out


def summary(daily: pd.DataFrame) -> pd.Series:
    """CAGR, volatility, Sharpe (no risk-free rate), max drawdown and the
    share of days with a trade, from a simulator's daily output."""
    r = daily["ret"]
    eq = (1 + r).cumprod()
    years = len(r) / 252
    traded = (daily["n_trades"] > 0) if "n_trades" in daily else (daily["side"] != 0)
    return pd.Series({
        "days": len(r),
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "volatility": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "days_traded": traded.mean(),
    })
