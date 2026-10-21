"""Core backtesting engine.

A strategy turns market data into target portfolio weights. Weights decided
with data up to the close of day t are traded at the open of day t+1, so a
strategy can never trade on a price it could not have seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """Daily OHLCV for several symbols plus optional side data.

    `bars` maps symbol -> DataFrame with columns open, high, low, close, volume
    (lower case), indexed by date. `extra` holds anything else a strategy needs,
    such as FRED series, EDGAR filings or sentiment scores, keyed by name.
    """

    bars: dict[str, pd.DataFrame]
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = None
        for df in self.bars.values():
            idx = df.index if idx is None else idx.union(df.index)
        self.index = idx.sort_values()

    @property
    def symbols(self) -> list[str]:
        return list(self.bars)

    def field(self, name: str) -> pd.DataFrame:
        """One field (e.g. 'close') for all symbols as a date x symbol frame."""
        return pd.DataFrame({s: df[name] for s, df in self.bars.items()}).reindex(self.index)

    def slice(self, start=None, end=None) -> "MarketData":
        return MarketData({s: df.loc[start:end] for s, df in self.bars.items()}, self.extra)


class Strategy:
    """Base class for plug-in strategies.

    Subclasses set `params` defaults and implement `generate_targets`, which
    returns a date x symbol DataFrame of target weights (fraction of equity,
    negative for shorts). Rows may be sparse; the engine forward-fills so a
    strategy only needs to emit rows on rebalance days. NaN means "no change".
    """

    name = "strategy"
    params: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.params)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown params {sorted(unknown)}")
        self.p = {**self.params, **params}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.p.items())
        return f"{type(self).__name__}({args})"


@dataclass
class CostModel:
    """Trading frictions charged on every fill.

    commission: flat fee per order, in dollars.
    per_share: fee per share or coin unit, in dollars.
    pct_fee: percentage fee on notional, e.g. 0.001 for a 0.1% crypto taker fee.
    spread_bps: full bid-ask spread in basis points; half is paid per fill.
    slippage_bps: extra adverse price movement per fill, in basis points.
    """

    commission: float = 0.0
    per_share: float = 0.0
    pct_fee: float = 0.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0

    def fill_price(self, price: float, qty: float) -> float:
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 1e4
        return price * (1 + adverse) if qty > 0 else price * (1 - adverse)

    def fees(self, price: float, qty: float) -> float:
        return self.commission + self.per_share * abs(qty) + self.pct_fee * abs(qty) * price


# Reasonable defaults for a small retail account. Override per run.
COSTS = {
    "etf": CostModel(spread_bps=2, slippage_bps=2),
    "large_cap": CostModel(spread_bps=5, slippage_bps=5),
    "small_cap": CostModel(spread_bps=40, slippage_bps=20),
    "crypto": CostModel(pct_fee=0.004, spread_bps=10, slippage_bps=10),
}


@dataclass
class Result:
    equity: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.Series
    initial_capital: float
    strategy: str

    def stats(self, benchmark: pd.Series | None = None, rf: float | pd.Series = 0.0) -> pd.Series:
        from .metrics import summary

        return summary(self, benchmark=benchmark, rf=rf)


class Backtester:
    """Runs a strategy over MarketData with next-open execution and costs.

    initial_capital: starting cash in dollars.
    costs: a CostModel or a dict of symbol -> CostModel.
    fractional: allow fractional units (true for crypto and most brokers' stocks).
    min_trade_value: skip rebalances smaller than this many dollars per symbol.
    allow_short: if False, negative target weights are clipped to zero.
    max_leverage: gross exposure cap as a multiple of equity.
    cash_rate: annual rate earned on idle cash, a float or a daily Series
        (e.g. FRED DTB3 / 100). Negative cash is charged the same rate.
    rebalance: "on_change" trades only when the target weights change, so
        positions drift with prices in between; "daily" trades back to the
        targets every bar, which pays costs on drift.
    """

    def __init__(
        self,
        initial_capital: float = 5_000.0,
        costs: CostModel | dict[str, CostModel] | None = None,
        fractional: bool = True,
        min_trade_value: float = 1.0,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        cash_rate: float | pd.Series = 0.0,
        rebalance: str = "on_change",
    ) -> None:
        self.initial_capital = initial_capital
        self.costs = costs if costs is not None else COSTS["etf"]
        self.fractional = fractional
        self.min_trade_value = min_trade_value
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.cash_rate = cash_rate
        if rebalance not in ("on_change", "daily"):
            raise ValueError("rebalance must be 'on_change' or 'daily'")
        self.rebalance = rebalance

    def _cost_model(self, symbol: str) -> CostModel:
        if isinstance(self.costs, dict):
            return self.costs.get(symbol, COSTS["etf"])
        return self.costs

    def _daily_cash_rate(self, index: pd.DatetimeIndex) -> pd.Series:
        if isinstance(self.cash_rate, pd.Series):
            annual = self.cash_rate.reindex(index).ffill().fillna(0.0)
        else:
            annual = pd.Series(self.cash_rate, index=index)
        days = index.to_series().diff().dt.days.fillna(1).clip(lower=1)
        return (1 + annual) ** (days / 365.0) - 1

    def run(self, strategy: Strategy, data: MarketData, start=None, end=None) -> Result:
        targets = strategy.generate_targets(data)
        idx = data.index
        if start is not None or end is not None:
            idx = idx[(idx >= pd.Timestamp(start or idx[0])) & (idx <= pd.Timestamp(end or idx[-1]))]
        symbols = data.symbols
        targets = targets.reindex(columns=symbols).reindex(data.index).ffill().reindex(idx)
        if not self.allow_short:
            targets = targets.clip(lower=0)
        if self.rebalance == "on_change":
            filled = targets.fillna(0.0)
            changed = filled.ne(filled.shift()).any(axis=1)
            changed.iloc[0] = targets.iloc[0].notna().any()
            targets = targets.where(changed, np.nan)
        # Signal formed at close t executes at open t+1.
        orders = targets.shift(1)

        opens = data.field("open").reindex(idx).to_numpy(float)
        closes = data.field("close").reindex(idx).ffill().to_numpy(float)
        orders_a = orders.to_numpy(float)
        cash_r = self._daily_cash_rate(idx).to_numpy(float)
        models = [self._cost_model(s) for s in symbols]
        n_days, n_sym = closes.shape

        cash = self.initial_capital
        qty = np.zeros(n_sym)
        eq_hist = np.empty(n_days)
        pos_hist = np.empty((n_days, n_sym))
        w_hist = np.empty((n_days, n_sym))
        cost_hist = np.zeros(n_days)
        trades = []
        last_px = np.full(n_sym, np.nan)
        pending, waiting = None, np.zeros(n_sym, dtype=bool)

        for i in range(n_days):
            if i > 0:
                cash *= 1 + cash_r[i]
            o = opens[i]
            mark = np.where(np.isnan(o), last_px, o)
            equity_open = cash + np.nansum(qty * mark)

            if not np.isnan(orders_a[i]).all():
                pending = np.nan_to_num(orders_a[i])
                waiting = np.ones(n_sym, dtype=bool)
            if pending is not None and equity_open > 0:
                tgt = pending
                gross = np.abs(tgt).sum()
                if gross > self.max_leverage:
                    tgt = tgt * self.max_leverage / gross
                deltas = []
                for j in range(n_sym):
                    px = o[j]
                    if not waiting[j] or np.isnan(px) or px <= 0:
                        continue
                    # A symbol with no bar today (weekend, holiday, not yet
                    # listed) keeps its order until it next trades.
                    waiting[j] = False
                    want = tgt[j] * equity_open / px
                    if not self.fractional:
                        want = np.trunc(want)
                    d = want - qty[j]
                    if abs(d * px) >= self.min_trade_value:
                        deltas.append((d, j))
                # Sells first so their cash funds the buys.
                for d, j in sorted(deltas):
                    cm, px = models[j], o[j]
                    fp = cm.fill_price(px, d)
                    fee = cm.fees(fp, d)
                    if d > 0:
                        # Never let buys push cash below what leverage allows.
                        floor = -(self.max_leverage - 1) * equity_open
                        affordable = (cash - floor - fee) / fp
                        if affordable < d:
                            d = max(affordable, 0.0)
                            if not self.fractional:
                                d = np.floor(d)
                            if d * fp < self.min_trade_value:
                                continue
                            fee = cm.fees(fp, d)
                    cash -= d * fp + fee
                    qty[j] += d
                    slip = abs(d) * abs(fp - px)
                    cost_hist[i] += fee + slip
                    trades.append((idx[i], symbols[j], d, px, fp, fee, slip))

            if pending is not None and not waiting.any():
                pending = None
            c = closes[i]
            last_px = np.where(np.isnan(c), last_px, c)
            value = qty * np.nan_to_num(c)
            equity = cash + value.sum()
            eq_hist[i] = equity
            pos_hist[i] = qty
            w_hist[i] = value / equity if equity else 0.0

        trades_df = pd.DataFrame(
            trades, columns=["date", "symbol", "qty", "ref_price", "fill_price", "fees", "slippage"]
        )
        return Result(
            equity=pd.Series(eq_hist, index=idx, name="equity"),
            positions=pd.DataFrame(pos_hist, index=idx, columns=symbols),
            weights=pd.DataFrame(w_hist, index=idx, columns=symbols),
            trades=trades_df,
            costs=pd.Series(cost_hist, index=idx, name="costs"),
            initial_capital=self.initial_capital,
            strategy=repr(strategy),
        )
