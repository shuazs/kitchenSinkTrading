# Architecture

`pip install -e ".[dev]"` (extras: `sentiment`, `ml`, `polymarket`).
Package `backtester/`; public API in `backtester/__init__.py`.

## Engine (`backtester/core.py`, ~280 lines)

- `MarketData(bars: dict[symbol -> OHLCV df], extra: dict)` — lower-case
  columns, date index. `.field("close")` gives a date x symbol frame,
  `.slice(start, end)`, `.symbols`. `extra` carries side data (FRED series,
  insider events, Kalshi features...) keyed by name; strategies read it via a
  `*_key` param.
- `Strategy` — subclass, set `params = {...}` defaults (unknown kwargs raise),
  read them as `self.p[...]`, implement `generate_targets(data)` returning
  date x symbol **target weights** (fraction of equity, negative = short).
  Rows may be sparse; NaN = no change; the engine forward-fills.
- `Backtester(initial_capital=5000, costs=COSTS["etf"] | {sym: CostModel},
  fractional=True, min_trade_value=1, allow_short=False, max_leverage=1.0,
  cash_rate=0.0 | daily Series, rebalance="on_change" | "daily")`.
  `.run(strategy, data, start, end) -> Result`.
- Execution: weights from close t trade at open t+1. `on_change` trades only
  when targets change (positions drift between), `daily` re-trades to target.
- `CostModel(commission, per_share, pct_fee, spread_bps, slippage_bps)`; pays
  half the spread + slippage per fill. `COSTS` presets: `etf` 2+2 bp,
  `large_cap` 5+5, `small_cap` 40+20, `crypto` 0.4% fee + 10+10 bp.
- `Result`: `equity, positions, weights, trades, costs, initial_capital,
  strategy`; `.stats(benchmark, rf)` -> `metrics.summary`.

## Analysis helpers

- `metrics.py`: `summary()` (CAGR, vol, Sharpe, Sortino, max DD, turnover,
  cost drag, exposure, beta vs benchmark), `drawdown()`.
- `research.py`: `sweep(cls, grid, data, bt)` (one row per combination; the
  row count is the number of trials), `walk_forward(cls, grid, data, bt,
  train_years, test_years, metric)` (re-pick params per window, report only
  unseen windows), `event_study(data, events, horizons)`,
  `summarize_events()`.
- `portfolio.py`: `sleeve_returns({name: Result})`, `combine(returns, weights, rebalance, band, cost_bps, cash_rate, dd_brake)` -> `Result`: a fund holding several strategies as sub-accounts.
- `strategies/indicators.py`: `rsi`, `month_ends`.

## Standalone simulators (not `Strategy` subclasses)

Short options and tick/minute-level work can't be expressed as daily weights,
so these have their own loops: `strategies/wheel.py` (`Wheel`),
`strategies/spreads.py` (`CreditSpreads`), `strategies/basis.py`
(`BasisCarry`), and every module in [research-modules.md](research-modules.md).
Options are priced with Black-Scholes off VIX (no free option history).

## Minimal example

```python
from backtester import Backtester, COSTS
from backtester.data.yahoo import load_market
from backtester.strategies import TrendFilter
data = load_market(["SSO", "SPY", "SHY"], start="2007-01-01")
res = Backtester(5_000, costs=COSTS["etf"]).run(TrendFilter(risk_on="SSO", signal="SPY", risk_off="SHY"), data)
print(res.stats())
```

Known limits: yfinance survivorship bias, no free option or 0DTE history,
Yahoo headlines only recent, taxes not modelled.
