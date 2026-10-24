# kitchenSinkTrading

Research and a plug-in backtester for trading strategies on a $5,000 retail
account: stocks, ETFs, options, crypto and US prediction markets (Kalshi,
Polymarket US). Every result uses realistic costs, no look-ahead (signal at
the close, trade at the next open) and reports walk-forward or holdout
numbers next to in-sample ones.

## Where things are

| Folder | What it holds | Start with |
| --- | --- | --- |
| `backtester/` | the engine (`core.py`), metrics, research helpers, standalone simulators, `strategies/`, `data/` loaders, `ml/` | [docs/guide/architecture.md](docs/guide/architecture.md) |
| `examples/` | one script per study: runs a backtest and prints the tables behind each verdict | [docs/guide/examples.md](docs/guide/examples.md) |
| `paper_loggers/` | log-only live forward tests of the paper-trading candidates, plus the keys template | [paper_loggers/README.md](paper_loggers/README.md) |
| `tests/` | offline tests on synthetic data (`pytest`) | [docs/guide/testing.md](docs/guide/testing.md) |
| `docs/` | the guide (short reference) and the research index with summaries | [docs/README.md](docs/README.md) |

The guide, one page per need:

- [strategies.md](docs/guide/strategies.md): every strategy, its params, result and verdict (keeper / weak / no edge)
- [research-modules.md](docs/guide/research-modules.md): simulators outside `strategies/` (intraday, Kalshi, 0DTE, pairs, ML)
- [data.md](docs/guide/data.md): loaders, env vars, shared datasets, hosts blocked from the cloud
- [protocol.md](docs/guide/protocol.md): pass rules for calling a result robust out of sample
- [paper-trading.md](docs/guide/paper-trading.md): candidates on paper, how each is logged, when it graduates

## Quick start

```bash
pip install -e ".[dev]"               # extras: ml, sentiment, polymarket
pytest                                 # offline, synthetic data
python examples/synthetic_demo.py      # offline demo; random walks, no real edge
python examples/run_trend_filter.py    # real data (Yahoo + FRED)
```

## Where the research stands

- **Keepers** (run in the $5k fund plan, `examples/run_fund_plan.py`): UPRO
  vol target, multi-asset ETF trend, pension month-end, BTC+ETH trend. The
  Balanced mix roughly matches SPY's return at half the drawdown.
- **On paper** (no real money until the live log keeps half the backtest
  edge): Treasury auction cycle (the one strict pass), ETF breakout +
  meta-label filter, and the near misses in
  [paper-trading.md](docs/guide/paper-trading.md).
- **No edge**: most ideas tested so far, from sentiment and insider buying to
  intraday rules and prediction-market leads; check
  [strategies.md](docs/guide/strategies.md) before starting a new one.

Known limits: yfinance lists only securities that exist today (survivorship
bias), there is no free option price history (options are modelled), and
taxes are not modelled.
