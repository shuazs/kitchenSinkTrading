# kitchenSinkTrading

[![tests](https://github.com/shuazs/kitchenSinkTrading/actions/workflows/tests.yml/badge.svg)](https://github.com/shuazs/kitchenSinkTrading/actions/workflows/tests.yml)

A plug-in backtesting engine and a systematic research program for trading
strategies on a $5,000 retail account: stocks, ETFs, options, crypto and US
prediction markets (Kalshi, Polymarket US).

The point of the project is less "find a strategy that works" than "find out
honestly which ones do". Every result uses realistic costs, no look-ahead
(signal at the close, trade at the next open) and is reported out of sample
next to the in-sample number. Most ideas failed, and the repo keeps the
failures alongside the few that held up.

## At a glance

| | |
| --- | --- |
| Code | ~24k lines of Python: event-driven engine, 30 strategy files, 20+ standalone simulators, data loaders for 15+ free sources |
| Research | 40+ strategies from the literature and forums, then ~100 pre-registered hypotheses in 12 batches under one written pass/fail protocol |
| Validation | walk-forward re-fitting, a locked Jul 2024+ holdout, placebo/shuffled-date tests, deflated Sharpe ratio and probability of backtest overfitting (CSCV) |
| Outcome | 1 hypothesis passed every rule (Treasury auction cycle); 4 keepers combine into a $5k fund plan; dozens of documented "no edge" results |
| Tests | 200+ offline tests on synthetic data, ~20 s, run in CI |

## The engine

```python
from backtester import Backtester, COSTS
from backtester.data.yahoo import load_market
from backtester.strategies import TrendFilter

data = load_market(["SSO", "SPY", "SHY"], start="2007-01-01")
res = Backtester(5_000, costs=COSTS["etf"]).run(
    TrendFilter(risk_on="SSO", signal="SPY", risk_off="SHY"), data)
print(res.stats())   # CAGR, vol, Sharpe, Sortino, max DD, turnover, cost drag, beta
```

- **Plug-in strategies.** A strategy is a class with default `params` and one
  method, `generate_targets(data)`, that returns target portfolio weights per
  date. The engine handles fills, costs, cash and leverage, so a new idea is
  usually 30-80 lines.
- **No look-ahead by construction.** Weights computed from the close of day t
  trade at the open of t+1; side data (SEC filings, news, auction dates) is
  keyed to the moment it became public. Tests check this for every strategy.
- **Small-account realism.** Cost presets per asset class (ETF, large and
  small cap, crypto), spread and slippage per fill, optional whole-share
  mode, minimum trade size, cash checks so buys never overdraw, leverage
  caps, shorting and interest on cash.
- **Research tooling.** Parameter sweeps that count every trial, walk-forward
  optimisation, event studies, a portfolio combiner that runs strategies as
  sub-accounts with rebalancing bands, and an ML layer (purged walk-forward
  folds, meta-labeling, HAR volatility forecasts, a model contest scored by
  deflated Sharpe).
- **Simulators the daily engine can't express**: short options (wheel, credit
  spreads priced off a skew fitted to real SPY quotes), 1-minute intraday
  rules, Kalshi and Polymarket order-level replays with exact fee schedules,
  pairs and stat-arb, and event windows.

Details: [docs/guide/architecture.md](docs/guide/architecture.md).

## Key results

Backtests at $5,000 with costs; CAGR / max drawdown. Full catalog with every
parameter and verdict: [docs/guide/strategies.md](docs/guide/strategies.md).

**Held up**

| Strategy | Result |
| --- | --- |
| Treasury auction cycle (the one strict pass) | Long bonds cheapen before 10y/30y auctions: TLT -38 bp per cycle (t -2.9), negative in 2008-16, 2017-24 and the untouched Jul 2024+ holdout, in every bond ETF tested, and at the 0.1th percentile of 1,000 random windows. A TLT/TBF sleeve: 9.8%/-25% vs TLT 2.3%/-48%. |
| $5k fund plan (four keepers combined) | 30% UPRO vol target, 35% multi-asset trend, 15% pension month-end, 20% BTC+ETH trend: 15.2%/-16% vs SPY 15.3%/-34% (Nov 2018-Oct 2026), Sharpe 1.00 vs 0.69. |
| UPRO volatility target | 17.4%/-27%. |
| BTC+ETH ensemble trend | 28%/-32% at 0.4% fees (2018-26). |

**Didn't hold up** (a sample; the rest are in the catalog)

- 3,448 intraday rule variants on 1-minute S&P, Nasdaq, BTC and ETH bars:
  about 1 bp per trade before costs, so nothing after costs.
- A search over ~3 million combinations of the repo's own signals and
  sleeves: no searched combination beat its benchmark out of sample
  (probability of overfitting 0.7-0.9); plain equal weight was the robust mix.
- FinBERT news and Reddit sentiment, insider cluster buying, Congress-trade
  copying, futures positioning (COT), 13F cloning, options credit spreads,
  crypto pairs, and taker strategies on Kalshi and Polymarket all landed
  below costs or failed the holdout.
- One promising prediction-market result (+5.6c/share) turned out to be a
  bug that silently dropped losing trades; the fix and the retraction are
  documented ([research-modules.md](docs/guide/research-modules.md)).

The rules a result must pass to be called robust are written down before
each test: [docs/guide/protocol.md](docs/guide/protocol.md). Near misses are
forward-tested on paper with log-only loggers before any money goes in:
[docs/guide/paper-trading.md](docs/guide/paper-trading.md).

## Quick start

```bash
pip install -e ".[dev]"                # extras: ml, sentiment, polymarket
pytest                                 # offline, synthetic data, ~20 s
python examples/synthetic_demo.py      # offline demo; random walks, no real edge
python examples/run_auction_cycle.py   # real data (Yahoo + Treasury auction dates)
python examples/run_fund_plan.py       # the combined $5k fund
```

Real-data scripts download from free sources (Yahoo Finance, FRED, SEC
EDGAR, Hugging Face dataset mirrors). SEC scripts need a
`SEC_USER_AGENT` env var; see [docs/guide/data.md](docs/guide/data.md).

## Repository layout

| Folder | What it holds |
| --- | --- |
| `backtester/` | engine (`core.py`), metrics, research helpers, portfolio combiner, standalone simulators, `strategies/`, `data/` loaders, `ml/` |
| `examples/` | one script per study; each prints the tables behind its verdict ([examples.md](docs/guide/examples.md)) |
| `paper_loggers/` | log-only forward tests of the paper-trading candidates ([README](paper_loggers/README.md)) |
| `tests/` | offline tests ([testing.md](docs/guide/testing.md)) |
| `docs/` | the guide (short reference pages) and the research index with summaries ([docs/README.md](docs/README.md)) |

## Known limits

- yfinance lists only securities that exist today, so single-stock results
  carry survivorship bias (flagged wherever it matters).
- There is no free option price history, so option strategies use modelled
  prices.
- Taxes are not modelled. The Jul 2024+ holdout was seen during development
  for strategies found early on; only paper trading is truly out of sample.
- Nothing here is investment advice.

## How it was built

The research was run as a series of AI-assisted research threads (Claude
Code) that I directed: I set the goals, constraints and validation protocol,
chose which ideas to pursue, and reviewed and merged the work in 40+ pull
requests. `CLAUDE.md` and `docs/guide/` are written so that each new thread
can orient from short docs instead of re-reading the codebase.
