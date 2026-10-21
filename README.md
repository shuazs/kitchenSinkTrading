# kitchenSinkTrading

Research and backtesting of trading strategies for small, risk-tolerant retail
accounts across stocks, options and crypto. The research shortlist that drives
which strategies get built is in the project doc "Trading strategy research shortlist".

## Quick start

```bash
pip install -e ".[dev]"          # add ,sentiment for FinBERT
pytest                            # engine tests on synthetic data
python examples/synthetic_demo.py # offline demo; random walks, no real edge
python examples/run_trend_filter.py   # real data, needs Yahoo + FRED access
```

## How it fits together

```
backtester/
  core.py          MarketData, Strategy base class, CostModel, Backtester
  metrics.py       CAGR, Sharpe, Sortino, drawdown, turnover, cost drag, beta
  research.py      parameter sweeps and walk-forward tests
  memecoin_calls.py  pump.fun bonding-curve fills for following meme coin calls
  data/
    yahoo.py       yfinance bars, today's option chains, headlines archive
    fred.py        FRED series (T-bill rate, yield curve, credit spreads)
    edgar.py       SEC company facts, earnings surprise (SUE), Form 4 insider trades
    sentiment.py   FinBERT headline scores, aligned to when news was public
    memecoins.py   pump.fun call feed and launch outcomes (Hugging Face datasets)
    synthetic.py   random-walk prices for tests
  strategies/      one file per plug-in strategy
```

## Writing a strategy

A strategy returns target weights (fraction of equity per symbol). Rows can be
sparse: emit them only on rebalance days and the engine carries them forward.

```python
from backtester import Strategy

class GoldenCross(Strategy):
    params = {"fast": 50, "slow": 200}

    def generate_targets(self, data):
        close = data.field("close")
        on = close.rolling(self.p["fast"]).mean() > close.rolling(self.p["slow"]).mean()
        return on.astype(float).div(on.sum(axis=1).clip(lower=1), axis=0)
```

## What the engine does to keep results honest

- **No look-ahead.** Weights computed from the close of day t trade at the open of day t+1.
  EDGAR data is keyed to filing date, news to publication time.
- **Costs on every fill.** Commission, per-share and percentage fees, half the bid-ask
  spread and slippage. Presets in `COSTS`: `etf`, `large_cap`, `small_cap`, `crypto`.
- **Small-account realism.** Whole-share mode (`fractional=False`), minimum trade size,
  cash checks so buys never overdraw, leverage cap, optional shorting, interest on cash.
- **Overfitting checks.** `sweep` records how many variants you tried; `walk_forward`
  re-picks parameters on each training window and only reports the following unseen window.

Known limits: yfinance only lists securities that exist today (survivorship bias),
intraday history is short (1-minute bars for about 7 days, hourly for about 2 years),
there is no free options price history, and Yahoo returns only recent headlines, so
sentiment needs to be collected going forward. Taxes are not modelled.
