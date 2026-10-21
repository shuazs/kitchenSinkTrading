# kitchenSinkTrading

Research and backtesting of trading strategies for small, risk-tolerant retail
accounts across stocks, options and crypto. The research write-ups behind each
strategy, with links and short summaries, are in [docs/README.md](docs/README.md).

## Quick start

```bash
pip install -e ".[dev]"          # add ,sentiment for FinBERT, ,ml for the model contest
pytest                            # engine tests on synthetic data
python examples/synthetic_demo.py # offline demo; random walks, no real edge
python examples/run_trend_filter.py   # real data, needs Yahoo + FRED access
```

## How it fits together

```
backtester/
  core.py          MarketData, Strategy base class, CostModel, Backtester
  metrics.py       CAGR, Sharpe, Sortino, drawdown, turnover, cost drag, beta
  research.py      parameter sweeps, walk-forward tests, event studies
  intraday.py      day-trading simulators on minute bars (SPY/QQQ intraday momentum)
  intraday_lab.py  intraday rule families (ORB, VWAP, gaps, noise area, crypto hours)
                   on one minute-grid engine, with walk-forward selection
  memecoin_calls.py  pump.fun bonding-curve fills for following meme coin calls
  jump_trades.py   trade an asset after a sudden prediction-market move (hourly bars)
  ml/              signal combination: weekly feature panel from the strategies,
                   purged walk-forward folds, equal weight / ridge / LightGBM /
                   MLP / LSTM models, Deflated Sharpe
  data/
    yahoo.py       yfinance bars, today's option chains, headlines archive
    fred.py        FRED series (T-bill rate, yield curve, credit spreads)
    fx.py          currency total-return series (spot + interbank rates) for carry
    edgar.py       SEC company facts, earnings surprise (SUE), Form 4 insider trades,
                   earnings dates (8-K Item 2.02)
    alpaca.py      minute bars from Alpaca (free account) for intraday tests
    minute_mirrors.py  free 1-minute S&P, MES, NQ/QQQ and BTC bars 2021-2026 (Hugging Face)
    thirteenf.py   hedge fund 13F holdings (SEC bulk data sets), CUSIP to ticker
    activists.py   Schedule 13D filings by known activists (EDGAR form index)
    sp500.py       S&P 500 membership history and dated adds and deletions
    sentiment.py   FinBERT headline scores, aligned to when news was public
    kalshi.py      weekly Kalshi prediction-market features (CPI, payrolls, Fed, recession)
    trump_posts.py Trump's tweets (2017-2021) and Truth Social posts, daily topic counts
    gov_stakes.py  US government equity stakes in listed companies, by first report
    memecoins.py   pump.fun call feed and launch outcomes (Hugging Face datasets)
    polymarket.py  Polymarket fill history (Hugging Face) and live odds, Iran escalation index
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

## Options spreads

`CreditSpreads` (put credit spreads, call spreads, iron condors) and `Wheel` are
standalone daily simulators, since target weights cannot express short options.
Both price options with Black-Scholes off VIX because there is no free option
history. `CreditSpreads` adds the VIX term structure and a put skew fitted to
real SPY quotes (`examples/fit_spy_skew.py`), and runs each trade at today's SPY
price so dollar costs stay realistic. `examples/run_spreads.py` compares costs,
management rules, sizing and skew assumptions on a $5,000 account. The skew is
the assumption that decides the answer, so real chain data is the next check.

## Intraday strategies

`backtester/intraday.py` simulates two intraday momentum rules on SPY or QQQ:
the first-half-hour rule (Gao, Han, Li and Zhou 2018) and the noise-area
breakout (Zarattini, Aziz and Barbon 2024). `backtester/intraday_lab.py`
adds rule families with parameter grids on one engine (fills one minute
after the signal, costs per unit traded, flat at the close): end-of-day
momentum, noise area, opening range breakout, VWAP trend, VWAP reversion,
gap fade/follow, crypto hour-of-day, and the overnight/session split.

Minute bars come free from Hugging Face mirrors (`data/minute_mirrors.py`):
a SPY stand-in from the cash S&P 500 index (Jun 2021 on), NQ futures
(2024 to Jul 2025) and Binance BTC (Jun 2021 on). The QQQ stand-in splices in
QQQ itself from Mar 2025, and ETH covers Dec 2024 to Nov 2025; those two are
read from `KS_MINUTE_DIR` (the project's shared minute-data folder). With Alpaca keys
(`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, free paper account)
`run_intraday_momentum.py` uses real SPY and QQQ bars from 2016 instead.

    python examples/run_intraday_momentum.py      # the two paper rules
    python examples/run_intraday_strategies.py    # all families, walk-forward, 2021-2025

Day trading a margin account under $25,000 still runs into the pattern-day-trader
rule at most brokers: the SEC approved removing it (effective June 4, 2026), but
brokers have until October 2027 to drop it. Until yours does, the realistic
$5,000 setup is long-only in a cash account.

## Combining signals with machine learning

`backtester/ml/` turns the strategies' signals into one weekly feature panel
(trend, breakout, cross-sectional momentum, volatility, calendar, VIX term
structure, yield curve) and runs a contest: equal weight, ridge, LightGBM, a
small MLP and an LSTM Deep Momentum Network, all on the same purged
walk-forward folds, sizing (`SignalBlend`) and costs, with a locked holdout
from July 2024. `python examples/run_signal_blend.py` prints the comparison
and the pass criteria from the design doc (needs `pip install -e ".[ml]"`).
`examples/run_signal_blend_kalshi.py` reruns it with nine weekly Kalshi
prediction-market features added (`data/kalshi.py`; the daily file comes from
the Kalshi research and is read from disk).

`examples/run_ml_research.py` tries four other framings on the same signals:
meta-labeling breakout trades (`ml/metalabel.py`, triple-barrier labels in
`ml/labels.py`), a statistical jump model for regimes (`ml/regime.py`), HAR
and LightGBM volatility forecasts for vol-target sizing (`ml/vol_forecast.py`)
and monthly learning-to-rank on the S&P 100 (`ml/rank.py`). Only the ETF
meta-label filter beat its controls; see docs/README.md.

Known limits: yfinance only lists securities that exist today (survivorship bias),
yfinance intraday history is short (free 1-minute mirrors cover 2021-2026),
there is no free options price history, and Yahoo returns only recent headlines, so
sentiment needs to be collected going forward. Taxes are not modelled.
