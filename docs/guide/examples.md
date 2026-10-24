# Example scripts

`examples/*.py`; run from the repo root, e.g. `python examples/run_trend_filter.py`.
Most need network (Yahoo/FRED/SEC/HF) and print result tables. Strategy
verdicts are in [strategies.md](strategies.md).

## Engine demo

- `synthetic_demo.py`: Runs every strategy on random-walk prices to show the workflow offline.

## Core strategies

- `run_trend_filter.py`: Leveraged trend filter vs buy and hold, on real data.
- `run_breakout.py`: Crypto N-day-high breakout vs buy and hold, on real data.
- `run_robustness.py`: Robustness checks: crypto breakout across fee levels and coins, and walk-forward tests for the breakout and the leveraged trend filter.
- `run_mean_reversion.py`: RSI(2) pullback buying on index ETFs, on real data.
- `run_dual_momentum.py`: Dual momentum (Antonacci GEM and a wider ETF version), on real data.
- `run_wheel.py`: Wheel (cash-secured puts, then covered calls) vs buy and hold, $5,000 account.
- `run_spreads.py`: SPY put credit spreads and iron condors on modeled prices, $5,000 account.
- `fit_spy_skew.py`: Fit the spread pricer's skew to today's SPY option chain.
- `run_insider.py`: Insider cluster buying in small caps vs buy and hold, on real data.
- `run_opportunistic_insider.py`: Opportunistic vs routine insider purchases (Cohen, Malloy and Pomorski 2012), as an event study and as a filter on the insider strategies.

## More ETF/stock ideas

- `run_vol_target.py`: Volatility-targeted UPRO vs buy and hold and the 200-day trend filter.
- `run_vix_term_structure.py`: Short volatility (SVXY) only while VIX is below 3-month VIX, on real data.
- `run_congress_trades.py`: Congress purchase disclosures: event study and $5k copy portfolios by subset (all, $50k+, clusters, party, leaders, fast filers); `[out_dir]` saves CSVs.
- `run_cot_positioning.py`: CFTC COT signals (flow reversal, hedging pressure, commercial index) on 20 futures ETFs: periods, walk-forward, cost sensitivity.
- `run_sector_rotation.py`: Sector SPDR momentum rotation vs SPY, on real data.
- `run_turn_of_month.py`: Turn-of-month on SPY, QQQ and 3x ETFs vs buy and hold, on real data.
- `run_seasonality.py`: Same-calendar-month seasonality on sector and country ETFs vs SPY and the momentum sector rotation, on real data.
- `run_earnings_premium.py`: Earnings announcement premium on large caps, on real data.
- `run_hedged_momentum.py`: S&P 100 momentum (top 10/20 by 12-1 month return), with and without an SH hedge.

## Fund-style

- `run_multi_asset_trend.py`: Multi-asset ETF trend following (a retail managed-futures fund) and the stocks-plus-trend return stack, against DBMF/KMLM and SPY.
- `run_risk_parity.py`: Leveraged risk parity with 3x ETFs (a retail All Weather) against 60/40, the Reddit HFEA mix (55% UPRO / 45% TMF) and SPY.
- `run_pension_rebalance.py`: Front-running pension funds' month-end stock/bond rebalancing.
- `run_hedge_fund_clone.py`: Cloning concentrated hedge funds' best ideas from 13F filings.
- `run_activists.py`: Following activist hedge funds into their targets on the day a new Schedule 13D is filed.
- `run_index_changes.py`: S&P 500 additions and deletions: the index-rebalancing trade event desks run, and what is left of it for a small account.
- `run_fx_carry.py`: G10 + MXN currency carry on a $5,000 account, as micro FX futures would trade it.
- `run_basis_carry.py`: Crypto cash-and-carry (long spot, short the CME front month) on $5,000.
- `run_fundamental_allocator.py`: Long-term fundamental portfolio manager (S&P 500 picks on SEC fundamentals + intl + small caps, macro switch) vs SPY: design, holdout, walk-forward, survivorship check, big-tech (QQQ) tilts. Needs `SEC_USER_AGENT`.

## Fund

- `run_fund_plan.py`: Re-runs the keepers and combines them into the $5,000 fund (Balanced / Growth) with rebalancing and drawdown-brake variants; `[out_dir]` saves equity CSVs.
- `fund_signals.py`: Today's dollar target per symbol for the fund plan (`[equity] [balanced|growth]`), no orders.

## Crypto

- `run_crypto_strategies.py`: Crypto momentum basket, volatility-targeted breakout and a capped meme coin sleeve, against the plain BTC+ETH breakout and buy and hold.
- `run_ensemble_trend.py`: Ensemble Donchian trend on a rotating top-N crypto universe vs BTC and the single-lookback breakouts, on real data.
- `run_memecoin_calls.py`: Would following public pump.fun meme coin calls make money?.

## Trump / events

- `run_sentiment_overlays.py`: Market sentiment (mood index, VIX, credit, Michigan, crypto funding) as forecaster and overlay on the keepers; `[out_dir]` saves CSVs.
- `run_trump_overlay.py`: Cut leverage after a burst of Trump posts on tariffs or markets.
- `run_taco_dip.py`: The "TACO" trade: buy index ETFs after a tariff-scare selloff.
- `run_gov_stakes.py`: Buy companies after the first report of a US government equity stake.

## ML

- `run_signal_blend.py`: Can a neural net combine our signals better than a linear blend?.
- `run_signal_blend_kalshi.py`: Do Kalshi prediction-market features help the signal combiner?.
- `run_signal_combos.py`: Every strategy as a signal or sleeve, every subset combined (~3M trials), walk-forward + holdout + deflated Sharpe + PBO; writes CSVs to `/mnt/project-files/signal-combos/` (~15 min).
- `run_ml_research.py`: Four machine-learning framings that use the repo's research as inputs, each tested walk-forward with costs on $5,000.

## Intraday

- `run_intraday_momentum.py`: Intraday momentum on SPY and QQQ: first-half-hour rule and noise-area breakout.
- `run_intraday_strategies.py`: Intraday rule families on free 1-minute bars, with walk-forward selection.

## Prediction markets

- `run_iran_oil.py`: Polymarket Iran-war odds as a leading signal for crude oil.
- `watch_polymarket_iran.py`: Paper-trade the Iran-to-oil rule live.
- `watch_kalshi_polymarket_us.py`: Log Kalshi and Polymarket US game books side by side, no orders, on Josh's machine; `--once` lists pairs, `--report` replays the Kalshi -> Polymarket rule net of the US fee.
- `run_shortterm_combo.py`: Combine short-horizon signals for SPY/QQQ and test them as 0DTE option trades.
- `run_kalshi_minute_leadlag.py`: Do Kalshi's hourly S&P / Nasdaq / BTC ladders lead the underlying by minutes, and is that enough to trade 0DTE options?.
- `run_crypto_pairs.py`: Crypto pairs: BTC/ETH daily, 19-coin rolling pairs + walk-forward, intraday 5m/15m/1h; writes CSVs to `crypto-pairs/results/`.
- `run_prediction_market_pairs.py`: Prediction markets vs the asset they track, traded as pairs.
- `run_kalshi_catchup.py`: Buy Kalshi hourly index/crypto contracts after a sharp move in the underlying.
- `run_kalshi_crowd.py`: Kalshi crowd bias after big moves: calibration, taker lean and fade/follow backtests.
- `run_news_sentiment.py`: FinBERT stock-news buckets, WSB mention-spike fade, Reddit mood -> Kalshi S&P/BTC (reads `/mnt/project-files/finbert-sentiment/`).
- `watch_wsb_spikes.py`: log-only live WSB mention-spike fade (paper shorts at the open, Alpaca borrow check); runs on Josh's machine.
- `run_event_markets.py`: Non-financial Kalshi/Polymarket markets: taker calibration, sum-to-$1 and two-venue arbitrage, following Kalshi on Polymarket sports; `--families` for the per-market-family tables.
- `run_scheduled_flows.py`: alpha hunt batch 3, nine scheduled-flow event windows (roll, Russell, auctions, tax, BTC expiry, MSCI, coupons, VIX expiry, quarter-end) by period with placebos.
- `run_alpha_batch8.py`: alpha hunt batch 8 (odd-lot tender offers, CEF activist 13D and rights offerings, MSCI factor ETFs, Deribit expiry, Kalshi combos) by period; needs `SEC_USER_AGENT`, first run fetches EDGAR filings and the Kalshi mirror (hours).
- `run_alpha_batch9.py`: alpha hunt batch 9 (dividend run-up and capture, initiations, CEF ex-dates and January, IPO lockups, splits) by period; first run downloads ~8k symbols from Yahoo in chunks (~30 min).
- `run_alpha_batch11.py`: alpha hunt batch 11 (Polymarket mentions/tennis replication of Kalshi near misses, Polymarket theme jumps -> ETFs, TDF rebalancing, January window cross-asset, BTC month-end, sharpened LETF reversal).
- `run_alpha_batch10.py`: alpha hunt batch 10 (Kalshi 15-min BTC and alt-coin fair value, near-expiry favourites, single-stock LETF reversal with control group, EMLC sleeve, YieldMax roll weeks).
- `run_auction_cycle.py`: Treasury auction cycle (candidate): per-cycle TLT/EDV windows, placebo, sleeves, 20% in the fund plan.
- `run_kalshi_maker.py`: Replay passive Kalshi quotes on 131M prints (`--shards`, `--k`).
- `run_cef_discount.py`: Closed-end fund discount z-score grid + walk-forward (needs `SEC_USER_AGENT`).
- `run_weather_brackets.py`: Kalshi temperature brackets vs NWP consensus (HF replay rows).
- `run_spinoffs.py`: Spin-off event study with the dead-children survivorship bound (needs `SEC_USER_AGENT`).
- `run_fomc_overlay.py`: Pre-FOMC drift event study and UPRO VT overlay.
- `run_kalshi_vol.py`: Sell XSP 0DTE iron flies when Kalshi's S&P range is narrow vs VIX1D (modelled option prices).

Notes: `watch_polymarket_iran.py` and `watch_wsb_spikes.py` are log-only and run on Josh's machine
(`pip install -e ".[polymarket]"`); `run_signal_blend*.py` and `run_ml_research.py`
need `.[ml]`; Kalshi scripts read the shared datasets in `/mnt/project-files/`.
