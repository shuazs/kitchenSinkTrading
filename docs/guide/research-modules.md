# Research modules (outside `strategies/`)

Simulators with their own loops, mostly intraday or prediction-market work.
Each has one example script and one test file. Verdicts are final unless noted.

| Module | Main functions / classes | What it tests | Verdict |
| --- | --- | --- | --- |
| `intraday.py` | `minute_grid`, `first_half_hour`, `noise_area`, `summary` | Gao et al. first-half-hour and Zarattini noise-area on SPY/QQQ minute bars | below costs |
| `intraday_lab.py` | `Grid`, `simulate(grid,pos,cost_bps,lag)`, rule fns (`momentum_window`, `noise_area`, `opening_range`, `vwap_trend`, `vwap_reversion`, `gap`, `hour_of_day`, `session_split`), `run_family`, `walk_forward` | 3,448 intraday variants on 1-min SPX/NQ/QQQ/BTC/ETH, 2021-2026 | no edge: ~1 bp/trade S&P pre-cost; QQQ 15-min ORB failed 2026 holdout |
| `shortterm.py` | `hourly_panel`, `zscore_combo`, `ridge`, `lightgbm`, `walk_forward`, `score`, `zero_dte`, `OptionCosts` | hourly SPY/QQQ signal combo -> ATM 0DTE options | no edge: hit rate 46-53%, needs 52-57% |
| `kalshi_minute.py` | `parse_strikes`, `trade_implied`, `leadlag_table`, `tail_hit_rate`, `zero_dte_proxy`, `breakeven_hit_rate` | do Kalshi hourly ladders lead SPX/NDX/BTC by 1-10 min | no: flow follows the past minute |
| `kalshi_catchup.py` | `known_prices`, `minute_vol`, `digital_fair_value`, `catchup_backtest`, `summarize`, `taker_fee` | buy stale Kalshi contracts after a sharp underlying move | BTC loses (bots); S&P 2-min moves ~$3k/14 mo, t 2.0, tiny fills, maybe luck |
| `kalshi_vol.py` | `ladder_quantiles`, `end_of_day_ladders`, `range_panel`, `add_signal`, `iron_fly_price`, `fair_iv_mult`, `iron_fly_backtest`, `fly_summary`, `FlyCosts` | Kalshi end-of-day S&P range vs VIX1D -> sell XSP 0DTE iron flies at 12:30/14:30 ET | candidate: narrow days keep 62-67% of implied variance vs 45-46% wide; 12:30 fly +13.5%/trade at break-even prices (t 2.5, mostly 2025); option prices modelled, paper-trade first |
| `kalshi_crowd.py` | `move_z`, `move_triggers`, `spike_triggers`, `calibration`, `Ladder`, `simulate`, `summarize` | crowd bias after big moves / volume spikes | no edge; contracts stay calibrated |
| `pairs.py` | `kalshi_fee`, `half_life`, `rolling_hedge`, `zscore`, `asset_leg_trades`, `fade_contract_spread` | prediction market vs asset as pairs | costs exceed edge |
| `crypto_pairs.py` | `adf_t`, `band_positions`, `simulate_pair`, `rotation_returns`, `rolling_spread_z`, `select_pairs`, `rolling_pairs`, `trade_list`, `COST_BPS` | crypto pairs/stat-arb: BTC/ETH z-score, 19-coin cointegration/distance pairs, long-only rotation, 5m-1h intraday | no edge: BTC/ETH spread trends (reversion Sharpe -0.65 even at futures fees); pair walk-forward -10%/yr; intraday 0-4 bp gross per round trip vs 14 bp futures cost |
| `jump_trades.py` | `jump_events`, `jump_trades`, `summarize`, `grid` | trade asset after a prediction-market jump (hourly) | Iran -> oil is the one lead (see strategies.md) |
| `news_sentiment.py` | `session_for`, `stock_events`, `bucket_table`, `spread_by_year`, `cashtag_whitelist`, `ticker_mentions`, `mention_spikes`, `fade_book`, `kalshi_directional`, live: `post_session`, `spike_candidates`, `load_wsb_tickers` (`data/wsb_tickers.txt`) | FinBERT on Benzinga headlines (2009-20) and Reddit posts (2021-25) -> stocks, WSB buzz, Kalshi S&P/BTC | news: real but below costs; Reddit mood -> Kalshi: no edge; shorting WSB 10x-mention spikes at the open is a fat-tailed candidate (+1.8%/trade both periods, t ~1.6-2, squeezes -60 to -70%) |
| `event_markets.py` | `kalshi_category`, `taker_prints`, `calibration`, `exclusive_events`, `bracket_arb`, `arb_episodes`, `cross_arb`, `snapshot_quotes`, `favourite_bets`, `first_entries`, `bankroll`, `lead_follow`, `name_tokens`, `poly_subject`, `match_by_name`, `POLY_US_SPORTS` | non-financial Kalshi/Polymarket (sports, politics, weather, culture): taker calibration, longshot/favourite rules, jumps, sum-to-$1 and two-venue arbitrage, Kalshi -> Polymarket lead | takers lose 1-3c/contract everywhere; arbs 0.5-2c taken by bots in <1s; **retracted (batch 4)**: the Kalshi -> Polymarket sports lead (+5.6c/share, US games +3.6c net of the US fee) came from `lead_follow` dropping trades with no exit print after the hold (the ~20% that lose 26c); with those closed at settlement (`settle=`) or a causal exit it is -2.3c/share, every month negative. Mention markets: No-takers at 20-80c earn +3.6c/contract after fees (2025 t 5.3, Jan 2026 t 2.6; `run_kalshi_mentions_no.py`), forward-test candidate |
| `kalshi_maker.py` | `orders`, `quote_reference`, `maker_fills(prints,k,latency_s,through)`, `settle`, `summarize`, `maker_fee` | passive Kalshi quotes at last print ±k, 2 s latency, trade-through or trade-at fills, held to settlement | no edge: -2.8c/contract holdout (through), -1.3c (first in queue); long-tail markets -7 to -15c |
| `weather_brackets.py` | `load_rows`, `brier`, `edge_trades`, `taker_fee` | Kalshi temperature brackets vs 7-model NWP consensus (HF openthomas replay, Aug-Oct 2026) | no edge: market Brier 0.122 vs model 0.163; -3 to -4c/contract every month |
| `weather_trades.py` | `run_revisions`, `trading_day_revisions`, `rolling_z`, `kalshi_bounds`, `locked_side`, `event_returns` | weather as a signal (2026-10-29, `run_weather.py`): GEFS forecast revisions -> UNG / gas producers / grain ETFs, Europe-Asia weather -> TTF/LNG, GHCN actuals x EIA storage day, Corn Belt stress, utility earnings, hurricanes, Kalshi daily-high lock-in and monthly rain/snow | all fail except **NEAR**: Kalshi daily-high brackets already decided by two METARs still print at 60-97c on ~1% of city-days, every such print won (+11c in-sample, +4.6c holdout, ~$1k/yr at printed size); paper logger `kalshi_weather_lock.py` |
| `spinoffs.py` | `child_events`, `dead_fate`, `survivorship_bound` | Form 10-12B spin-offs, buy day 21, hold 6-12 months vs IWM, dead children added back | survivors +19.5%/yr holdout; with 208 dead +7.8%, t 1.25: unproven |
| `event_windows.py` | `anchor_positions`, `nth_business_day`, `event_returns`, `split_stats`, `placebo`, `window_mask`, `russell_recon`, `vix_expiry`, `last_friday`, `third_friday` | event windows around scheduled flows (alpha hunt batch 3) | no edge: GSCI roll, Russell recon, 2/5/7/3/20y auctions, tax dates, CME BTC expiry, MSCI, coupon dates, VIX expiry, quarter-end all fail holdout or placebo |
| `auction_cycle.py` | `cycles`, `window_mask`, `window_returns`, `simulate` | Treasury 10y/30y auction cycle (Lou-Yan-Zhang): long bonds cheapen before auctions | **candidate**: TLT -38 bp/cycle, t -2.9, every period and window 3-7 days negative; see strategies.md |
| `letf.py` | `LETF` (single-stock 2x ETFs per underlying), `down_z`, `pressure` | single-stock LETF close-rebalancing pressure (batch 10) | NEAR, post-hoc; paper (see paper-trading.md) |
| `combo_search.py` | `TimingSearch` (every subset x AND/OR/majority/share rule x vehicle), `SleeveSearch` (every subset of sleeve returns, equal / inverse-vol), `evaluate` (half-yearly walk-forward, holdout, `deflated_sharpe`, `pbo` CSCV), `Moments`, `open_to_open`, `unit_cost` | every combination of the repo's strategies as signals (785k SPY timing, 4k BTC/ETH timing, 2.2M sleeve mixes) | no combination beats its simple benchmark out of sample; holding all non-crypto sleeves equal-weight beats any searched mix |
| `memecoin_calls.py` | `buy`, `sell`, `simulate_calls`, `bankroll` | following pump.fun calls on the bonding curve | loses in every realistic variant |

Kalshi fee: `0.07 * p * (1-p)` per contract (taker), rounded up to the cent.
Makers earn what takers lose (0.4-2c/contract by category, ~2c on mentions) before maker fees and queue position.
Passive (maker) quotes replayed from prints in `kalshi_maker.py` lose; a live order-book recorder would only matter for queue-first bots.

Alpha hunt (2026-10-23, doc https://claude.ai/artifact/PtWhQ465ANNZFY7WZCxKHs): 20 hypotheses, one candidate (auction cycle). Also killed with scratch scripts only (`/mnt/project-files/alpha-hunt/scripts/`): small-cap PEAD, federal contract awards, micro-cap QVM, January rebound, Polymarket->Kalshi lead, bond month-end, vol target on other ETFs, VRP-gated SVXY, country-ETF reversal, duration timing, factor-ETF momentum. Batch 3 (scheduled flows, doc https://claude.ai/artifact/GRfc2PDuhE4eaRMrP3c5VH): 9 more, all killed (`run_scheduled_flows.py`).

## `ml/`

| File | Contents |
| --- | --- |
| `features.py` | `build_panel(data, symbols)`: weekly (date, symbol) panel of `FEATURES` + `ret`; `asset_features`, `market_features` |
| `splits.py` | `walk_forward_folds` (yearly, purged), `holdout_fold` (locked holdout from 2024-07) |
| `models.py` | `EqualWeight`, `RidgeModel`, `GradientBoosting`, `MLP`, `LSTMMomentum`; `LADDER` |
| `contest.py` | `out_of_sample_positions`, `information_coefficient`, `positions_frame`, `deflated_sharpe` |
| `labels.py` | `entries_and_exits`, `trade_returns`, `triple_barrier` |
| `metalabel.py` | `event_table`, `Logistic`, `GBM`, `walk_forward_probs`, `decide`, `MetaFiltered` (the keeper candidate) |
| `regime.py` | `jump_features`, `JumpModel`, `walk_forward_regimes`, `RegimeSwitch` |
| `vol_forecast.py` | `garman_klass`, `vol_features`, HAR, `walk_forward_vol`, `qlike`, `ForecastVolTarget` |
| `rank.py` | `monthly_panel`, `walk_forward_scores`, `rank_ic`, `TopN` |

Rule used throughout: refit yearly on purged expanding windows, score the
July 2024+ holdout once, compare to random/shuffled controls.
