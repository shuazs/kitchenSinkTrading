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
| `jump_trades.py` | `jump_events`, `jump_trades`, `summarize`, `grid` | trade asset after a prediction-market jump (hourly) | Iran -> oil is the one lead (see strategies.md) |
| `memecoin_calls.py` | `buy`, `sell`, `simulate_calls`, `bankroll` | following pump.fun calls on the bonding curve | loses in every realistic variant |

Kalshi fee: `0.07 * p * (1-p)` per contract (taker), rounded up to the cent.
Untested: passive (maker) Kalshi orders, which need a live order-book recorder.

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
