# Strategy catalog

All in `backtester/strategies/`, exported from `backtester.strategies`.
Results: $5,000, realistic costs, Yahoo/FRED/SEC data; CAGR / max drawdown.
Verdict: **keeper** (worth running or paper trading), **weak** (works but
trails a simple benchmark), **no edge**. Details per doc in `docs/README.md`.
Defaults shown are the class defaults, not necessarily the best run.

## Keepers

| Strategy | File / class | Key params (default) | Result | Example |
| --- | --- | --- | --- | --- |
| Crypto ensemble trend | `crypto.py` `EnsembleTrend` | lookbacks (5..250), top_n 20, target_vol 0.25, band 0.25 | 20%/yr, -20% DD at 50% vol target (Sharpe 1.2, ~20% invested); 100% target 36%/-41%; walk-forward 12%/-9%; BTC+ETH only (`top_n` 2, `min_history` 60) 28%/-32% at 0.4% fees 2018-26, ~110 trades/yr vs ~1,900 for top 20 | `run_ensemble_trend.py` |
| BTC+ETH Donchian breakout | `breakout.py` `Breakout` | entry 20, exit 10, max_positions | 39.5%/-64% vs BTC 31.4%/-83% (2017+); fees decide it (0.1% fee: 50.8%); walk-forward 36% | `run_breakout.py`, `run_robustness.py` |
| Vol-targeted breakout | `crypto.py` `VolTargetBreakout` | Breakout + target_vol 0.5, vol_lookback 30 | cuts DD -62% -> -37%, same Sharpe | `run_crypto_strategies.py` |
| UPRO vol target | `vol_target.py` `VolTarget` | symbol UPRO, target_vol 0.2, lookback 20, band 0.1 | 17.4%/-27%; HAR forecast version (`ml.vol_forecast.ForecastVolTarget`) Sharpe 0.86 -> 0.89 | `run_vol_target.py` |
| Multi-asset ETF trend | `multi_asset_trend.py` `MultiAssetTrend` | lookbacks (21,63,126,252), target_vol 0.15, max_gross 1.5, shorts via inverse ETFs | 6.9%/-18% since 2007 with BTC in the universe, +39% in 2008; ETFs only 3.9%/-22% (1.5x) or 3.3%/-16% at 1x gross; 50% SSO + 50% trend 16.3% vs SPY 15.1% | `run_multi_asset_trend.py` |
| Pension month-end | `pension_rebalance.py` `PensionRebalance` | stock SPY, bond TLT, days 4, when_bonds_lead SSO | SPY +1.1% avg in last 4 days after bonds lead (t 3.1); SPY 11.1% -> 13.4%/yr | `run_pension_rebalance.py` |
| Hedged S&P 100 momentum | `stock_momentum.py` `HedgedMomentum` | top_n 10, lookback 252, skip 21, hedge None/SH | 22%/yr unhedged vs 15.7% EW (survivor-biased); SH-hedged 4.3% | `run_hedged_momentum.py` |
| ETF breakout + meta-label filter (candidate) | `ml/metalabel.py` `MetaFiltered` around `Breakout` | 20/10 on 32 ETFs, skip worse half | Sharpe 0.61 -> 0.76 walk-forward, 0.67 -> 1.43 holdout (Jul 2024+); paper-trade | `run_ml_research.py` |
| Treasury auction cycle (candidate, paper only) | `auction_cycle.py` `simulate` (standalone, close-to-close) + `data/treasury_auctions.py` | out of TLT from the close 5 days before each 10y/30y auction cycle to the last auction close; optional TBF while out | TLT -38 bp/cycle in window (t -2.9, negative in 2008-16, 2017-24 and Jul 2024+ holdout); TLT/TBF sleeve 9.8%/-25% vs TLT 2.3%/-48%; 20% sleeve lifts Balanced fund Sharpe 1.04 -> 1.15 | `run_auction_cycle.py` |
| Polymarket Iran -> oil (paper only) | `jump_trades.py` (not a Strategy) | jump >= 5 pts/hr, hold 3h | +0.66%/trade, 34 trades, t 2.4 | `run_iran_oil.py`, `watch_polymarket_iran.py` |

## Weak or mixed

| Strategy | File / class | Result |
| --- | --- | --- |
| Leveraged trend filter | `trend_filter.py` `TrendFilter` (risk_on SSO, ma 200) | 12.6%/-43% vs SPY 10.9%/-55%; same Sharpe, cuts crash; walk-forward no better than fixed 200d |
| SVXY on VIX < VIX3M | `vix_term_structure.py` `VixTermStructure` | 29%/-55%, fragile to params |
| Sector rotation | `sector_rotation.py` `SectorRotation` (top_n 3, SPY 200d) | 9.9%/-20% |
| Turn of month | `turn_of_month.py` `TurnOfMonth` | TQQQ at TOM else QQQ 22.6%/-52% |
| Fundamental portfolio manager | `fundamental_allocator.py` `FundamentalAllocator` (default 80/10/10 QQQ/EFA+EEM/IJR, macro switch unemployment+SPY 200d -> IEF/BIL, sleeve 200d trend, vol_target 0.18; `stock_etf=None` + 60/20/20 = S&P 500 picks on SEC fundamentals, factors quality+lowvol, top_n 25, max_sector 0.25) | Default: 16.0%/-27% vs SPY 17.4%/-34% in 2019-26, 14.6%/-27% vs 10.7%/-55% 2007-26 (QQQ tilt chosen after the holdout). Stock picker: 2007-18 11.0%/-17% vs 6.5%/-55%, holdout 11.1%/-35% (no edge) |
| Wheel | `wheel.py` `Wheel` (standalone; XLF, put 0.25d, call 0.30d) | XLF 4.9%/-59% vs 5.7%/-83%; ±2%/yr skew uncertainty |
| RSI(2) mean reversion | `mean_reversion.py` `RSIMeanReversion` (rsi_n 2, entry 10, exit 70, trend_ma 200) | 6.5%/-20%, 26% invested; walk-forward 4.5% |
| Dual momentum (GEM) | `dual_momentum.py` `DualMomentum` | 5.9%/-33% vs SPY 10.9% |
| Credit spreads / condors | `spreads.py` `CreditSpreads` (standalone, modeled prices, fitted skew) | ~4%/yr since 2010 vs SPY 14%; management and bigger size worse |
| Risk parity (3x ETFs) | `risk_parity.py` `RiskParity` (target_vol 0.2, trend_ma) | 10%/yr since 2010, -23% in 2022 unless trend filter |
| 13F best-ideas clone | `hedge_fund_clone.py` `HedgeFundClone` + `best_ideas()` | 15.8% vs SPY 13.8%, QQQ 19%; -38% in 2022 |
| Crypto x-sectional momentum | `crypto.py` `CrossSectionalMomentum` (top_n 3, lookback 21) | 58% survivor-biased, 38% walk-forward, weak since 2022 |
| Crypto sleeves | `crypto.py` `Sleeves` (list of (strategy, symbols, weight)) | 10% DOGE/SHIB breakout sleeve +2pp CAGR |
| FX carry | `carry.py` `FXCarry` (2 long/2 short, 10% vol) | 5.7%/-23% since 2005, needs ~2.3x gross; trend filter hurts |
| Activist 13D | `events.py` `EventHold` + `data/activists.py` | +2.8% first month vs IWM, ~16 events/yr -> ~2%/yr |

## No edge

| Strategy | File / class | Result |
| --- | --- | --- |
| Insider cluster buys | `insider.py` `InsiderCluster`, `cluster_signals`, `classify_insiders` | 1.7%/-58% vs IWM 8.1%; opportunistic filter 3.5% -> 6.9% |
| Earnings premium | `earnings.py` `EarningsPremium` | 11.4% vs SPY 11.1%; costs take the edge |
| Same-month seasonality | `seasonality.py` `SameMonthSeasonality` | lagged EW universe in every 5-yr block |
| Crypto basis carry | `basis.py` `BasisCarry` (standalone) | ~3%/yr since 2021, below T-bills |
| Market sentiment overlays | `sentiment_overlay.py` `SentimentOverlay` (wraps any strategy; `signal_key`, low/high, low_mult/high_mult) + `data/sentiment_indices.py` `mood_index` | extreme fear -> higher next-month SPY return but ~1.8x vol; no fear/greed, VIX-backwardation, credit or Michigan rule beats UPRO vol target or multi-asset trend in both 2009-18 and 2019-26 (walk-forward Sharpe 0.60 vs 0.72); VIX>VIX3M exit helps plain SPY only (Sharpe 0.45->0.61, 0.93->1.00) |
| FinBERT market news mood overlay | same, `signal_key` 5d Benzinga headline z (`/mnt/project-files/finbert-sentiment/`) | halving exposure after very upbeat news weeks: UPRO VT Sharpe 0.68->0.73 (2010-15), 0.80->0.82 (2016-Jun 2020); tiny, data ends 2020; news mood mostly echoes yesterday's return |
| FinBERT Reddit mood overlay | same, `signal_key` 5d/7d z of WSB+stocks+investing / r/CryptoCurrency mood (2021-25) | 12 rule x base tests, 2021-23 vs 2024-25: best were +0.02-0.08 Sharpe in both (UPRO VT x2 after gloom, ensemble trend halved after cheer), consistent with chance |
| Crypto funding-rate overlay | same, `signal_key` 7d BTC/ETH funding | cut on crowded longs helped 2021-23, hurt 2024-26 (2 episodes) |
| Trump post overlay | `trump.py` `PostRiskOverlay` | no vol prediction; Sharpe flat, turnover x2 |
| TACO dip | `trump.py` `EventDip` | 23 events, t 0.47 |
| Gov equity stakes | `trump.py` `GovStakeEvents` | 13% gap before tradeable; 20/60d holds lose |
| S&P 500 index changes | `events.py` `EventHold` + `data/sp500.py` | nothing tradeable from free data |
| ML signal blend (ridge/LGBM/MLP/LSTM) | `signal_blend.py` `SignalBlend` + `ml/` | no model beats equal weight / ridge; Kalshi features don't help |
| Signal combination search | `combo_search.py` + `run_signal_combos.py` (all strategies as 0/1 signals or sleeves, every subset) | SPY timing best pick walk-forward Sharpe -0.04 vs SPY 0.71 (DSR 0.02, PBO 0.69); BTC/ETH best fails holdout (-0.23 vs hold 0.19); sleeve mixes: equal weight of all 16 non-crypto sleeves (Sharpe 0.88 WF, 0.75 holdout, -21%/-14% DD) beats the searched pick (0.81 / 0.62) |
| Regime switch, learning-to-rank | `ml/regime.py` `RegimeSwitch`, `ml/rank.py` `TopN` | trails 200d filter; ranking = survivor bias |
| Closed-end fund discounts | `cef_discount.py` `CEFDiscount` (lookback 252, entry_z 1.5, exit_z 0.5, 10 slots) + `data/cef.py` | 261 CEFs: holdout (2019+) alpha 1.1%/yr vs equal-weight CEFs, 2.9% walk-forward; below the 3% bar |
| Pre-FOMC drift overlay | `fomc_overlay.py` `PreFOMCOverlay` (base, add 0.5) + `data/fomc.py` | SPY pre-FOMC return = any 2 days since 2009; UPRO VT Sharpe -0.09 / 0.00 |
| Calendar overlays (OPEX, holidays) | `fomc_overlay.py` `CalendarOverlay` (per-day multiplier), `opex_days` | OPEX week up / week after down: Sharpe -0.07/-0.10; pre-holiday x1.5: +0.015/0.00 |
| Scheduled-flow windows (batch 3) | `event_windows.py` (standalone) | GSCI roll USO-USL, Russell recon IWM-IWB, other auction cycles, tax/coupon dates, CME BTC expiry, MSCI, VIX expiry, quarter-end: fail holdout or placebo |
| Alpha hunt batch 8 | `examples/run_alpha_batch8.py` (standalone, needs `SEC_USER_AGENT`) | Odd-lot tender offers (company SC TO-I with odd-lot priority, buy 99 sh under the offer): fixed price +6.0%/+7.6% per event (t 3.7/5.8, no losers, ~1.3/yr, 0 holdout events) = NEAR; Dutch +3.6%/+1.6% fails after a $25 fee; fund tenders prorate odd lots. CEF rights offering short 25d vs CEFs: gross +2.1% holdout (t 2.8) but 0.6% after 10%/yr borrow = NEAR. CEF activist 13D (decayed), MSCI factor ETFs, Deribit BTC expiry, Kalshi combos buy No 5-40c (taker holdout -0.7c, t -4.2): fail; SPAC trust arb untestable |
| Alpha hunt batch 9 | `examples/run_alpha_batch9.py` (standalone, needs `SEC_USER_AGENT`) | Dividend run-up 10d before ex vs SPY (+28bp t 4.0 2005-16, -5bp, -34bp holdout), ex-dividend capture (net of 10bp spread -8/-6bp since 2017), initiations/25%+ raises (60d ~0), CEF ex-date discount widening (keeps falling, -16 to -28bp/5d), CEF tax-loss January (+0.8%/yr, 50% of years), IPO lockup rebound (CAR keeps falling: -5.2% in 20d holdout), forward splits: fail |
| Alpha hunt batch 11 | `examples/run_alpha_batch11.py` (standalone) | Polymarket replication of Kalshi near misses: taker-No entries on mentions at 20-80c +1.6c gross (+3.7c to Jun 2025, -0.2c since; -0.6c net clustered by event) = no replication; No at 20-40c +3.0c net, t 1.7 clustered = weak (an earlier +6c used fills on the wrong side of the spread); tennis favourites 80-95c -0.9c net over 9.5k matches = FAIL. Iran->oil jump rule on Ukraine/Taiwan/shutdown/tariff markets: pooled -4bp (t -1.4) = FAIL. TDF rebalancing, January window outside commodities, BTC month-end since ETFs: FAIL. Commodity January re-check +2.4% (t 4.5); sharpened LETF top decile +1.3% holdout (t 1.5) = still NEAR |
| Alpha hunt batch 10 | `examples/run_alpha_batch10.py` (standalone) | Single-stock 2x LETF rebalancing reversal (buy the underlying next open->close after a < -2 sd day when LETF pressure > 0.02 of ADV): +2.6%/trade Jul 2024-Jun 2025 (t 3.5), +0.5% since (t 1.3), non-LETF control +0.4% = NEAR. Kalshi 15-min BTC vs BRTI fair value and alt coins vs spot (taker -0.6 to -2.4c holdout), near-expiry favourites, EMLC carry sleeve (Sharpe 0.1-0.23), YieldMax roll weeks: fail; MSTR mNAV not run |
| COT positioning | `cot_positioning.py` `CotPositioning` (signal_key, top_n 3, short False) + `data/cot.py` `positioning_signals` (`flow1/4` KRT reversal, `hedging_pressure`, `commercial_index`, `combo`) | 20 futures ETFs, weekly, 2011-2026: KRT flow reversal Sharpe 0.41 pre-cost but 72x turnover -> -0.7 net; hedging pressure wrong sign; commercial index L/S 0.24/0.17/0.02 (design/valid/holdout), ~1%/yr over T-bills; equal weight beats all long-only versions |
| Congress trade copying | `events.py` `EventHold` + `data/congress.py` `buy_events` (keyed on filing date) | 2014-2026, hold 63d, 10 slots: all buys trail SPY (-7.9/-0.4/-0.6 pts design/valid/holdout); $50k+ buys +12/+19 pts since 2020 but ~10 mega-cap tech names, -3 pts 2014-19, flips with hold 21/126d; 3+ member clusters +0.5% per 63d event, portfolio mixed |
| Benchmark | `benchmark.py` `BuyAndHold` | — |

## Combined fund

| Plan | Code | Result |
| --- | --- | --- |
| $5,000 fund: Balanced 30% UPRO vol target, 35% multi-asset trend (ETFs, 1x), 15% pension month-end, 20% BTC+ETH ensemble trend; Growth 40/20/10/30 | `portfolio.py` `sleeve_returns`, `combine` (quarterly + 10-pt band, optional drawdown brake) | Balanced 15.2%/-16% vs SPY 15.3%/-34% (Nov 2018-Oct 2026, Sharpe 1.00 vs 0.69); 2022-26 9.2%/-15% vs SPY 11.8%/-25% (same Sharpe); drawdown brakes cost 1-2 pts/yr; `run_fund_plan.py`, today's targets `fund_signals.py` |

Intraday, Kalshi, 0DTE, pairs and meme-coin results are in
[research-modules.md](research-modules.md).
