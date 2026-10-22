# Strategy catalog

All in `backtester/strategies/`, exported from `backtester.strategies`.
Results: $5,000, realistic costs, Yahoo/FRED/SEC data; CAGR / max drawdown.
Verdict: **keeper** (worth running or paper trading), **weak** (works but
trails a simple benchmark), **no edge**. Details per doc in `docs/README.md`.
Defaults shown are the class defaults, not necessarily the best run.

## Keepers

| Strategy | File / class | Key params (default) | Result | Example |
| --- | --- | --- | --- | --- |
| Crypto ensemble trend | `crypto.py` `EnsembleTrend` | lookbacks (5..250), top_n 20, target_vol 0.25, band 0.25 | 20%/yr, -20% DD at 50% vol target (Sharpe 1.2, ~20% invested); 100% target 36%/-41%; walk-forward 12%/-9% | `run_ensemble_trend.py` |
| BTC+ETH Donchian breakout | `breakout.py` `Breakout` | entry 20, exit 10, max_positions | 39.5%/-64% vs BTC 31.4%/-83% (2017+); fees decide it (0.1% fee: 50.8%); walk-forward 36% | `run_breakout.py`, `run_robustness.py` |
| Vol-targeted breakout | `crypto.py` `VolTargetBreakout` | Breakout + target_vol 0.5, vol_lookback 30 | cuts DD -62% -> -37%, same Sharpe | `run_crypto_strategies.py` |
| UPRO vol target | `vol_target.py` `VolTarget` | symbol UPRO, target_vol 0.2, lookback 20, band 0.1 | 17.4%/-27%; HAR forecast version (`ml.vol_forecast.ForecastVolTarget`) Sharpe 0.86 -> 0.89 | `run_vol_target.py` |
| Multi-asset ETF trend | `multi_asset_trend.py` `MultiAssetTrend` | lookbacks (21,63,126,252), target_vol 0.15, max_gross 1.5, shorts via inverse ETFs | 6.9%/-18% since 2007, +39% in 2008; 50% SSO + 50% trend 16.3% vs SPY 15.1% | `run_multi_asset_trend.py` |
| Pension month-end | `pension_rebalance.py` `PensionRebalance` | stock SPY, bond TLT, days 4, when_bonds_lead SSO | SPY +1.1% avg in last 4 days after bonds lead (t 3.1); SPY 11.1% -> 13.4%/yr | `run_pension_rebalance.py` |
| Hedged S&P 100 momentum | `stock_momentum.py` `HedgedMomentum` | top_n 10, lookback 252, skip 21, hedge None/SH | 22%/yr unhedged vs 15.7% EW (survivor-biased); SH-hedged 4.3% | `run_hedged_momentum.py` |
| ETF breakout + meta-label filter (candidate) | `ml/metalabel.py` `MetaFiltered` around `Breakout` | 20/10 on 32 ETFs, skip worse half | Sharpe 0.61 -> 0.76 walk-forward, 0.67 -> 1.43 holdout (Jul 2024+); paper-trade | `run_ml_research.py` |
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
| Trump post overlay | `trump.py` `PostRiskOverlay` | no vol prediction; Sharpe flat, turnover x2 |
| TACO dip | `trump.py` `EventDip` | 23 events, t 0.47 |
| Gov equity stakes | `trump.py` `GovStakeEvents` | 13% gap before tradeable; 20/60d holds lose |
| S&P 500 index changes | `events.py` `EventHold` + `data/sp500.py` | nothing tradeable from free data |
| ML signal blend (ridge/LGBM/MLP/LSTM) | `signal_blend.py` `SignalBlend` + `ml/` | no model beats equal weight / ridge; Kalshi features don't help |
| Regime switch, learning-to-rank | `ml/regime.py` `RegimeSwitch`, `ml/rank.py` `TopN` | trails 200d filter; ranking = survivor bias |
| Benchmark | `benchmark.py` `BuyAndHold` | — |

Intraday, Kalshi, 0DTE, pairs and meme-coin results are in
[research-modules.md](research-modules.md).
