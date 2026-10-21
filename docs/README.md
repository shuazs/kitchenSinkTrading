# Research write-ups

The research behind this repo lives in six project docs on claude.ai (private

The research behind this repo lives in project docs on claude.ai (private
to the project owner). This page links them, says which code each one drove,
and keeps a short summary so the main conclusions survive here too. Results
are backtests with the caveats in each doc (survivorship bias, modeled option
prices, short crypto history), not forecasts.

| Doc | Code | Examples |
| --- | --- | --- |
| [Trading strategy research shortlist](https://claude.ai/artifact/Jeej8Zf3GeZ5YtXSr8sqUj) | `strategies/trend_filter.py`, `breakout.py`, `wheel.py`, `mean_reversion.py`, `dual_momentum.py`, `insider.py` | `run_trend_filter.py`, `run_breakout.py`, `run_wheel.py`, `run_mean_reversion.py`, `run_dual_momentum.py`, `run_insider.py` |
| [Strategy backtests on real data](https://claude.ai/artifact/DGHDmCB4QAUhNubmXSa4eq) | the above | `run_robustness.py` |
| [Defined-risk options spreads](https://claude.ai/artifact/JCmkpwmNzN9XkAGzzbtdpo) | `strategies/spreads.py` | `run_spreads.py`, `fit_spy_skew.py` |
| [More strategy ideas: research and forums](https://claude.ai/artifact/8y1EZth95owGPrFMepp5n2) | `strategies/vol_target.py`, `vix_term_structure.py`, `sector_rotation.py`, `turn_of_month.py` | `run_vol_target.py`, `run_vix_term_structure.py`, `run_sector_rotation.py`, `run_turn_of_month.py` |
| [Academic strategy shortlist](https://claude.ai/artifact/Cr3Wy5dch15aHhypKRvSgo) | `strategies/crypto.py` (`EnsembleTrend`), `insider.py` (`classify_insiders`), `earnings.py`, `seasonality.py`, `intraday.py`, `data/alpaca.py` | `run_ensemble_trend.py`, `run_opportunistic_insider.py`, `run_earnings_premium.py`, `run_seasonality.py`, `run_intraday_momentum.py` |
| [Crypto strategies and meme coin sentiment](https://claude.ai/artifact/G2XAGWzHpp7rkDpTdA8FzM) | `strategies/crypto.py`, `memecoin_calls.py`, `data/memecoins.py` | `run_crypto_strategies.py`, `run_memecoin_calls.py` |
| [Trump-driven trading strategies](https://claude.ai/artifact/66WWbb1fHzF8kY1mnLSzhR) | `strategies/trump.py`, `data/trump_posts.py`, `data/gov_stakes.py` | `run_trump_overlay.py`, `run_taco_dip.py`, `run_gov_stakes.py` |
| [Large-account strategies scaled to $5k](https://claude.ai/artifact/JKf6qVGf7N7Ep47XbSixYL) | `strategies/carry.py`, `basis.py`, `stock_momentum.py`, `data/fx.py` | `run_fx_carry.py`, `run_basis_carry.py`, `run_hedged_momentum.py` |
| [Fund strategies for a $5k account](https://claude.ai/artifact/N273u9qQ2t6VC6xEf6KoZf) | `strategies/multi_asset_trend.py`, `risk_parity.py`, `pension_rebalance.py`, `hedge_fund_clone.py`, `events.py`, `data/thirteenf.py`, `data/activists.py`, `data/sp500.py` | `run_multi_asset_trend.py`, `run_risk_parity.py`, `run_pension_rebalance.py`, `run_hedge_fund_clone.py`, `run_activists.py`, `run_index_changes.py` |

## Summaries

**Research shortlist.** Nine candidates for a small, risk-tolerant account,
ranked by evidence: leveraged-ETF 200-day trend filter, crypto N-day-high
breakout and momentum, cash-secured put writing (the wheel), RSI(2) mean
reversion on ETFs, dual momentum, FinBERT headline sentiment, Form 4 insider
cluster buying, post-earnings drift and a FRED macro overlay. Avoid buying
0DTE or short-dated options, overnight-only holding, and anything tuned on a
single bull market. Strategies 1, 2, 3, 4 and 7 were built first, with
$5,000 starting capital.

**Backtests on real data.** Results for the first strategies on Yahoo, FRED
and SEC data with realistic costs, plus robustness checks (crypto fee levels,
a wider coin basket and walk-forward parameter selection).

**Defined-risk options spreads.** SPY 30-45 DTE put credit spreads and iron
condors priced with Black-Scholes on a VIX-based surface (term structure from
VIX9D/VIX3M/VIX6M, put skew fitted to real SPY quotes on 2026-10-20).
Held to expiry they made about 4%/yr since 2010 against SPY's 14%; 50%
take-profit with a 21-DTE exit and larger sizing did worse. The answer
depends on the skew assumption, so real option chain history is the next
check. Skip 0DTE and single-stock spreads.

**More strategy ideas.** Four ideas from papers and trading forums, tested on
real data: UPRO volatility-targeted to 20% (17.4% CAGR, -27% max drawdown),
SVXY held while VIX < VIX3M (29%, -55%, but fragile to its parameters), top-3
sector rotation with a SPY 200-day filter (9.9%, -20%), and TQQQ over the
turn of the month, QQQ otherwise (22.6%, -52%).

**Crypto and meme coins.** A 10-coin weekly momentum basket made 58% CAGR
since 2018 (survivor-biased; 38% walk-forward vs BTC's 20%, and weak since
2022). Volatility targeting cut the breakout's drawdown from -62% to -37% but
did not raise its Sharpe. A 10% DOGE/SHIB breakout sleeve added about 2 points
of CAGR. Following public pump.fun calls lost money in every realistic
variant, and sentiment trading on new meme coin launches is not realistic to
backtest or trade from a small account.

**Trump-driven strategies.** Real-time trading of his posts is out of reach
(institutions get them in milliseconds and prices often move first) and his
disclosed holdings are mostly bonds reported weeks late, so three slow ideas
were tested on daily bars with 52,534 posts (2017-2021 tweets, 2022-May 2026
Truth Social). A tariff-post burst does not predict next-week SPY volatility
(13.9% vs 14.5% annualised), and cutting UPRO's leverage after bursts barely
moved Sharpe while doubling turnover. SPY drops of 1.5%+ on tariff-post days
bounced +0.9% over 5 days vs +0.4% for other drops, but on 23 events with a
Welch t of 0.47 and the same 61% hit rate; the TACO filter is not a reliable
edge. After 13 government equity stake reports the median stock gapped 13%
before anyone could buy; holding 5 days from the first open made money in
this small sample, but 20- and 60-day holds lost (median -4% and -17%).

**Academic strategy shortlist.** A literature search (replication studies:
Jensen-Kelly-Pedersen, McLean-Pontiff, Chen-Velikov) found most stock
anomalies net close to zero after costs and publication. Four picks were
tested on real data with $5,000 and realistic costs:
- *Ensemble crypto trend* (8 Donchian lookbacks, top 20 liquid coins,
  volatility-sized): Sharpe 1.2 and -20% max drawdown since late 2018 at a
  50% vol target, but only about 20% invested (20% CAGR); a 100% target gave
  36% CAGR with -41% drawdown. BTC held: 38%, -77%. Walk-forward 12%, -9%.
  Survivorship bias remains for coins Yahoo has dropped.
- *Opportunistic insider buys*: opportunistic purchases had higher mean
  excess returns than routine ones (+2.0% vs +1.4% over 60 days vs IWM),
  but medians were negative, and no variant beat IWM after small-cap costs.
  Dropping routine insiders did lift the cluster strategy from 3.5% to 6.9% CAGR.
- *Earnings announcement premium* (8-K Item 2.02 dates, predicted a year
  ahead): +0.34% per trade vs SPY in 2006-12, +0.20% in 2013-19, +0.15%
  since 2020, gross. Costs take all of it: 11.4% CAGR vs SPY's 11.1%.
- *Same-month seasonality* on sector and country ETFs: picks lagged the
  equal-weight universe in every 5-year block; no edge with ETF-length history.
- *Intraday SPY/QQQ momentum*: simulators built; real test waits on Alpaca
  minute data (free account, see the README).

**Large-account strategies scaled to $5k.** Which big-account strategies fit
a $5,000 account, and how. Diversified trend following and leveraged risk
parity rank highest (built from the fund-strategies research). The $25,000
pattern-day-trader minimum was removed (SEC approval April 2026, effective
June 4, 2026; brokers have until October 2027). Portfolio-margin option
selling, futures books, covered-call portfolios and box spreads do not scale
down. Three were tested here. G10 + MXN currency carry (2 long, 2 short, 10%
volatility target) made 5.7% CAGR since 2005 with a -23% drawdown, but needs
about 2.3x gross exposure, more than one micro FX contract per leg allows on
$5k; adding a 3-month trend filter made it worse. Crypto cash-and-carry on
the CME basis made about 3% a year since 2021, below T-bills, because a
quarter of the account sits as margin and Yahoo's futures and spot closes are
too far apart in time to measure the basis cleanly. Top-10 S&P 100 momentum
made 22% CAGR since 2007 against 15.7% for an equal-weight basket of the same
(survivor-biased) names; fully hedging it with SH left 4.3%.

**Fund strategies for a $5k account.** What CTAs, hedge funds, risk-parity
and event-driven desks run, rebuilt with ETFs and public SEC data. Multi-asset
ETF trend (15 ETFs plus BTC, vol-scaled, inverse ETFs for shorts) made 6.9%/yr
since mid-2007 with a -18% worst drawdown, +39% in 2008 and +5% in 2022, and
beat DBMF and KMLM on Sharpe over their lives. 50% SSO plus 50% trend made
16.3% against SPY's 15.1% since 2009 at the same Sharpe. Pension month-end
rebalancing is the strongest new signal: after bonds beat stocks month to
date, SPY rose 1.1% on average over the month's last 4 days (t = 3.1, similar
in both halves of the sample); holding SSO then lifted SPY from 11.1% to
13.4%/yr since 2006. Leveraged risk parity with 3x ETFs made 10%/yr since
2010, behind SPY, and lost 23% in 2022 unless a 200-day trend filter was on.
The 13F best-ideas clone (top 3 holdings of 300-900 concentrated managers,
15 stocks, quarterly) made 15.8%/yr since late 2013 against SPY's 13.8%, but
QQQ made 19% with a better Sharpe, and the clone fell 38% in 2022. Activist
13D targets beat IWM by 2.8% in the first month after filing, but there are
only about 16 a year, so a 5-slot book made about 2%/yr. S&P 500 additions
and deletions left nothing tradeable from free data.
