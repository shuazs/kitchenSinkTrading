# Research write-ups

The research behind this repo lives in claude.ai (private
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
| [Crypto strategies and meme coin sentiment](https://claude.ai/artifact/G2XAGWzHpp7rkDpTdA8FzM) | `strategies/crypto.py`, `memecoin_calls.py`, `data/memecoins.py` | `run_crypto_strategies.py`, `run_memecoin_calls.py` |
| [Academic strategy shortlist](https://claude.ai/artifact/Cr3Wy5dch15aHhypKRvSgo) | `strategies/crypto.py` (`EnsembleTrend`), `insider.py` (`classify_insiders`), `earnings.py`, `seasonality.py`, `intraday.py`, `data/alpaca.py` | `run_ensemble_trend.py`, `run_opportunistic_insider.py`, `run_earnings_premium.py`, `run_seasonality.py`, `run_intraday_momentum.py` |
| [Fund strategies for a $5k account](https://claude.ai/artifact/N273u9qQ2t6VC6xEf6KoZf) | `strategies/multi_asset_trend.py`, `risk_parity.py`, `pension_rebalance.py`, `hedge_fund_clone.py`, `events.py`, `data/thirteenf.py`, `data/activists.py`, `data/sp500.py` | `run_multi_asset_trend.py`, `run_risk_parity.py`, `run_pension_rebalance.py`, `run_hedge_fund_clone.py`, `run_activists.py`, `run_index_changes.py` |
| [Large-account strategies scaled to $5k](https://claude.ai/artifact/JKf6qVGf7N7Ep47XbSixYL) | `strategies/carry.py`, `basis.py`, `stock_momentum.py`, `data/fx.py` | `run_fx_carry.py`, `run_basis_carry.py`, `run_hedged_momentum.py` |
| [Trump-driven trading strategies](https://claude.ai/artifact/66WWbb1fHzF8kY1mnLSzhR) | `strategies/trump.py`, `data/trump_posts.py`, `data/gov_stakes.py` | `run_trump_overlay.py`, `run_taco_dip.py`, `run_gov_stakes.py` |
| [Neural nets for combining our signals](https://claude.ai/artifact/PAQWQ3c7znrpmPXXzSE65J) | `ml/` (features, splits, models, contest), `strategies/signal_blend.py` | `run_signal_blend.py` |
| [Machine learning on the repo's research signals](https://claude.ai/artifact/73kpBbdn17TCsHkBjvNuNv) | `ml/labels.py`, `metalabel.py`, `regime.py`, `vol_forecast.py`, `rank.py` | `run_ml_research.py` |
| [Prediction markets as leading signals](https://claude.ai/artifact/ANDHSBxUprwff7eo4XFGde) | `jump_trades.py`, `data/polymarket.py`, `data/yahoo.py` (`load_hourly`) | `run_iran_oil.py`, `watch_polymarket_iran.py` |
| [Short-term signal combination for 0DTE](https://claude.ai/artifact/VWco2QLDqaWA6dSAzNKMWc) | `shortterm.py` | `run_shortterm_combo.py` |
| [Kalshi minute signals for 0DTE](https://claude.ai/artifact/UEKZrhbdbKtC5f59sVmHig) | `kalshi_minute.py`, `data/minute_mirrors.py` | `run_kalshi_minute_leadlag.py` |
| [Prediction markets as pairs trades](https://claude.ai/artifact/9zz3e8t7mY7g2Mmq8ZPL7F) | `pairs.py` | `run_prediction_market_pairs.py` |
| [Trading Kalshi off stock moves](https://claude.ai/artifact/NE4wvaFAWDEu56vDUWotKi) | `kalshi_catchup.py` | `run_kalshi_catchup.py` |
| Kalshi crowd bias after big moves (no write-up; results in [PR #18](https://github.com/shuazs/kitchenSinkTrading/pull/18)) | `kalshi_crowd.py` | `run_kalshi_crowd.py` |
| [Non-financial prediction markets](https://claude.ai/artifact/S3h1yCUVXdGTDbXTrCRm6K) | `event_markets.py`, `data/kalshi_hf.py`, `data/polymarket.py` (`side`) | `run_event_markets.py` |
| [Intraday strategies on 1-minute data](https://claude.ai/artifact/Vker1MLqEfUTXXELHms2UL) | `intraday_lab.py`, `intraday.py`, `data/minute_mirrors.py` | `run_intraday_strategies.py`, `run_intraday_momentum.py` |
| Long-term fundamental portfolio manager (no write-up; results in the PR and below) | `strategies/fundamental_allocator.py`, `data/fundamentals.py`, `data/fred.py` (`unemployment_rate`) | `run_fundamental_allocator.py` |
| [Trading off FinBERT news and Reddit sentiment](https://claude.ai/artifact/8ENP71Kn66HMJra878J8cH) | `news_sentiment.py`, `data/news_mirrors.py`, `data/sentiment.py` | `run_news_sentiment.py` |
| [Sentiment strategies: literature review and tests on our keepers](https://claude.ai/artifact/Ph7f95qxwcbCxpmbqQ2GLW) | `sentiment_overlay.py`, `data/sentiment_indices.py` | `run_sentiment_overlays.py` |
| [Crypto pairs trading](https://claude.ai/artifact/25jMx1tG5YpQwAzxrKfqcW) | `crypto_pairs.py` | `run_crypto_pairs.py` |
| Signal combination search (no write-up; results in the PR and below) | `combo_search.py` | `run_signal_combos.py` |

## Summaries

**Signal combination search.** Every strategy turned into a daily 0/1
signal or a $5k sleeve, then every subset combined: 785,508 SPY timing
trials (15 signals x AND/OR/majority/share x SPY/SSO/UPRO, flat or 20% vol
target), 3,962 BTC/ETH timing trials, 2.2M sleeve mixes (equal or
inverse-vol). Picked half-yearly walk-forward, Jul 2024+ held out. SPY
timing: walk-forward pick Sharpe -0.04 vs SPY 0.71, deflated Sharpe 0.02,
PBO 0.69. BTC/ETH: in-sample best (Sharpe 1.55) gives -0.23 in the holdout.
Sleeves without crypto (2012+): searched pick 0.81 walk-forward / 0.62
holdout, but holding all 16 sleeves equal-weight does better (0.88 / 0.75,
drawdowns -21% / -14% vs SPY -34% / -19%) at lower return (13% vs SPY
15-17%). Verdict: searching combinations finds luck (PBO 0.7-0.9); naive
diversification across sleeves is the only robust result, same as the $5k
fund plan.

**Non-financial prediction markets.** 131M Kalshi prints on 441k sports,
politics, weather, culture and "mentions" markets (2021 to Jan 2026) and 145M
Polymarket fills on 10k non-financial markets over $1M. Takers lose 1.4-2.6c
per contract after fees in every category; makers earn 0.6-2c of it.
Favourite/longshot entry rules, pre-game favourites on Polymarket and
momentum/reversion after 10c+ hourly jumps all lose after costs. Sum-to-$1
arbitrage (Kalshi team A + team B, temperature brackets, Polymarket
multi-outcome events) and Kalshi-vs-Polymarket arbitrage exist at 0.5-2c per
set, but both legs usually print within one second, so bots already take
them. The candidate: Kalshi leads Polymarket on sports games. When last
prices differ by 5c+, trading Polymarket toward Kalshi 60s late and holding
60 min made +5.6c/share (t 7.2, 1,943 trades over 1,009 games, 9 of 10 months
positive; $5k to ~$34k at 5% stakes capped at printed size). It needs
Polymarket's international book (no US persons), assumes no Polymarket fee,
and covers only Apr 2025 to Jan 2026, so a log-only watcher comes first.
Weather-vs-forecast was not testable (forecast archives blocked). A follow-up
matched 5,640 Kalshi/Polymarket pairs across nine families (US games, soccer
incl. draws, UFC, esports, tennis, championship futures, elections, awards,
Fed decisions): the lead is a live-event effect (US games +5.2c, soccer
+7.8c, UFC +32c on few trades, esports +7.6c per share at 60s late/60 min);
Fed, futures, elections and awards show none, and there the venues sit within
about 1.6c of each other. Flat $250 trades capped at printed size made ~$69k
over Apr 2025-Jan 2026 with at most $1.6k tied up (upper bound).
Josh can only trade Polymarket US, a separate book with no public history,
whose sports takers pay 5%·p(1-p). Net of that fee the lead is still +3.6c
(US games) and +6.4c (soccer) per share, ~$57k on the same flat trades.

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

**Neural nets for combining signals.** A literature review (Gu, Kelly and Xiu;
Deep Momentum Networks; the Momentum Transformer; Nagel's critique of
"virtue of complexity") and a fair contest: 20 weekly features taken from
the strategies here, purged expanding walk-forward folds, and equal weight,
ridge, LightGBM, an MLP and an LSTM traded with the same sizing and costs.
Neither net beat ridge and equal weight by the 0.2 Sharpe bar on ETFs
(2012 to mid-2024), crypto (2021 to mid-2024) or the S&P 100. On crypto and
stocks, staying long with the same sizing beat every model, and the existing
strategies (MultiAssetTrend, EnsembleTrend, top-10 momentum) matched or beat
them all. The locked holdout from July 2024 agreed. Weekly rank correlation
of the forecasts with next week's return was at most about 0.05. Adding nine
weekly Kalshi features (CPI, payrolls, Fed and recession contracts) did not
reliably help: forecast accuracy did not improve, other models moved either
way by up to 0.5 Sharpe, and the LSTM's gains (better than time-shuffled
Kalshi data in 4 of 6 checks) are a weak lead to recheck with more data.

**Machine learning on the research signals.** Four framings other than
forecasting weekly returns, tested walk-forward (refit yearly, purged) with
costs on $5,000 and scored once on the July 2024 holdout. Meta-labeling
worked on ETFs: a classifier that sees each 20/10 Donchian breakout on 32
ETFs, with the repo's signals at entry, and skips the worse half lifted
Sharpe from 0.61 to 0.76 (random-skip control 0.64 +- 0.10) and from 0.67 to
1.43 on the holdout (control 0.83 +- 0.28); kept trades beat skipped ones by
0.6 points each (t 3.6). It failed on 55 coins. A statistical jump model
(Shu, Yu and Mulvey 2024) cut SPY's drawdown but trailed the 200-day filter
and hurt the UPRO vol-target keeper. HAR volatility forecasts cut forecast
error 40% against trailing volatility and nudged the UPRO keeper's Sharpe
up (0.86 to 0.89; 0.68 to 0.75 on the holdout); LightGBM did worse than HAR.
Monthly S&P 100 ranking with ridge or LambdaMART beat momentum, but holding
today's most volatile members does nearly as well, so survivor bias explains
it. Reinforcement learning was ruled out for lack of data.

**Prediction markets as leading signals.** Hourly tests from 2024 to 2026
found Polymarket and Kalshi prices moving in the same hour as related assets,
not before them; for Fed-decision markets 2-year Treasury futures moved about
an hour first. The exception is a Polymarket Iran escalation index (322
markets): after a jump of 5 points or more while WTI futures traded, oil kept
going the jump's way for 3 hours, +0.66% per trade after costs over 34 trades
(t 2.4, 65% hits). The grid of thresholds and holds is positive almost
everywhere but rarely significant, entering an hour late halves it, and half
the trades come from the June 2025 war, so it is being paper traded with
`watch_polymarket_iran.py` rather than run with money. Jumps while oil was
closed were already in the reopening price (+1.6% gap).

**Short-term signal combination for 0DTE.** Hourly SPY/QQQ panel (Nov 2023
to Oct 2026) of intraday momentum, cross-asset, overnight BTC/Treasury,
VIX and Kalshi/Polymarket inputs, combined by equal weight, ridge and
LightGBM with monthly walk-forward refits and a May-Oct 2026 holdout. Rest-of-
day direction hit rates were 46-53% (51-57% on the most confident fifth),
never significantly above 50%, and the Kalshi/Polymarket inputs made models
worse. Buying ATM 0DTE options held to the close needs about 52% accuracy at
fair prices and 57% with a 10% volatility premium, so there is no trade.
Option prices are modelled (no free 0DTE history).

**Kalshi minute signals for 0DTE.** Kalshi's hourly S&P, Nasdaq and BTC
ladders lag the underlying at 1-10 minutes too: taker flow correlates 0.15
with the previous minute and about 0 with the next. Extreme signals are right
49-52% of the time, while a bought ATM 0DTE option needs 60-72% to break even;
the out-of-sample option proxy loses in every variant. `minute_mirrors.py`
loads free 1-minute SPX, MES, NQ and BTC bars from Hugging Face, which also
unblocks intraday tests without Alpaca keys.

**Prediction markets as pairs trades.** Kalshi ladders (S&P, Nasdaq-100, BTC,
ETH) turned into implied prices, Kalshi strike prices against their fair value
from spot, and Polymarket Iran and Fed indices against oil and Treasury
futures, traded with z-score pairs rules. The Kalshi spreads close in 4 to 8
minutes, but by Kalshi moving: the asset leg earns 0 to 2 bp gross (ETH about
4.5 bp over 30 minutes if entered within a minute), under costs, and fading
Kalshi contracts loses once fills come at the real bid and ask and the
0.07 x p x (1-p) fee is paid. Polymarket Fed pairs fail; Iran vs oil only
repeats the jump rule. No pair is tradeable on $5k; maker orders on Kalshi
would need live order-book data to test.

**Trading Kalshi off stock moves.** Kalshi's hourly S&P, Nasdaq, BTC and ETH
ladders lag the underlying at the hourly scale, but at 1-minute resolution
their prints already price in the last minute's move. Buying the contract
whose last ask had not caught up after a sharp move (Oct 2024 to Jan 2026,
fees, bid/ask from trade prints, fills capped at printed size) lost on BTC
(t about -3 over ~1,800 fills: bots take it), was noise on ETH and the
Nasdaq-100, and made about $3,000 on the S&P 500 after 2-minute moves
(t 2.0, positive in both halves) on small fills, median 80 contracts. Fed
contracts traded off 2-year Treasury futures moves had no edge after costs.

**Kalshi crowd bias after big moves.** Kalshi alerts users when BTC, ETH or
the indexes trend hard; with no record of when, those moments were proxied by
big 15-minute to 1-day moves and by Kalshi volume spikes (Oct 2024 to Jan
2026, hourly S&P, Nasdaq-100, BTC and ETH ladders). Continuation sides were
not overpriced afterwards (15 of 16 cells within 2 standard errors), taker
flow stayed about 50/50, and 240 fade/follow variants looked like the
random-time control; maker variants lost to adverse selection. No edge.

**Intraday strategies on 1-minute data.** Free 1-minute S&P, Nasdaq, BTC and
ETH bars (2021-2026) let the intraday rules run without Alpaca. 3,448 rule
variants (noise area, opening range, VWAP trend and reversion, end-of-day
momentum, gaps, crypto hour of day, overnight split) at four cost levels:
trend rules are profitable before costs in 80-100% of variants, but earn only
about 0.5-1.5 bp per trade on S&P and 1-4 bp on BTC, below ETF (1.5 bp) and
spot crypto (10-25 bp) costs. Walk-forward picks lose; the one survivor, QQQ
15-minute opening range (+10% in 2025), lost 7%/yr in the Jan-Oct 2026 holdout.

**Long-term fundamental portfolio manager.** A 60/20/20 allocator: S&P 500
stocks picked monthly on SEC fundamentals (point-in-time membership, frames
data public 60-90 days after period end, sector-relative ranks, 25 names,
sector cap), EFA+EEM and IJR, a growth-trend macro switch (rising
unemployment and SPY below its 200-day average -> Treasuries or T-bills),
200-day trend on the non-US sleeves and an 18% vol cap. Quality + low vol
was the best stock factor mix on 2007-2018 (14.8%/yr vs SPY 13.2%, -14% vs
-19% drawdown in 2010-2018), and the full manager made 11.0%/yr with a -17%
drawdown vs SPY 6.5% and -55% in 2007-2018, almost all from sitting out 2008.
The untouched 2019-2026 holdout failed: 11.1%/yr vs SPY 17.4%, drawdown -35%
vs -34%. Quality + low vol lagged mega-cap growth (equal-weight RSP also
trailed SPY by 3.7%/yr), international and small caps lagged, and the macro
switch reacted after the March 2020 crash. Walk-forward of the factor mix
(2014-2026): 9.4%/yr vs 13.6%. A SPY core and a 2x ETF in calm regimes
didn't fix it; adding a credit-spread signal held the holdout drawdown to
-19% but at 9.8%/yr. Survivorship bias of Yahoo's universe: equal weight of
the members it still has beat RSP by about 1.2%/yr in 2010-2018 (72-86% of
members covered) and by 0% in 2019-2026 (89-99%). Verdict: no edge over SPY.
Tilting to big tech (tested after seeing the holdout): with QQQ as the US
sleeve at 80/10/10 it made 16.0%/yr vs SPY 17.4% with a -27% vs -34%
drawdown in 2019-2026, and 14.6% vs 10.7% (-27% vs -55%) over 2007-2026;
that is a bet that Nasdaq keeps beating the S&P, as it has since 2007.
Mega-cap fundamental picks worked in 2019-2026 but not in 2010-2018.
Josh picked the 80/10/10 QQQ version as the class default; the stock picker
is `stock_etf=None` with 60/20/20 sleeves.

**Kalshi S&P range vs VIX1D for selling 0DTE.** Kalshi's end-of-day S&P
ladders imply an expected range for the close. Every return model on 29
daily Kalshi features (lasso, elastic net, random forest, gradient
boosting, nearest neighbours, logistic, small MLP) lost to predicting zero,
but the ladder width forecasts rest-of-day volatility about as well as
VIX1D. At 12:30 and 14:30 ET (not 10:30), days when Kalshi's range is
narrow vs VIX1D (bottom third of its own history) let a volatility seller
keep 62-67% of VIX1D-implied variance, vs 45-46% on wide days (SPX
1-minute bars, Jan 2024 to Jan 2026). As modelled XSP iron flies at
break-even option prices, the 12:30 rule made +13.5% on risk per trade
(t 2.5, 165 trades) while selling every day lost 4%; with a 10% variance
premium it made +21% (t 3.6, both halves positive, max drawdown -28% at 10%
risk per fly). Option prices are modelled, so this needs real 0DTE quotes or
paper trading before any money (`backtester/kalshi_vol.py`,
`examples/run_kalshi_vol.py`).

**FinBERT news and Reddit sentiment.** FinBERT on 282k Benzinga headlines
(2009-2020) finds a real but small effect: stocks with very positive pre-open
news beat very negative ones by 0.19% open-to-close (t 3.9, holdout too). Most
of the news is already in the opening gap, and two legs of costs eat the rest,
so it's no edge at $5k. Daily news mood and Reddit mood (4.0M posts from
r/wallstreetbets & co., 2021-2025) don't predict SPY or BTC, and trading Kalshi
S&P/BTC "closes above" contracts on them loses 0.3-9.5 cents per contract after
fees. One candidate: short the stock at the open when its WSB mentions jump 10x
or more, cover at the close. That made +1.8% per trade in 2021-22 and +2.2% in
2023-25 (57-58% wins, t 1.6-2.0), with single squeezes of -60 to -70%. It
needs borrow checks and paper trading (`backtester/news_sentiment.py`,
`examples/run_news_sentiment.py`).

**Sentiment strategies.** Literature review of survey and composite indices
(Baker-Wurgler, AAII, aligned index), VIX/options, news text and LLM scores,
social media and retail attention, and crypto sentiment: slow measures
forecast years, fast text forecasts days but sits in small caps and decays as
tools spread. Tests (surveys blocked, so a Fear & Greed style mood index was
rebuilt from Yahoo): extreme fear precedes the best SPY month but with ~1.8x
the volatility; no fear/greed, VIX-backwardation, credit, Michigan or crypto
funding overlay beat UPRO vol target, multi-asset trend, BTC/ETH breakout or
ensemble trend in both design and holdout. VIX > VIX3M exit helps plain SPY
only. Mood at entry doesn't sort ETF breakout trades. The FinBERT Benzinga
market news mood (2009-Jun 2020) mostly echoes yesterday's return; trimming
after upbeat news weeks added only 0.02-0.05 Sharpe. Reddit mood
(2021-25) overlays moved Sharpe by under 0.1 either way. No edge.

**Crypto pairs trading.** BTC/ETH z-score mean reversion loses even at 7 bp
futures fees (median Sharpe -0.65 in 2018-23, -0.11 in 2024+): the ETH/BTC
ratio trends. Ratio momentum is positive in design but fades after 2024, and
the long-only rotation trails buying BTC (14% vs 28%/yr since 2024). Rolling
cointegration/distance pairs over 19 coins lose after costs (walk-forward
-10%/yr, -77% DD). Intraday 5m-1h pairs earn 0-4 bp gross per round trip vs a
14 bp futures round trip. No edge; US spot accounts can't short alts anyway.
