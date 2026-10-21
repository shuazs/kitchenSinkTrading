# Research write-ups

The research behind this repo lives in five project docs on claude.ai (private
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
