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
| [Crypto strategies and meme coin sentiment](https://claude.ai/artifact/G2XAGWzHpp7rkDpTdA8FzM) | `strategies/crypto.py`, `memecoin_calls.py`, `data/memecoins.py` | `run_crypto_strategies.py`, `run_memecoin_calls.py` |
| [Large-account strategies scaled to $5k](https://claude.ai/artifact/JKf6qVGf7N7Ep47XbSixYL) | `strategies/carry.py`, `basis.py`, `stock_momentum.py`, `data/fx.py` | `run_fx_carry.py`, `run_basis_carry.py`, `run_hedged_momentum.py` |

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
