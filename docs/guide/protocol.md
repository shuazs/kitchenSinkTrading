# Alpha loop validation protocol (v1, 2026-10-23)

A strategy "holds up out of sample" only if it passes ALL of these. Write the
hypothesis, parameters and the variant count down BEFORE running.

P1 Costs: repo `COSTS` presets or stricter. ETFs >= 3 bp per side plus spread;
   crypto >= 10 bp per side; Kalshi taker 0.07*p*(1-p) per contract, maker
   0.0175*p*(1-p); Polymarket US sports taker 5%*p*(1-p). Short borrow on
   inverse ETFs is in their returns.
P2 No look-ahead: signal at close t, trade at open t+1. Market-on-close at t
   is allowed only for dates known in advance (auctions, calendars).
P3 Locked holdout: Jul 2024 onward. Parameters are chosen on data before
   Jul 2024 only. The holdout is run once and reported as is.
P4 Holdout result: same sign as in-sample, >= half the in-sample edge
   (per trade / per event / excess return), Sharpe > 0 after costs, AND either
   beats its stated benchmark on Sharpe or adds Sharpe to the Balanced fund
   plan (fund-plan/) as a 10-20% sleeve.
P5 Walk-forward / stability: in-sample split into >= 3 sub-periods; edge
   positive in at least 2/3 of them.
P6 Multiple testing: report N variants tried. In-sample t >= 2, and when
   N > 10 the deflated Sharpe probability (`backtester.ml.contest.deflated_sharpe`)
   must be >= 0.9 (or t >= 3 for event studies).
P7 Robustness: neighbouring parameters (about +-50%) keep >= half the edge;
   a placebo (shifted dates, shuffled signal or lagged entry) shows none.
P8 Retail-feasible at $5,000: instruments a US retail account can trade
   (Josh is US-only: Kalshi and Polymarket US, no Polymarket international),
   realistic fills, capacity at least $5k.

Verdicts: PASS (all of P1-P8), NEAR (fails one of P5-P7 narrowly; keep
paper-trading), FAIL.
Holdout caveat: for strategies found in earlier PRs, the Jul 2024+ period
was often seen during development. Say so; only paper trading is truly OOS.
