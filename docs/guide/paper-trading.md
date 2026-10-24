# Paper-trading list

Every strategy that has to prove itself on live data before real money, in
one place. Nothing here is traded with real money yet.

**Graduation rule** (from the $5k fund plan): a candidate stays at $0 until
its live log keeps at least half the backtest edge after costs. For frequent
rules that means about 3 months of logs; for annual or monthly ones, the
event count shown. A candidate that graduates gets at most $250.
Pass rules for the backtests are in [protocol.md](protocol.md).

## Active

| Candidate | Status | Rule | How to log | Edge to beat (half of backtest) | Graduates after | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Treasury auction cycle | PASS (only one so far) | Out of TLT (or short via TBF) from the close 5 trading days before each month's 10y/30y auctions to the close of the last auction day; MOC orders | `auction_cycle.py` `simulate`; check auction dates by hand (treasurydirect.gov is blocked here) | TLT minus BIL over the window <= -18 bp per cycle | 6 cycles | PR #33, #36 |
| ETF breakout + meta-label filter | NEAR | 20/10 breakout on 32 ETFs, skip trades the LightGBM filter ranks in the worse half | `run_ml_research.py`, log each signal and the filter's call | Sharpe gain of the filter over plain breakout >= 0.08 | 3 months | PR #16, #36 |
| Polymarket Iran odds -> oil | paper | Iran-war odds jump >= 5 pts/hour -> long MCL for 3 hours | `watch_polymarket_iran.py` on Josh's machine; log in `~/.cache/kitchensink/polymarket/iran_signals.csv` | +0.33% per trade | 15 signals | PR #12 |
| Kalshi mention markets, buy No 20-80c | NEAR | Taker No on mention markets priced 20-80c, 6 h to 7 days before close | `python examples/watch_kalshi_near_misses.py` on Josh's machine (`--once` lists entries, `--report` scores settled trades); log `~/.cache/kitchensink/kalshi_near_misses/trades.csv` (+ `quotes.csv`); rule `mentions_no` | +1.7c per contract after fees | 200 events | PR #36, #37, #41 |
| Kalshi ATP favourites, Yes 80-95c | NEAR | Taker Yes on the ATP match favourite at 80-95c (WTA did not confirm) | Same watcher, rule `atp_yes` (one taker buy per market at the ask) | +1.3c per contract after fees | 200 events | PR #37, #41 |
| January commodity window | NEAR, post-hoc | Long last year's top-third commodities, short the bottom third, close of January business day 5 to close of day 10 (ETF basket: USO UNG GLD SLV CPER CORN WEAT SOYB CANE) | Paper trade Jan 2027, by hand or `run_alpha_batch6.py` | +1.2% net for the 5 days | Jan 2027 and 2028 | PR #38, #39 |
| Kalshi S&P hourly after a sharp 2-minute move | maybe-edge, needs a watcher | Buy the S&P hourly contract in the direction of a sharp 2-minute index move | Not built yet (would be a log-only watcher on Josh's machine) | half of +$3k per 14 months, tiny fills | 3 months | PR #17 |

## Retired (do not run)

| Candidate | Why | Source |
| --- | --- | --- |
| Kalshi -> Polymarket US sports lead (`watch_kalshi_polymarket_us.py`) | Retracted: an exit-selection bug made it look positive; with causal exits it loses 2.3c a share | PR #36 |
| WSB mention-spike fade (`watch_wsb_spikes.py`) | Failed its holdout (t 0.7); the edge was only in hard-to-borrow names | PR #36 |
