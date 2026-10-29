# Paper-trading list

Every strategy that has to prove itself on live data before real money, in
one place. Nothing here is traded with real money yet. Each candidate has a
log-only logger in [`paper_loggers/`](../../paper_loggers/README.md) that runs
on Josh's machine (keys in `paper_loggers/keys.env`, template
`keys.example.env`); `python paper_loggers/run_daily.py` runs the daily ones.

**Graduation rule** (from the $5k fund plan): a candidate stays at $0 until
its live log keeps at least half the backtest edge after costs. For frequent
rules that means about 3 months of logs; for annual or monthly ones, the
event count shown. A candidate that graduates gets at most $250.
Pass rules for the backtests are in [protocol.md](protocol.md).

## Active

| Candidate | Status | Rule | Logger | Edge to beat (half of backtest) | Graduates after | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Treasury auction cycle | PASS (only one so far) | Out of TLT (or short via TBF) from the close 5 trading days before each month's 10y/30y auctions to the close of the last auction day; MOC orders | `treasury_auction_cycle.py` (daily; prints today's MOC action) | TLT minus BIL over the window <= -18 bp per cycle | 6 cycles | PR #33, #36 |
| ETF breakout + meta-label filter | NEAR | 20/10 breakout on 32 ETFs, skip trades the LightGBM filter ranks in the worse half | `etf_breakout_meta.py` (daily; logs every breakout with the filter's call) | Sharpe gain of the filter over plain breakout >= 0.08 | 3 months | PR #16, #36 |
| Polymarket Iran odds -> oil | paper | Iran-war odds jump >= 5 pts/hour -> long MCL for 3 hours | `polymarket_iran_oil.py` (always on) | +0.33% per trade | 15 signals | PR #12 |
| Kalshi mention markets, buy No 20-80c | NEAR | Taker No on mention markets priced 20-80c, 6 h to 7 days before close | `kalshi_near_misses.py`, rule `mentions_no` (always on); logs minutes since the market opened | +1.7c per contract after fees | 200 events | PR #36, #37, #41; batches 11-12, note 1 |
| Kalshi mentions No, transformer filter | NEAR, weak | The `mentions_no` entries, kept only when the fine-tuned MiniLM (`run_mention_transformer.py`, text + word history + price) puts the No edge in the top half | no new logger: score the `mentions_no` trades.csv offline with the model (it logs ticker, title and price) | top half beats bottom half by >= 4c per contract | 200 events | this PR; note 5 |
| Kalshi ATP favourites, Yes 80-95c | NEAR, weaker | Taker Yes on the ATP match favourite at 80-95c (WTA did not confirm) | same logger, rule `atp_yes` | +1.3c per contract after fees | 200 events | PR #37, #41; note 2 |
| January commodity window | NEAR, post-hoc | Long last year's top-third commodities, short the bottom third, close of January business day 5 to close of day 10 (ETF basket: USO UNG GLD SLV CPER CORN WEAT SOYB CANE) | `commodity_january.py` (daily; acts only in January) | +1.2% net for the 5 days | Jan 2027 and 2028 | PR #38, #39; note 3 |
| Odd-lot fixed-price tender offers | NEAR | Buy 99 shares at the close after a company SC TO-I with odd-lot priority and a fixed price above the market (no Dutch auctions, no funds); tender everything at a broker with no tender fee | `odd_lot_tenders.py` (daily; EDGAR full-text search, needs `SEC_USER_AGENT`) | +3% per event after fees | 6 events | batch 8 |
| CEF rights offering short | NEAR | Short a closed-end fund vs a CEF basket (PCEF live) for 25 trading days from the close after its rights-offering prospectus (497/424B*) | `cef_rights_short.py` (daily; fill in `borrow_pct`) | +1% per event net of spreads and borrow | 10 events | batch 8 |
| Single-stock LETF rebalancing reversal | NEAR, post-hoc | After a day 2 sd or more down in a stock with 2x single-stock ETFs when LETF pressure (sum \|L(L-1)\| x LETF 20d $ volume / stock 20d $ volume) > 0.02, buy at the next open, sell at that close | `letf_reversal.py` (daily; every name in `backtester/letf.py` `ALL_LETF`, logs pressure) | +0.5% per trade vs SPY after costs | 40 trades (~1 year) | batch 10; note 4 |
| Kalshi S&P hourly after a sharp 2-minute move | maybe-edge | Buy the KXINXU hourly contract in the direction of a 2.5-sigma 2-minute S&P move when the ask is 3c+ under fair value | `kalshi_sp_catchup.py` (always on in market hours; logs Yahoo data lag) | half of +$3k per 14 months, tiny fills | 3 months | PR #17 |

Notes from later batches:

1. Mentions No: the Polymarket check (batch 11) did not confirm (-0.6c net,
   clustered by event); No at 20-40c is only +3.0c (t 1.7) there and +1.9c
   (t 2.1) on Kalshi at real taker-No prices (batch 12). The first-taker-No
   version is only +0.5c if the market's first hour is skipped, so most of the
   edge is early entries. Score only taker prints of the side bought: a print
   on the other side understates the cost by the spread.
2. ATP favourites: Polymarket tennis favourites lose 0.9c net over 9.5k
   matches (batch 11). Keep logging.
3. January window: batch 11 re-check +2.4% (t 4.5), 2024-26 all positive,
   absent in country/sector ETFs and crypto; batch 12: index members +1.5%
   (t 2.4) vs non-members +0.1%, none before 2000 in FRED energy.
4. LETF reversal: top-decile pressure (>= 0.154) +1.3%/trade in the holdout
   but t 1.5 (batch 11), so `--report` scores > 0.02 and the top decile
   separately; batch 12: 35 newer 2x names +0.55%/trade (t 1.5, 169 events).
5. Mentions transformer filter (walk-forward, entry 6 h after listing at real
   taker prints): top half +6.4c Aug-Oct 2025 (event t 1.5) and +8.7c on the
   Nov-Jan holdout (t 2.7), bottom half ~0, but within one event its picks beat
   its rejects by only 2-4c (t ~1), so it mostly picks events. Same setup gives
   the plain rule ~0 by event. Three seeds agree: top half +6.1 to +7.2c
   Aug-Oct (t 1.5-3.1), +8.7 to +9.4c holdout (t 2.7-3.1), bottom half ~0.

## Not yet on paper

- Kalshi S&P range vs VIX1D -> sell XSP 0DTE iron flies (`kalshi_vol.py`,
  research-modules.md): modelled option prices, so it needs real XSP quotes
  before a logger is worth building.
- Kalshi 30-year mortgage-rate markets vs a Treasury nowcast (`run_kalshi_mortgage_nowcast.py`): NEAR (+6.6c in-sample t 2.1, +11.8c on 11 holdout weeks), but Kalshi has not listed FRM since Feb 2024; build a logger only if the series comes back.

## Retired (do not run)

| Candidate | Why | Source |
| --- | --- | --- |
| Kalshi -> Polymarket US sports lead (`paper_loggers/retired/kalshi_polymarket_us.py`) | Retracted: an exit-selection bug made it look positive; with causal exits it loses 2.3c a share | PR #36 |
| WSB mention-spike fade (`paper_loggers/retired/wsb_spikes.py`) | Failed its holdout (t 0.7); the edge was only in hard-to-borrow names | PR #36 |
