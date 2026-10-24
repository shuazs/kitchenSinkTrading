# Paper loggers

One log-only forward test per candidate on the
[paper-trading list](../docs/guide/paper-trading.md). None of them place
orders: they read public market data, print what the rule would do, and append
paper trades to CSV logs. Run them on your own computer (Kalshi, Polymarket,
Alpaca and TreasuryDirect are blocked from the project's cloud sandbox).

## Setup

```bash
pip install -e ".[ml,polymarket]"                    # from the repo root
cp paper_loggers/keys.example.env paper_loggers/keys.env
# fill in SEC_USER_AGENT in keys.env (keys.env is gitignored)
```

## Loggers

| Logger | Candidate | Run | Graduates after |
| --- | --- | --- | --- |
| `treasury_auction_cycle.py` | Treasury auction cycle (PASS) | daily after 4pm ET; prints the MOC action for today | 6 cycles at TLT - BIL <= -18 bp |
| `etf_breakout_meta.py` | ETF breakout + meta-label filter | daily after the close | 3 months, filter Sharpe gain >= 0.08 |
| `commodity_january.py` | January commodity window | daily (only acts in January) | Jan 2027 and Jan 2028 >= +1.2% net |
| `odd_lot_tenders.py` | Odd-lot fixed-price tender offers | daily; needs `SEC_USER_AGENT` | 6 events at >= +3% |
| `cef_rights_short.py` | CEF rights-offering short | daily; needs `SEC_USER_AGENT`; fill `borrow_pct` | 10 events at >= +1% net |
| `letf_reversal.py` | Single-stock LETF rebalancing reversal | daily after 4pm ET | 40 trades at >= +0.5% vs SPY |
| `kalshi_near_misses.py` | Kalshi mentions No 20-80c, ATP favourites Yes 80-95c | always on (polls every 15 min) | 200 events at half the backtest edge |
| `polymarket_iran_oil.py` | Polymarket Iran odds -> oil (MCL) | always on (polls every minute) | 15 signals at >= +0.33% |
| `kalshi_sp_catchup.py` | Kalshi S&P hourly after a sharp 2-minute move | always on in market hours (polls every 20 s) | 3 months at half the backtest P&L rate |

`run_daily.py` runs the six daily loggers in one go (`--report` prints all
their scorecards). Each logger also takes `--report`; the always-on ones take
`--once` for a quick look. Example cron line (Mon-Fri 4:30pm New York time):

```
30 16 * * 1-5 cd ~/kitchenSinkTrading && python paper_loggers/run_daily.py >> ~/.cache/kitchensink/paper/daily.log 2>&1
```

Logs live in `~/.cache/kitchensink/paper/<logger>/` (`KS_CACHE_DIR` moves the
root). The two older loggers keep their original paths:
`~/.cache/kitchensink/kalshi_near_misses/` and
`~/.cache/kitchensink/polymarket/iran_signals.csv`.

Shared helpers are in `_common.py` (keys.env loading, NYSE calendar, Yahoo
closes, EDGAR requests, CSV logs).

## Retired (do not run)

`retired/kalshi_polymarket_us.py` (sports lead, retracted in PR #36) and
`retired/wsb_spikes.py` (WSB fade, failed its holdout) are kept for reference
only; see the Retired table in the paper-trading list.
