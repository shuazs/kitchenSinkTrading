# Data

Loaders in `backtester/data/`. Downloads cache under `~/.cache/kitchensink`
(`KS_CACHE_DIR` overrides). Load only what the task needs; most are slow on
first call.

## Env vars

- `SEC_USER_AGENT="name email"`: required by every EDGAR call.
- `KS_MINUTE_DIR`: folder of ready-made minute parquet files (default `/mnt/project-files/minute-data`).
- `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`: optional Alpaca paper keys (minute bars from 2016).

## Loaders

| File | Main functions | Source |
| --- | --- | --- |
| `yahoo.py` | `load_bars(sym,start,end,interval)`, `load_market(symbols,start,**extra) -> MarketData`, `load_hourly` (~730 days of 1h), `option_chain` (today only), `headlines`, `append_headlines` | yfinance |
| `yahoo_batch.py` | `prefetch(symbols)` | yfinance, many tickers |
| `fred.py` | `load_series(id)`, `tbill_rate()` (for `cash_rate`), `unemployment_rate()` (HF mirror fallback, ends Feb 2026) | FRED CSV, no key |
| `fundamentals.py` | `market_data()` -> MarketData for `FundamentalAllocator`; `panel()` (EDGAR frames, all filers, 60-90 day lag), `as_of`, `sectors` (SIC), `load_bars` (total-return OHLC + `raw_close`), `sp500_universe` (renames mapped), `coverage` | data.sec.gov frames + Yahoo; first run ~15 min |
| `fx.py` | `load_fx_market(ccys)`, `total_return_index`, `known_rate_gap` | Yahoo FX + FRED rates |
| `edgar.py` | `company_facts`, `fact_series`, `earnings_surprise` (SUE), `earnings_dates` (8-K 2.02), `insider_trades`, `parse_form4` | data.sec.gov |
| `insider_bulk.py` | `open_market_trades(start,end)`, `attach_current_tickers`, `add_market_cap` | SEC Form 3/4/5 quarterly sets |
| `thirteenf.py` | `load_holdings(since)`, `cusip_tickers` | SEC 13F data sets |
| `activists.py` | `events(since)` | EDGAR form index 13D |
| `sp500.py` | `membership()`, `changes(since)` | S&P 500 history (GitHub) |
| `sentiment.py` | `score(texts)`, `daily_sentiment` | FinBERT (huggingface.co) |
| `news_mirrors.py` | `load_benzinga()` (1.57M stock-tagged headlines 2009-02..2020-06; 685k with minute times), `reddit_month("2021-01")` (streams a Pushshift month, keeps finance subs; 7-14 min/month, needs `zstandard`) | HF mirrors `idleengine/financial-news-multisource`, `Avni2110/pushshift-reddit` |
| `sentiment_indices.py` | `mood_index` (Fear & Greed style, 0-100, from SPY/VIX/VIX3M/HYG/IEF/TLT), `mood_components`, `consumer_sentiment` (FRED UMCSENT, lagged), `funding_rates(coin)` (Binance BTC/ETH perp, May 2021-Feb 2026) | Yahoo, FRED, HF `Torch-Trade/*_perp_funding_8h_*` |
| `kalshi.py` | `load_daily_features(path)`, `weekly_features` | local daily file from Kalshi research |
| `minute_mirrors.py` | `load_minute(sym)`, `rth_bars(sym)` | HF mirrors / `KS_MINUTE_DIR`; SPX, MES, NQ, BTC (+QQQ, ETH, NDX local) |
| `alpaca.py` | `load_minute_bars`, `load_yahoo_intraday` (last 60 days) | Alpaca (blocked from cloud) |
| `kalshi_hf.py` | `load_markets()` (every market + result), `load_trades(tickers, name)` | HF `TrevorJS/kalshi-trades`, scans 16 shards |
| `polymarket.py` | `load_markets`, `load_fills` (`side=True` keeps taker side), `hourly_prices`, `escalation_index`, `iran_markets`, `live_markets` | HF snapshot; live API needs local machine |
| `memecoins.py` | `load_calls`, `load_launches` | HF pump.fun datasets |
| `trump_posts.py` | `load_posts`, `daily_counts(posts,index,pattern)`, `covered` | GitHub archives |
| `gov_stakes.py` | `stake_events(index)` | hand-built table |
| `cef.py` | `universe(candidates)`, `load_cef_market(tickers)`, `cef_market_from_frames` (NAV in `extra["nav"]`) | Yahoo, NAV tickers X<ticker>X (live funds only) |
| `edgar_forms.py` | `form_index(forms, start_year, end_year)`, `filer_status(cik, since)`, `classify_filings` (acquired / bankrupt) | EDGAR full-index form.idx + submissions (caches whole ~50 MB files; `examples/run_alpha_batch5.py` `form_counts` streams them and caches per-filer counts in `edgar_forms/`) |
| `fomc.py` | `meetings()`, `statement_dates()` | HF `ZipLime/fomc-events` |
| `treasury_auctions.py` | `load_auctions()`, `long_bond_auction_dates()` (10y/30y incl. reopenings, 2008+) | HF `Farmaanaa/us_treasury_auctions` |
| `cot.py` | `load_pit`, `weekly_panel` (dated by publication, `known`), `positioning_signals`, `to_trading_days`, `ETF_MAP` (20 COT markets -> ETFs) | HF `ZipLime/commitments-of-traders` (2010+, real release times incl. shutdowns) |
| `congress.py` | `load_trades` (PTR stock trades, `filing_date` = public day), `buy_events` (per ticker/day: `n_members`, `amount_low`, `members_60d`, `leader`), `LEADERS` | HF `ZipLime/congress-trading` (House 2014+, Senate thin before 2023) |
| `synthetic.py` | `synthetic_market(symbols, seed, **kw)` | random walk, for tests |
| `live_books.py` | `kalshi_games`, `kalshi_book`, `poly_us_markets`, `poly_us_book`, `pair_games`, `gap_followthrough` (replay, after the 5%·p(1-p) US fee) | live Kalshi + Polymarket US (gateway.polymarket.us) books, keyless; blocked from cloud |

## Shared datasets (`/mnt/project-files/`, cloud threads only)

| Folder | Contents |
| --- | --- |
| `minute-data/` | 1-min parquet, UTC bar-start index: SPX 2021-06..2026-10 (RTH), MES 2021-06.. (back-adjusted, use returns), NQ 2024-01..2025-07, BTC 2021-06..2026-10, QQQ, ETH 2024-12..2025-11, NDX (time known), `kalshi_implied_1m`. See its README. |
| `crypto-pairs/` | `daily_close.parquet` (yfinance closes, 19 coins, 2017-11+), `bars_5m.parquet` (Binance spot BTC/ETH/SOL/BNB/XRP/ADA 5-min OHLCV, 2021-05..2026-02, from HF `Torch-Trade/<coin>usdt_spot_1m_05_2021_to_03_2026`, 1-min there), `results/` |
| `prediction-markets/` | Kalshi target markets/trades (Fed, CPI, recession...), Polymarket targets + fills since Oct 2024, hourly Iran/Fed theme indices, yfinance 1h prices |
| `kalshi-leading/`, `kalshi-catchup/`, `kalshi-crowd/`, `kalshi-pairs/`, `kalshi-0dte/` | Kalshi trades/prints per series (KXINXU, KXNASDAQ100U, KXBTCD, KXETHD), derived features, result CSVs, scratch scripts; `kalshi-leading/data/` also has `sel_trades.parquet`/`sel_markets.parquet` (55 finance series, 2021-2026), `vix1d_1h.parquet` and `kalshi_vol_panel.parquet` for `run_kalshi_vol.py` |
| `event-markets/` | Non-financial Kalshi + Polymarket: `nf_markets.parquet` (441k Kalshi markets with `cat`), `pm_nf_markets.parquet` (10k Polymarket markets >$1M), `raw/nf_trades_*.parquet` (131M Kalshi prints) and `raw/pm_nf_fills.parquet` (145M fills with side, 2024+), `xvenue_*` (1,104 Kalshi/Polymarket matched games), `xv2_*` (5,640 matched pairs across 9 market families with prints, fills and results), result parquet files, `scripts/` |
| `cot-data/` | COT point-in-time tables (`pit`, `positions`, `markets`, `releases` parquet) and the write-up HTML |
| `congress-data/` | Congress trades and reference tables (parquet), `results/` CSVs, write-up HTML |
| `shortterm-combo/` | hourly SPY/QQQ panels, Kalshi 30-min implied closes, walk-forward and holdout results |
| `finbert-sentiment/` | FinBERT-scored Benzinga headlines (`benzinga_scores`), Reddit posts 2021-2025 with tickers + scores (`reddit_posts`), daily mood series (`benzinga_daily_mood`, `reddit_daily_mood`), Kalshi nearest-strike books (`kalshi_book_spx/btc`), event tables, scratch scripts |
| `signal-combos/` | `run_signal_combos.py` output: top-2,000 boards, walk-forward picks and summaries per search, sleeve equity curves |
| `intraday-lab/`, `ml-research/` | result tables for those studies |
| `alpha-hunt/` | alpha hunt outputs: `kalshi_maker_output.txt`/`kalshi_maker_agg.parquet`, `treasury_auctions.parquet`, `sue.parquet` (SUE from SEC Financial Statement Data Sets 2009-2026), `pead_events.parquet`, `gov_award_events.parquet` (HF ZipLime/us-government-contracts), spin-off tables, `scripts/` |
| `sentiment-research/results/` | mood index and sentiment overlay result CSVs from `run_sentiment_overlays.py` |

Kalshi history mirror: HF `TrevorJS/kalshi-trades` (Jun 2021 - Jan 2026).
Polymarket: HF `SII-WANGZJ/Polymarket_data`.
Other HF finance sources surveyed 2026-10-22 (untested): `ZipLime/us-government-contracts`, `Arimancy/cftc-cot-weekly` (COT legacy from 1986, no release times), `XOOMAR/sec-fails-to-deliver`; short-interest mirrors have no usable history.

## Network from the cloud environment

Reachable: Yahoo, FRED (was in maintenance 2026-10-21/22), SEC EDGAR, huggingface.co, PyPI, raw.githubusercontent.com.
Blocked: treasurydirect.gov, fiscaldata.treasury.gov, BLS, OECD, dbnomics, Alpaca, Polymarket and Kalshi live APIs, cboe.com, CNN, trumpstruth.org,
and every sentiment survey source (aaii.com, NYU Baker-Wurgler, alternative.me, nasdaq data, Binance/Bybit/OKX APIs).
No free delisted stock prices exist on Yahoo or HF (checked 2026-10-23), so single-stock event tests stay survivor-biased; bound them with `spinoffs.survivorship_bound`.
Live watchers (e.g. `watch_polymarket_iran.py`) must run on Josh's machine.
