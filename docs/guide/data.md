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
| `fred.py` | `load_series(id)`, `tbill_rate()` (for `cash_rate`) | FRED CSV, no key |
| `fx.py` | `load_fx_market(ccys)`, `total_return_index`, `known_rate_gap` | Yahoo FX + FRED rates |
| `edgar.py` | `company_facts`, `fact_series`, `earnings_surprise` (SUE), `earnings_dates` (8-K 2.02), `insider_trades`, `parse_form4` | data.sec.gov |
| `insider_bulk.py` | `open_market_trades(start,end)`, `attach_current_tickers`, `add_market_cap` | SEC Form 3/4/5 quarterly sets |
| `thirteenf.py` | `load_holdings(since)`, `cusip_tickers` | SEC 13F data sets |
| `activists.py` | `events(since)` | EDGAR form index 13D |
| `sp500.py` | `membership()`, `changes(since)` | S&P 500 history (GitHub) |
| `sentiment.py` | `score(texts)`, `daily_sentiment` | FinBERT (huggingface.co) |
| `sentiment_indices.py` | `mood_index` (Fear & Greed style, 0-100, from SPY/VIX/VIX3M/HYG/IEF/TLT), `mood_components`, `consumer_sentiment` (FRED UMCSENT, lagged), `funding_rates(coin)` (Binance BTC/ETH perp, May 2021-Feb 2026) | Yahoo, FRED, HF `Torch-Trade/*_perp_funding_8h_*` |
| `kalshi.py` | `load_daily_features(path)`, `weekly_features` | local daily file from Kalshi research |
| `minute_mirrors.py` | `load_minute(sym)`, `rth_bars(sym)` | HF mirrors / `KS_MINUTE_DIR`; SPX, MES, NQ, BTC (+QQQ, ETH, NDX local) |
| `alpaca.py` | `load_minute_bars`, `load_yahoo_intraday` (last 60 days) | Alpaca (blocked from cloud) |
| `polymarket.py` | `load_markets`, `load_fills`, `hourly_prices`, `escalation_index`, `iran_markets`, `live_markets` | HF snapshot; live API needs local machine |
| `memecoins.py` | `load_calls`, `load_launches` | HF pump.fun datasets |
| `trump_posts.py` | `load_posts`, `daily_counts(posts,index,pattern)`, `covered` | GitHub archives |
| `gov_stakes.py` | `stake_events(index)` | hand-built table |
| `synthetic.py` | `synthetic_market(symbols, seed, **kw)` | random walk, for tests |

## Shared datasets (`/mnt/project-files/`, cloud threads only)

| Folder | Contents |
| --- | --- |
| `minute-data/` | 1-min parquet, UTC bar-start index: SPX 2021-06..2026-10 (RTH), MES 2021-06.. (back-adjusted, use returns), NQ 2024-01..2025-07, BTC 2021-06..2026-10, QQQ, ETH 2024-12..2025-11, NDX (time known), `kalshi_implied_1m`. See its README. |
| `prediction-markets/` | Kalshi target markets/trades (Fed, CPI, recession...), Polymarket targets + fills since Oct 2024, hourly Iran/Fed theme indices, yfinance 1h prices |
| `kalshi-leading/`, `kalshi-catchup/`, `kalshi-crowd/`, `kalshi-pairs/`, `kalshi-0dte/` | Kalshi trades/prints per series (KXINXU, KXNASDAQ100U, KXBTCD, KXETHD), derived features, result CSVs, scratch scripts |
| `shortterm-combo/` | hourly SPY/QQQ panels, Kalshi 30-min implied closes, walk-forward and holdout results |
| `intraday-lab/`, `ml-research/` | result tables for those studies |
| `sentiment-research/results/` | mood index and sentiment overlay result CSVs from `run_sentiment_overlays.py` |

Kalshi history mirror: HF `TrevorJS/kalshi-trades` (Jun 2021 - Jan 2026).
Polymarket: HF `SII-WANGZJ/Polymarket_data`.

## Network from the cloud environment

Reachable: Yahoo, FRED, SEC EDGAR, huggingface.co, PyPI, raw.githubusercontent.com.
Blocked: Alpaca, Polymarket and Kalshi live APIs, cboe.com, CNN, trumpstruth.org,
and every sentiment survey source (aaii.com, NYU Baker-Wurgler, alternative.me, nasdaq data, Binance/Bybit/OKX APIs).
Live watchers (e.g. `watch_polymarket_iran.py`) must run on Josh's machine.
