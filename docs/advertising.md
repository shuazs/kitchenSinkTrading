# Advertising spend vs stock returns (2026-11-07)

**Verdict: no tradable edge.** Ad spend moves with sales, so it correlates with
returns in the same year, but nothing predicts returns after the 10-K is public.
No paper-trade candidate.

## Data and reachability
- Used: SEC XBRL frames `AdvertisingExpense` (annual, 2009-2025, ~7,000 firm-years with growth, 971 filers; 613 map to a current ticker and have Yahoo prices, ad spend >= $25M), `Revenues` for intensity; Indeed Hiring Lab postings by sector (GitHub raw, "Marketing", 2020-02 to 2026-10).
- Not reachable from the cloud: Google Trends, Meta Ad Library (Graph API), Google Ads Transparency, Wikimedia pageview API. No HF dataset with ad spend or TV ad data. Free TV/Nielsen ad spend does not exist.
- Frames carry no filing date: values count as public 100 days after fiscal year end. Only current tickers have prices (survivorship bias favours the signals).

## Academic prior
Grullon-Kanatas-Weston (2004): ads widen the shareholder base and lower the cost of capital. Lou (2014): firms raise ads to push up the short-run price, then it reverses. Chemmanur-Yan (2009): ad increases give a one-year positive then negative return. Fehle et al. (2005): Super Bowl ads, small announcement effects. Expectation: ad growth predicts a small positive 0-3 month return that reverses by 12-24 months.

## Results
Contemporaneous vs predictive, year-demeaned returns, rank correlation, mean by-year (t over 15 years):

| Signal | FY return | Next 12m after filing |
| --- | --- | --- |
| Ad growth | +0.109 (t 4.7) | -0.009 (t -0.4) |
| Sales growth | +0.225 (t 7.0) | +0.012 (t 0.4) |
| Ad growth minus sales growth | -0.051 (t -2.4) | -0.033 (t -1.3) |
| Ad/sales (intensity) | -0.071 (t -2.4) | -0.056 (t -2.0) |

Calendar-time quintile portfolios (monthly, top quintile minus equal-weight universe, %/yr; IS <= 2019 / OOS 2020+):
- Ad growth, 0-3 months after filing: +8.1 (t 1.4) / +4.0 (t 0.7). 3-12m: +2.8 / -4.9. 12-24m: +1.2 / -3.1. Reversal pattern from Lou is not there.
- Ad intensity 0-3m: +13.0 (t 2.3) IS but +0.1 OOS. 0-12m: +2.6 / -6.2. High-ad firms lag in 2020+.
- Change in intensity, ad minus sales growth: negative or zero everywhere (top-bottom -4 to -13 %/yr IS, -0 to -10 OOS).
- 0-3m windows have only ~30 months per period (cross-section needs 40+ firms). Long-short would need shorts; long-only top quintile never beats the universe OOS.

Side tests:
- Indeed marketing job postings (4-week change, 7-day publication lag, 83 non-overlapping 20-day windows): Spearman with next-month SPY/XLY/QQQ/OMC/GOOGL/META/PINS/SNAP/TTD within +-0.17, hi-lo 0 to +4%/month with the sign flipping across stocks. No signal.
- Super Bowl (17 games, 26-stock hand-picked advertiser basket vs SPY): 20-day run-up +0.64% (t 1.3, 12 of 17 up), Monday 0.0%, next 5 days +0.02%. Not significant; basket is chosen with hindsight.

Not tested (no data): Google Trends, Meta/Google ad counts, TV ad spend, quarterly ad expense (rarely tagged).
Scripts: `examples/run_advertising.py`, `backtester/data/advertising.py`; scratch copies in `/mnt/project-files/advertising/`.
