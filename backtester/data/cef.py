"""Closed-end fund prices and NAVs from Yahoo.

Yahoo carries a fund's daily NAV under X<ticker>X (ADX -> XADEX). `universe`
probes EDGAR-listed fund-like names for a NAV ticker, so it only finds funds
still trading today (merged or liquidated CEFs are missing).
"""

import pandas as pd

from ..core import MarketData


def nav_symbol(ticker: str) -> str:
    return f"X{ticker}X"


def universe(candidates: list[str], start: str = "2005-01-01", min_days: int = 500) -> list[str]:
    """Tickers in `candidates` that have at least `min_days` of Yahoo NAV history."""
    import yfinance as yf

    found = []
    for i in range(0, len(candidates), 100):
        close = yf.download([nav_symbol(t) for t in candidates[i : i + 100]], start=start, progress=False, auto_adjust=False)["Close"]
        found += [c[1:-1] for c in close.columns if close[c].notna().sum() >= min_days]
    return found


def load_cef_market(tickers: list[str], start: str = "2005-01-01", max_median_discount: float = 0.3) -> MarketData:
    """Total-return OHLCV (distributions folded in via Adj Close) plus raw
    close and NAV in `extra` ("nav", "raw_close"). Funds whose median
    discount is beyond `max_median_discount` (bad NAV mapping) are dropped."""
    import yfinance as yf

    px = yf.download(tickers, start=start, progress=False, auto_adjust=False)
    nav = yf.download([nav_symbol(t) for t in tickers], start=start, progress=False, auto_adjust=False)["Close"]
    return cef_market_from_frames(px, nav, max_median_discount)


def cef_market_from_frames(px: pd.DataFrame, nav: pd.DataFrame, max_median_discount: float = 0.3) -> MarketData:
    """Build the MarketData from yfinance-shaped frames (fields x tickers)."""
    nav = nav.rename(columns=lambda c: c[1:-1] if c.startswith("X") and c.endswith("X") else c)
    close = px["Close"]
    nav = nav.reindex(index=close.index, columns=close.columns)
    disc = close / nav - 1
    good = [t for t in close.columns if abs(disc[t].median()) < max_median_discount]
    bars = {}
    for t in good:
        f = (px["Adj Close"][t] / close[t]).ffill()
        df = pd.DataFrame({
            "open": px["Open"][t] * f, "high": px["High"][t] * f, "low": px["Low"][t] * f,
            "close": px["Adj Close"][t], "volume": px["Volume"][t],
        }).dropna(subset=["close"])
        if len(df) > 250:
            bars[t] = df
    # the strategy compares raw (unadjusted) close to NAV, so store the discount
    # inputs on the adjusted scale: NAV * adj factor
    adj = (px["Adj Close"][list(bars)] / close[list(bars)]).ffill()
    return MarketData(bars, {"nav": nav[list(bars)] * adj, "raw_close": close[list(bars)]})
