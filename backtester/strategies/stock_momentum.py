import pandas as pd

from ..core import MarketData, Strategy
from .indicators import month_ends

# S&P 100 members as of 2025. Today's list, so every name survived to now:
# backtests on it overstate momentum returns (survivorship bias).
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC",
    "BK", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
    "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG",
    "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT",
    "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
    "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW",
    "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ",
    "WFC", "WMT", "XOM",
]


class HedgedMomentum(Strategy):
    """Monthly: hold the `top_n` stocks with the best 12-1 month momentum.

    Jegadeesh and Titman (1993); the long leg of a long-short momentum fund
    in fractional shares. Momentum is the `lookback`-day return skipping the
    last `skip` days (the one-month reversal). Names are equally weighted.

    The big-fund version shorts the losers; here the market exposure is
    hedged instead with `hedge`, an inverse ETF such as SH (-1x S&P 500),
    sized to `hedge_ratio` times the book's beta to `market` over
    `beta_lookback` days. Long book plus hedge sum to 100% of equity, so no
    margin is needed. With `ma` set, the hedge is only put on while `market`
    is below its `ma`-day average, and otherwise the book runs unhedged.
    """

    name = "hedged_momentum"
    params = {
        "universe": None,
        "top_n": 10,
        "lookback": 252,
        "skip": 21,
        "market": "SPY",
        "hedge": None,
        "hedge_ratio": 1.0,
        "beta_lookback": 126,
        "ma": None,
    }

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        p = self.p
        uni = [s for s in (p["universe"] or SP100) if s in data.bars]
        close = data.field("close").ffill()
        rets = close.pct_change(fill_method=None)
        mom = close[uni].shift(p["skip"]) / close[uni].shift(p["lookback"]) - 1
        mkt = close[p["market"]]
        below = mkt < mkt.rolling(p["ma"]).mean() if p["ma"] else pd.Series(True, index=data.index)

        w = pd.DataFrame(0.0, index=month_ends(data.index), columns=data.symbols)
        for d in w.index:
            m = mom.loc[d].dropna()
            if len(m) < p["top_n"]:
                continue
            picks = m.nlargest(p["top_n"]).index
            h = 0.0
            if p["hedge"] and below.loc[d]:
                window = rets.loc[:d].iloc[-p["beta_lookback"]:]
                book = window[picks].mean(axis=1)
                mk = window[p["market"]]
                beta = book.cov(mk) / mk.var() if mk.var() > 0 else 1.0
                h = max(p["hedge_ratio"] * beta, 0.0)
            # Long weight L and hedge h*L must fit in 100% of equity.
            long_w = 1.0 / (1.0 + h)
            w.loc[d, picks] = long_w / len(picks)
            if h:
                w.loc[d, p["hedge"]] = h * long_w
        return w
