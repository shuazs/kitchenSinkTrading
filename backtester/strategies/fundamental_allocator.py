"""Long-term fundamental portfolio manager across US large caps, international
and small caps, with a macro regime switch and portfolio-level risk control.

Three sleeves (default 60/20/20):

- US stocks: point-in-time S&P 500 members ranked monthly, each factor
  ranked within its sector. Available factors: value (earnings, book and
  cash flow yield), quality (ROE, gross profitability, cash flow on assets,
  low leverage, low accruals), 12-1 month momentum and low volatility. The
  default is quality + low vol, the best mix on 2007-2018. The top `top_n`
  are held equally weighted (or inverse-vol), at most `max_sector` of the
  names per sector; holdings stay while in the top `buffer` x `top_n`.
- International: developed (EFA) and emerging (EEM) ETFs.
- Small caps: S&P 600 (IJR), whose index requires positive earnings, a
  built-in quality screen that the Russell 2000 lacks.

Risk is set in three layers, each optional:

1. Macro regime ("growth-trend timing", Philosophical Economics 2016): go
   to `defensive_mult` (0 = fully defensive) only when unemployment is
   rising (Sahm-style) and SPY is below its 200-day average. A credit
   signal (Baa spread, or HYG vs IEF as a stand-in) can be added through
   `macro_signals`; it cut drawdowns but cost about 3%/yr in 2010-2018.
2. Trend: the international and small-cap sleeves each step aside while
   below their own 200-day average, into the US stock sleeve
   (`trend_to="stocks"`) or the defensive asset.
3. Volatility cap: the risky book is scaled down when its 63-day ex-ante
   volatility is above `vol_target`.

Optional: hold `core_frac` of the US sleeve in SPY, and swap `lever_frac`
of it into a 2x ETF while the regime is calm.

What is not in stocks sits in the defensive asset with the better 6-month
return of `defensive` (Treasuries or T-bills), so 2022-style bond losses are
mostly avoided.

Result ($5,000, costs; examples/run_fundamental_allocator.py): 11.0%/yr,
max drawdown -17% in 2007-2018 vs SPY 6.5%, -55% (the gain is 2008), but
11.1% vs 17.4% and -35% vs -34% in the 2019-2026 holdout. Quality + low vol
lagged mega-cap growth, and the macro switch is too slow for a crash like
March 2020.

data.extra needs: "members" (date -> frozenset of tickers, point-in-time
index membership), "fundamentals" (long table from data.fundamentals.panel),
"cik" (ticker -> CIK), "sector" (ticker -> sector name) and optionally
"macro" (DataFrame with UNRATE and/or BAA10Y, dated as FRED dates them).
Bars need a `raw_close` column (the quoted price) for market value.
`data.fundamentals.market_data` builds all of it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from ..data.fundamentals import as_of
from .indicators import month_ends

# FRED dates a month's unemployment rate on the 1st; it is released about
# five weeks later.
UNRATE_LAG_DAYS = 36
_SCORE_CACHE: dict = {}

VALUE = ["ep", "bp", "cfp"]
QUALITY = ["roe", "gpa", "cfoa", "lev", "accr"]


def _sector_rank(x: pd.Series, sector: pd.Series, min_names: int = 5) -> pd.Series:
    """Percentile rank within sector; sectors with few names rank against everyone."""
    x = x.replace([np.inf, -np.inf], np.nan)
    overall = x.rank(pct=True)
    within = x.groupby(sector).rank(pct=True)
    counts = x.groupby(sector).transform("count")
    return within.where(counts >= min_names, overall)


def factor_scores(data: MarketData, dates: pd.DatetimeIndex, min_history: int = 252) -> dict[pd.Timestamp, pd.DataFrame]:
    """For each date, a ticker x factor table of sector-relative percentile
    ranks (value, quality, momentum, lowvol) over that date's index members.
    Cached per MarketData object, since every variant of the strategy
    shares it."""
    key = (id(data), tuple(dates), min_history)
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    ex = data.extra
    cik_of, sector_of = ex["cik"], ex["sector"]
    members = ex["members"]
    stocks = [s for s in data.symbols if s in cik_of]
    close = data.field("close")[stocks].ffill(limit=5)
    raw = data.field("raw_close")[stocks].ffill(limit=5)
    rets = close.pct_change(fill_method=None)
    vol = rets.rolling(252, min_periods=200).std()
    hist = close.notna().cumsum()
    ciks = sorted({cik_of[s] for s in stocks})
    fund = as_of(ex["fundamentals"], dates, ciks)

    out = {}
    for d in dates:
        mem = members[members.index <= d]
        if mem.empty:
            continue
        uni = [s for s in mem.iloc[-1] if s in cik_of and s in close.columns and hist.at[d, s] >= min_history
               and not np.isnan(close.at[d, s])]
        if len(uni) < 50:
            continue
        c = pd.Series({s: cik_of[s] for s in uni})

        def get(item):
            f = fund.get(item)
            return pd.Series(f.loc[d].reindex(c.values).values, index=uni) if f is not None else pd.Series(np.nan, index=uni)

        ni, eq, ocf = get("net_income"), get("equity"), get("op_cash_flow")
        assets, liab = get("assets"), get("liabilities")
        gp = get("gross_profit").fillna(get("revenue") - get("cost_of_revenue"))
        mcap = get("shares") * raw.loc[d, uni]
        mcap = mcap.where(mcap > 0)
        assets = assets.where(assets > 0)
        raw_f = pd.DataFrame({
            "ep": ni / mcap, "bp": eq / mcap, "cfp": ocf / mcap,
            "roe": (ni / eq).where(eq > 0), "gpa": gp / assets, "cfoa": ocf / assets,
            "lev": -liab / assets, "accr": -(ni - ocf) / assets,
        })
        sec = pd.Series({s: sector_of.get(s, "other") for s in uni})
        ranks = raw_f.apply(lambda col: _sector_rank(col, sec))
        i = close.index.get_loc(d)
        mom = close.iloc[i - 21] / close.iloc[max(i - 252, 0)] - 1
        tab = pd.DataFrame({
            "value": ranks[VALUE].mean(axis=1, skipna=True).where(ranks[VALUE].notna().sum(axis=1) >= 2),
            "quality": ranks[QUALITY].mean(axis=1, skipna=True).where(ranks[QUALITY].notna().sum(axis=1) >= 3),
            "momentum": _sector_rank(mom[uni], sec),
            "lowvol": _sector_rank(-vol.loc[d, uni], sec),
            "vol": vol.loc[d, uni],
            "sector": sec,
        })
        out[d] = tab
    _SCORE_CACHE[key] = out
    return out


def macro_stress(data: MarketData, index: pd.DatetimeIndex, sahm: float = 0.3, credit: float = 0.25,
                 credit_proxy: tuple | None = ("HYG", "IEF"), credit_band: float = 0.0,
                 signals=("unemployment", "credit")) -> pd.Series:
    """True when the economy looks to be weakening, from data public that day:
    the 3-month average unemployment rate is `sahm` points above its low of
    the prior 12 months, or the Baa-Treasury spread is `credit` points above
    its 1-year average. Without FRED's spread, `credit_proxy` (a credit ETF
    and a Treasury ETF of similar duration) stands in: credit is stressed
    while the credit ETF's return relative to Treasuries is below its 200-day
    average by more than `credit_band`, which is spreads widening."""
    m = data.extra.get("macro")
    stress = pd.Series(False, index=index)
    if m is None:
        m = pd.DataFrame()
    if "credit" in signals and "BAA10Y" not in m and credit_proxy and all(s in data.bars for s in credit_proxy):
        close = data.field("close")[list(credit_proxy)].ffill()
        ratio = close[credit_proxy[0]] / close[credit_proxy[1]]
        stress |= (ratio < ratio.rolling(200).mean() * (1 - credit_band)).reindex(index).fillna(False).astype(bool)
    if "unemployment" in signals and "UNRATE" in m:
        u = m["UNRATE"].dropna()
        u3 = u.rolling(3).mean()
        rising = u3 - u3.shift(1).rolling(12).min() >= sahm
        rising.index = rising.index + pd.Timedelta(days=UNRATE_LAG_DAYS)
        stress |= rising.reindex(index.union(rising.index)).ffill().reindex(index).fillna(False).astype(bool)
    if "credit" in signals and "BAA10Y" in m:
        b = m["BAA10Y"].dropna()
        wide = b > b.rolling(252, min_periods=200).mean() + credit
        wide.index = wide.index + pd.Timedelta(days=1)  # daily series publish next morning
        stress |= wide.reindex(index.union(wide.index)).ffill().reindex(index).fillna(False).astype(bool)
    return stress


class FundamentalAllocator(Strategy):
    name = "fundamental_allocator"
    params = {
        "sleeves": {"stocks": 0.6, "intl": 0.2, "small": 0.2},
        "intl": ("EFA", "EEM"),
        "small": ("IJR",),
        "market": "SPY",
        "stock_etf": None,  # hold this ETF instead of picking stocks
        "top_n": 25,  # None holds every eligible member equally
        "max_sector": 0.25,
        "factors": {"value": 0.0, "quality": 1.0, "momentum": 0.0, "lowvol": 1.0},
        "buffer": 2.0,
        "weighting": "equal",  # or "inv_vol"
        "stock_rebalance_months": 1,
        "macro": True,
        "caution_mult": 1.0,
        "defensive_mult": 0.0,
        "sleeve_trend": True,
        "trend_days": 200,
        "trend_to": "stocks",  # where a sleeve below trend goes: "defensive" or "stocks"
        "vol_target": 0.18,
        "vol_window": 63,
        "core_etf": "SPY",
        "core_frac": 0.0,  # share of the US sleeve held in the cap-weighted index
        "lever_etf": "SSO",
        "lever_frac": 0.0,  # share of the US sleeve swapped into the 2x ETF while risk-on
        "defensive": ("IEF", "BIL"),
        "credit_proxy": ("HYG", "IEF"),
        "credit_band": 0.0,
        "macro_signals": ("unemployment",),
        "min_history": 252,
    }

    def pick_stocks(self, tab: pd.DataFrame, held=()) -> pd.Series:
        """Weights (summing to 1) of the stock sleeve from one date's scores.
        Stocks already `held` stay while they rank in the top
        `buffer` x `top_n`, which cuts turnover without changing the tilt."""
        p = self.p
        fw = {k: v for k, v in p["factors"].items() if v}
        if p["top_n"] is None or not fw:
            return pd.Series(1.0 / len(tab), index=tab.index)
        score = sum(tab[k].fillna(0.5) * v for k, v in fw.items()) / sum(fw.values())
        score = score[tab[list(fw)].notna().sum(axis=1) >= max(1, len(fw) - 1)]
        cap = max(1, math.ceil(p["max_sector"] * p["top_n"]))
        ranked = score.sort_values(ascending=False).index
        keep = [s for s in ranked[: int(p["buffer"] * p["top_n"])] if s in set(held)]
        picks, per_sector = [], {}
        for s in keep + [s for s in ranked if s not in set(keep)]:
            sec = tab.at[s, "sector"]
            if per_sector.get(sec, 0) >= cap:
                continue
            picks.append(s)
            per_sector[sec] = per_sector.get(sec, 0) + 1
            if len(picks) == p["top_n"]:
                break
        if not picks:
            return pd.Series(dtype=float)
        if p["weighting"] == "inv_vol":
            w = 1 / tab.loc[picks, "vol"].clip(lower=0.005)
        else:
            w = pd.Series(1.0, index=picks)
        return w / w.sum()

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        p = self.p
        idx = data.index
        close = data.field("close").ffill(limit=5)
        rets = close.pct_change(fill_method=None)
        dates = month_ends(idx)
        picking = p["sleeves"].get("stocks") and not p["stock_etf"]
        scores = factor_scores(data, dates, p["min_history"]) if picking else {}

        above = close > close.rolling(p["trend_days"]).mean()
        stress = macro_stress(data, idx, credit_proxy=p["credit_proxy"], credit_band=p["credit_band"],
                              signals=p["macro_signals"]) if p["macro"] else pd.Series(False, index=idx)
        defensive = [s for s in p["defensive"] if s in data.bars]
        mom6 = close[defensive].pct_change(126, fill_method=None) if defensive else None

        w = pd.DataFrame(0.0, index=dates, columns=data.symbols)
        stock_w = pd.Series(dtype=float)
        for k, d in enumerate(dates):
            if picking and (k % p["stock_rebalance_months"] == 0 or stock_w.empty) and d in scores:
                stock_w = self.pick_stocks(scores[d], stock_w.index)
            risky = pd.Series(0.0, index=data.symbols)
            # Before fundamentals exist the stock sleeve is held as the market ETF.
            book = stock_w if not stock_w.empty else pd.Series({p["stock_etf"] or p["market"]: 1.0})
            if p["core_frac"]:
                book = (book * (1 - p["core_frac"])).add(pd.Series({p["core_etf"]: p["core_frac"]}), fill_value=0.0)
            weak = bool(stress.at[d]) if p["macro"] else False
            calm = not weak and bool(above.at[d, p["market"]])
            if p["lever_frac"] and calm and p["lever_etf"] in data.bars:
                # Risk-on: swap part of the US sleeve for a 2x ETF. Exposure
                # goes above 100% without margin; it comes off when either
                # the economy or the trend turns.
                book = (book * (1 - p["lever_frac"])).add(pd.Series({p["lever_etf"]: p["lever_frac"]}), fill_value=0.0)
            risky[book.index] += p["sleeves"].get("stocks", 0) * book
            for sleeve in ("intl", "small"):
                names = [s for s in p[sleeve] if s in data.bars and not np.isnan(close.at[d, s])]
                for s in names:
                    on = bool(above.at[d, s]) if p["sleeve_trend"] else True
                    share = p["sleeves"].get(sleeve, 0) / len(names)
                    if on:
                        risky[s] += share
                    elif p["trend_to"] == "stocks":
                        risky[book.index] += share * book
            if p["macro"]:
                down = not bool(above.at[d, p["market"]])
                risky *= p["defensive_mult"] if (weak and down) else (p["caution_mult"] if (weak or down) else 1.0)
            if p["vol_target"]:
                held = risky[risky > 0].index
                window = rets[held].loc[:d].iloc[-p["vol_window"]:].fillna(0.0)
                vol = float(np.sqrt(risky[held] @ window.cov().to_numpy() @ risky[held] * 252)) if len(held) else 0.0
                if vol > p["vol_target"]:
                    risky *= p["vol_target"] / vol
            w.loc[d] = risky
            rest = 1.0 - risky.sum()
            if rest > 1e-6 and defensive:
                m6 = mom6.loc[d].dropna()
                if not m6.empty:
                    w.at[d, m6.idxmax()] += rest
        return w
