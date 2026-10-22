"""Point-in-time company fundamentals for a whole stock universe, from SEC
EDGAR "frames", plus the prices and sectors a fundamental stock picker needs.

A frame is one XBRL concept for every filer in one calendar period (for
example net income for calendar 2019, or total assets at the end of Q2
2019), so a few hundred requests cover every S&P 500 company since 2009,
when XBRL filing began.

Two things to know about frames:

- Values are as last reported: a 2019 figure that was restated in a 2020
  filing shows the restated number. Restatements are rare and usually
  small, but it is a mild look-ahead.
- Frames carry no filing date. Each value is treated as public a fixed
  `lag` of days after its period ended, set to the SEC's slowest filing
  deadline (10-K: 60-90 days, 10-Q: 40-45 days) plus a margin.

Prices come from Yahoo with split and dividend history, so the module can
recover the unadjusted price that market value needs (`raw_close`), and
total-return OHLC for trading. Yahoo has no history for most delisted
stocks, which is the survivorship bias of free data: see `coverage`.
"""
from __future__ import annotations

import json

import pandas as pd

from .cache import cache_dir
from .edgar import _get, ticker_to_cik

# Flow items: one value per fiscal year (frame "CY2019"), public 90 days
# after the fiscal year ends. Annual figures are what Fama and French use too.
ANNUAL = {
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}
# Balance sheet items: quarter-end snapshots (frame "CY2019Q2I"), public
# 60 days after the quarter ends.
INSTANT = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
SHARES = ("dei", "EntityCommonStockSharesOutstanding", "shares")
LAG_DAYS = {"annual": 90, "instant": 60, "shares": 15}

# Tickers the S&P list used before a rename, mapped to the ticker Yahoo
# and the SEC use now, so a renamed company keeps its earlier history.
RENAMES = {
    "FB": "META", "ANTM": "ELV", "ABC": "COR", "RE": "EG", "PKI": "RVTY", "WLTW": "WTW",
    "CTL": "LUMN", "FLT": "CPAY", "PEAK": "DOC", "FBHS": "FBIN", "HFC": "DINO", "COG": "CTRA",
    "CDAY": "DAY", "BLL": "BALL", "JEC": "J", "HRS": "LHX", "SYMC": "GEN", "NLOK": "GEN",
    "CBS": "PARA", "VIAC": "PARA", "UTX": "RTX", "DWDP": "DD", "KRFT": "KHC", "BHGE": "BKR",
    "TSS": "GPN", "INFO": "SPGI", "MYL": "VTRS", "WCG": "CNC", "GAS": "SO",
}

# Two-digit SIC code ranges to broad sectors, for sector caps.
_SIC_SECTORS = [
    (100, 999, "materials"), (1000, 1499, "materials"), (1300, 1399, "energy"),
    (1500, 1799, "industrials"), (2000, 2199, "staples"), (2200, 2799, "discretionary"),
    (2800, 2829, "materials"), (2830, 2836, "health"), (2837, 2899, "materials"),
    (2900, 2999, "energy"), (3000, 3399, "materials"), (3400, 3569, "industrials"),
    (3570, 3579, "technology"), (3580, 3659, "industrials"), (3660, 3699, "technology"),
    (3700, 3799, "industrials"), (3800, 3840, "industrials"), (3841, 3851, "health"),
    (3852, 3999, "discretionary"), (4000, 4799, "industrials"), (4800, 4899, "communication"),
    (4900, 4999, "utilities"), (5000, 5199, "industrials"), (5200, 5999, "discretionary"),
    (6000, 6499, "financials"), (6500, 6799, "real_estate"), (7000, 7369, "discretionary"),
    (7370, 7379, "technology"), (7380, 7999, "discretionary"), (8000, 8099, "health"),
    (8100, 8999, "industrials"), (9000, 9999, "other"),
]


def frame(concept: str, period: str, taxonomy: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """One concept for every filer in one period, e.g. ("Assets", "CY2019Q2I").
    Cached once the period is over a year old, since late filers stop adding."""
    path = cache_dir("edgar_frames") / f"{taxonomy}_{concept}_{unit}_{period}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    try:
        rows = _get(f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json").json()["data"]
    except Exception as e:  # 404 when nobody reported the concept that period
        if "404" not in str(e):
            raise
        rows = []
    year = int(period[2:6])
    if year < pd.Timestamp.today().year - 1:
        path.write_text(json.dumps(rows))
    return pd.DataFrame(rows)


def _periods(kind: str, start_year: int, end_year: int) -> list[str]:
    if kind == "annual":
        return [f"CY{y}" for y in range(start_year, end_year + 1)]
    return [f"CY{y}Q{q}I" for y in range(start_year, end_year + 1) for q in range(1, 5)]


def panel(start_year: int = 2008, end_year: int | None = None, ciks=None) -> pd.DataFrame:
    """Long table of fundamentals: cik, item, end, available, val.

    `available` is the first date the value may be used. For items with
    several possible XBRL tags the first tag a company reported wins.
    """
    end_year = end_year or pd.Timestamp.today().year
    path = cache_dir("edgar_frames") / f"panel_{start_year}_{end_year}.csv.gz"
    if path.exists() and end_year < pd.Timestamp.today().year:
        out = pd.read_csv(path, parse_dates=["end", "available"])
    else:
        parts = []
        jobs = [("annual", item, tags, "us-gaap", "USD") for item, tags in ANNUAL.items()]
        jobs += [("instant", item, tags, "us-gaap", "USD") for item, tags in INSTANT.items()]
        jobs += [("shares", "shares", [SHARES[1]], SHARES[0], SHARES[2])]
        for kind, item, tags, tax, unit in jobs:
            per = _periods("annual" if kind == "annual" else "instant", start_year, end_year)
            got = []
            for rank, tag in enumerate(tags):
                for p in per:
                    f = frame(tag, p, tax, unit)
                    if f.empty:
                        continue
                    got.append(f.assign(rank=rank, item=item)[["cik", "end", "val", "rank", "item"]])
            if not got:
                continue
            df = pd.concat(got)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("rank").drop_duplicates(["cik", "end"], keep="first")
            df["available"] = df["end"] + pd.Timedelta(days=LAG_DAYS[kind])
            parts.append(df.drop(columns="rank"))
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(path, index=False)
    if ciks is not None:
        out = out[out["cik"].isin(set(ciks))]
    return out


def as_of(long: pd.DataFrame, dates, ciks) -> dict[str, pd.DataFrame]:
    """Latest available value of each item on each date: item -> date x cik."""
    dates = pd.DatetimeIndex(dates)
    out = {}
    for item, g in long[long["cik"].isin(set(ciks))].groupby("item"):
        # Keep the value with the latest period end among those public by then.
        g = g.sort_values(["available", "end"])
        wide = g.pivot_table(index="available", columns="cik", values="val", aggfunc="last")
        out[item] = wide.reindex(wide.index.union(dates)).ffill().reindex(dates)
    return out


def sectors(ciks, refresh: bool = False) -> dict[int, str]:
    """Broad sector for each CIK from its SIC code (EDGAR submissions)."""
    path = cache_dir("edgar") / "sic_codes.json"
    known = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() and not refresh else {}
    for c in ciks:
        if c in known:
            continue
        try:
            known[c] = _get(f"https://data.sec.gov/submissions/CIK{int(c):010d}.json").json().get("sic") or ""
        except Exception:
            known[c] = ""
        path.write_text(json.dumps(known))
    out = {}
    for c in ciks:
        sic = int(known.get(c) or 0)
        out[c] = next((name for lo, hi, name in reversed(_SIC_SECTORS) if lo <= sic <= hi), "other")
    return out


def load_bars(symbols, start: str = "2007-01-01", refresh: bool = False, chunk: int = 100) -> dict[str, pd.DataFrame]:
    """Daily bars with total-return OHLC (dividends and splits folded in, for
    trading) and `raw_close`, the price as quoted that day (for market value).
    Symbols Yahoo has no data for are left out."""
    import yfinance as yf

    root = cache_dir("yahoo_tr")
    missing_path = root / "_missing.txt"
    missing = set(missing_path.read_text().split()) if missing_path.exists() else set()
    symbols = list(dict.fromkeys(symbols))
    todo = [s for s in symbols if refresh or (not (root / f"{s}.csv").exists() and s not in missing)]
    for k in range(0, len(todo), chunk):
        batch = todo[k : k + chunk]
        raw = yf.download(batch, start=start, auto_adjust=False, actions=True, progress=False,
                          group_by="ticker", threads=True)
        for s in batch:
            try:
                df = raw[s].dropna(subset=["Close"]) if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(subset=["Close"])
            except KeyError:
                df = pd.DataFrame()
            if len(df) < 20:
                missing.add(s)
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)
            # Yahoo's Close is split-adjusted; undo later splits to get the quoted price.
            splits = df["Stock Splits"].replace(0, 1.0).fillna(1.0)
            later = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            adj = df["Adj Close"] / df["Close"]
            out = pd.DataFrame({
                "open": df["Open"] * adj, "high": df["High"] * adj, "low": df["Low"] * adj,
                "close": df["Adj Close"], "volume": df["Volume"], "raw_close": df["Close"] * later,
            })
            out.to_csv(root / f"{s}.csv")
        missing_path.write_text("\n".join(sorted(missing)))
    return {s: pd.read_csv(root / f"{s}.csv", index_col=0, parse_dates=True) for s in symbols if (root / f"{s}.csv").exists()}


def sp500_universe(since: str = "2009-01-01") -> tuple[pd.Series, list[str]]:
    """Point-in-time S&P 500 membership with renamed tickers mapped to today's.

    Returns (date -> frozenset of members, every ticker that was ever a member).
    """
    from .sp500 import membership

    m = membership()
    prior = m[m.index < since]
    m = m[m.index >= since] if prior.empty else pd.concat([prior.iloc[-1:], m[m.index >= since]])
    fix = lambda s: RENAMES.get(s.replace(".", "-"), s.replace(".", "-"))
    m = m.map(lambda members: frozenset(fix(s) for s in members))
    tickers = sorted(set().union(*m.values))
    return m, tickers


def coverage(members: pd.Series, have) -> pd.Series:
    """Share of index members on each date that have price data: what's
    missing is mostly companies later acquired or bankrupt."""
    have = set(have)
    return members.map(lambda s: len(s & have) / max(len(s), 1))


def stock_cik_map(tickers) -> dict[str, int]:
    cik = ticker_to_cik()
    return {t: cik[t] for t in tickers if t in cik}


ETFS = ["SPY", "QQQ", "RSP", "EFA", "EEM", "IJR", "IEF", "BIL", "HYG", "LQD", "SHY", "TLT", "GLD", "SSO"]


def market_data(since: str = "2009-01-01", etfs=ETFS, macro: pd.DataFrame | None = None):
    """MarketData for `FundamentalAllocator`: every S&P 500 member since
    `since` that Yahoo still has, the ETFs, and the point-in-time extras."""
    from ..core import MarketData

    members, tickers = sp500_universe(since)
    bars = load_bars(tickers + list(etfs))
    cik = stock_cik_map([s for s in tickers if s in bars])
    sec = sectors(sorted(set(cik.values())))
    long = panel(int(since[:4]) - 1)
    extra = {"members": members, "fundamentals": long[long["cik"].isin(set(cik.values()))], "cik": cik,
             "sector": {t: sec[c] for t, c in cik.items()}, "coverage": coverage(members, bars)}
    if macro is not None:
        extra["macro"] = macro
    bars = {s: b.loc["2007-01-01":] for s, b in bars.items() if s in cik or s in etfs}
    return MarketData(bars, extra)
