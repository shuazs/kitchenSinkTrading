"""Clone the highest-conviction holdings of concentrated hedge funds from 13F.

Research basis:
- Cohen, Polk and Silli (2010), "Best Ideas": a manager's largest
  overweights beat the market; the rest of the portfolio does not.
- Angelini, Iqbal and Jivraj (2019) and the Goldman Sachs "VIP" basket:
  stocks that show up most often among hedge funds' top 10 holdings have
  outperformed, with long stretches of lagging (2021-22).
- 13F lag: holdings are as of quarter end and published up to 45 days
  later, so a clone trades on stale positions and misses shorts and hedges.

Rule (point in time): on the first trading day after each 45-day filing
deadline, take every original 13F for the quarter filed by then. Keep
"concentrated" managers: US stock book between min_aum and max_aum, between
min_positions and max_positions names, top 10 holdings at least min_top10 of
the book. Each kept manager votes for its top_k positions. Stocks are ranked
by votes, then by summed weight, and the top n_stocks with prices are held
equally weighted until the next quarter. Manager selection uses only that
filing, so there is no hindsight about who turned out to be good.
"""
from __future__ import annotations

import pandas as pd

from ..core import MarketData, Strategy


def signal_date(period: pd.Timestamp) -> pd.Timestamp:
    """Day after the 13F deadline for a quarter ending on `period`."""
    return period + pd.Timedelta(days=46)


def best_ideas(
    holdings: pd.DataFrame,
    min_aum: float = 250e6,
    max_aum: float = 50e9,
    min_positions: int = 10,
    max_positions: int = 60,
    min_top10: float = 0.5,
    top_k: int = 3,
    keep: int = 60,
) -> pd.DataFrame:
    """Rank CUSIPs by how many concentrated managers hold them in their top_k.

    `holdings` comes from `backtester.data.thirteenf.load_holdings`. Returns
    the `keep` best-ranked CUSIPs per quarter: date (signal date), period,
    cusip, issuer, votes, weight_sum, managers (number kept that quarter), rank.
    """
    h = holdings.copy()
    h["date"] = h["period"].map(signal_date)
    h = h[h["filed"] < h["date"]]  # filed by the deadline
    ok = (
        h["total_usd"].between(min_aum, max_aum)
        & h["n_positions"].between(min_positions, max_positions)
        & (h["top10_share"] >= min_top10)
    )
    h = h[ok & (h["rank"] <= top_k)]
    n_mgr = h.groupby("period")["manager_cik"].nunique().rename("managers")
    g = (
        h.groupby(["period", "date", "cusip"], as_index=False)
        .agg(issuer=("issuer", "first"), votes=("manager_cik", "nunique"), weight_sum=("weight", "sum"))
        .sort_values(["period", "votes", "weight_sum"], ascending=[True, False, False])
    )
    g["rank"] = g.groupby("period").cumcount() + 1
    g = g[g["rank"] <= keep].join(n_mgr, on="period")
    return g.reset_index(drop=True)


class HedgeFundClone(Strategy):
    """Hold the top `n_stocks` 13F best ideas, equal weight, rebalanced quarterly.

    Put the output of `best_ideas` with a `symbol` column (see
    `thirteenf.cusip_tickers`) in `MarketData.extra[ideas_key]`. Names without
    a ticker or a price on the signal day are skipped and the next-ranked
    name takes the slot. min_votes drops names held by fewer managers.
    """

    name = "hedge_fund_clone"
    params = {"ideas_key": "best_ideas", "n_stocks": 15, "min_votes": 2}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        ideas = data.extra[self.p["ideas_key"]]
        ideas = ideas[ideas["symbol"].isin(data.symbols) & (ideas["votes"] >= self.p["min_votes"])]
        idx = data.index
        close = data.field("close")
        rows = {}
        self.holdings = {}
        for date, g in ideas.groupby("date"):
            i = idx.searchsorted(date)
            if i >= len(idx):
                continue
            day = idx[i]
            live = [s for s in g.sort_values("rank")["symbol"].drop_duplicates() if pd.notna(close.at[day, s])]
            picks = live[: self.p["n_stocks"]]
            if not picks:
                continue
            w = pd.Series(0.0, index=data.symbols)
            w[picks] = 1.0 / len(picks)
            rows[day] = w
            self.holdings[day] = picks
        return pd.DataFrame(rows).T.reindex(columns=data.symbols)
