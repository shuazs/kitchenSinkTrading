"""Batch 6: calendar effects in BTC and ETH daily returns (Coin Metrics PriceUSD).

Day-of-week, US market holidays, FOMC announcement day and next day, last-Friday options expiry window,
turn of month, quarter end, days since halving. Return of day t is close(t-1)->close(t) (UTC); a rule that
holds on day t is entered at close(t-1), which is known in advance for calendar events, so there is no lag
beyond the schedule itself. Split BTC 2014-2020 / 2021-2026-05, ETH 2016-2020 / 2021-2026-05.
Reports excess mean return in bp/day, t (Newey-West-free; days are weakly dependent) and whether both halves agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.data.coinmetrics import load_coin
from backtester.data import fomc

HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]


def us_holidays(idx):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    return USFederalHolidayCalendar().holidays(idx.min(), idx.max())


def last_friday(idx):
    out = []
    for p in pd.period_range(idx.min(), idx.max(), freq="M"):
        d = p.end_time.normalize()
        while d.weekday() != 4:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return pd.DatetimeIndex(out)


def main():
    try:
        fm = pd.DatetimeIndex(pd.to_datetime(fomc.statement_dates())).normalize()
    except Exception as e:
        print("no FOMC dates:", e)
        fm = pd.DatetimeIndex([])
    rows = []
    for coin, start, split in (("btc", "2014-10-01", "2021-01-01"), ("eth", "2016-03-01", "2021-01-01")):
        px = load_coin(coin).PriceUSD.dropna()
        r = px.pct_change().loc[start:]
        idx = r.index
        flags = {}
        for k in range(7):
            flags[f"dow={['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][k]}"] = pd.Series(idx.weekday == k, idx)
        hol = us_holidays(idx)
        flags["US holiday"] = pd.Series(idx.isin(hol), idx)
        flags["day before holiday"] = pd.Series(idx.isin(hol - pd.Timedelta(days=1)), idx)
        flags["day after holiday"] = pd.Series(idx.isin(hol + pd.Timedelta(days=1)), idx)
        flags["weekend"] = pd.Series(idx.weekday >= 5, idx)
        flags["FOMC day"] = pd.Series(idx.isin(fm), idx)
        flags["day after FOMC"] = pd.Series(idx.isin(fm + pd.Timedelta(days=1)), idx)
        lf = last_friday(idx)
        flags["last Fri (Deribit expiry)"] = pd.Series(idx.isin(lf), idx)
        flags["week before expiry"] = pd.Series(np.any([(idx > d - pd.Timedelta(days=7)) & (idx <= d) for d in lf], axis=0), idx)
        flags["day after expiry"] = pd.Series(idx.isin(lf + pd.Timedelta(days=1)), idx)
        dom = pd.Series(idx.day, idx)
        flags["turn of month (last 2, first 3)"] = (dom <= 3) | (idx.is_month_end) | (pd.Series(idx + pd.Timedelta(days=1), idx).dt.day == 1)
        flags["days 10-20"] = (dom >= 10) & (dom <= 20)
        flags["quarter end week"] = pd.Series(idx.month.isin([3, 6, 9, 12]) & (idx.day >= 24), idx)
        h = pd.Series(np.nan, idx)
        for hv in HALVINGS:
            d = (idx - pd.Timestamp(hv)).days
            h = h.where(~((d >= 0) & (d < 1460)), pd.Series(d // 91, idx))
        for q in range(0, 16):
            flags[f"halving quarter {q}"] = h == q
        for name, f in flags.items():
            f = f.astype(bool)
            for per, sl in (("A", slice(None, pd.Timestamp(split) - pd.Timedelta(days=1))), ("B", slice(split, None))):
                x = r.loc[sl]; m = f.loc[sl]
                if m.sum() < 15:
                    continue
                e = x[m] - x[~m].mean()
                rows.append(dict(coin=coin, flag=name, per=per, n=int(m.sum()), bp=1e4 * e.mean(), t=e.mean() / x[m].std() * np.sqrt(m.sum())))
    g = pd.DataFrame(rows)
    pv = g.pivot_table(index=["coin", "flag"], columns="per", values=["bp", "t", "n"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.dropna()
    pv.round(2).to_csv("/mnt/project-files/gdelt/calendar_batch6.csv")
    print(len(pv), "tests")
    print("|t|>2 in both halves, same sign:")
    print(pv[(pv.t_A.abs() > 2) & (pv.t_B.abs() > 2) & (np.sign(pv.t_A) == np.sign(pv.t_B))].round(2).to_string())
    print("\nbiggest |t| in A:")
    print(pv.reindex(pv.t_A.abs().sort_values(ascending=False).index).head(10).round(2).to_string())
    print("sign agreement A vs B:", (np.sign(pv.bp_A) == np.sign(pv.bp_B)).mean().round(2))


if __name__ == "__main__":
    main()
