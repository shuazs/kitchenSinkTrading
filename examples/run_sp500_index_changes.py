"""S&P 500 deletions and additions after the effective date (pre-registered).

H1: buy each deleted stock at the open of the first trading day after the
effective date D, hold 20 trading days, minus SPY. H2: short each added
stock the same way (+1%/yr borrow). H3: 60 days. Cost 20 bp round trip.
In-sample events 2001-2015, holdout 2016 on; t clustered by event date.
Membership: github fja05680/sp500; prices: yfinance (only names still on
Yahoo, so bankrupt deletions are missing and H1 is flattered). Hypotheses:
/mnt/project-files/sp500-changes/HYPOTHESES.md.

    python examples/run_sp500_index_changes.py
"""
import numpy as np
import pandas as pd

D = "/mnt/project-files/sp500-changes/"
HOLD, COST, BORROW = "2016-01-01", 0.002, 0.01


def event_returns(e, px, hold):
    o, c = px["open"], px["close"]
    days = c.index
    rows = []
    for _, r in e.iterrows():
        t = r["ticker"].replace(".", "-")
        if t not in c or c[t].isna().all():
            continue
        i = days.searchsorted(r["date"], side="right")  # first trading day after D
        if i + hold > len(days):
            continue
        if r["kind"] == "add" and c[t].iloc[max(0, i - 60):i].isna().any():
            continue  # spin-offs / new listings
        p0, p1 = o[t].iloc[i], c[t].iloc[i + hold - 1]
        s0, s1 = o["SPY"].iloc[i], c["SPY"].iloc[i + hold - 1]
        m0, m1 = o["MDY"].iloc[i], c["MDY"].iloc[i + hold - 1]
        if np.isnan([p0, p1, s0, s1]).any() or p0 <= 0:
            continue
        sign = 1 if r["kind"] == "del" else -1
        ex = sign * (p1 / p0 - s1 / s0) - COST - (BORROW * hold / 252 if sign < 0 else 0)
        rows.append({"date": r["date"], "ticker": t, "kind": r["kind"], "ex_spy": ex,
                     "ex_mdy": sign * (p1 / p0 - m1 / m0) - COST if m0 == m0 else np.nan})
    return pd.DataFrame(rows)


def stats(x, col="ex_spy"):
    g = x.groupby("date")[col].mean().dropna()
    t = g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan
    return f"{len(x):>4} trades {len(g):>3} dates {x[col].mean() * 100:+6.2f}% t {t:+5.2f}"


def main():
    e = pd.read_csv(D + "events.csv", parse_dates=["date"])
    e = e[e["date"] >= "2001-01-01"]
    px = pd.read_parquet(D + "prices.parquet")
    print(f"{len(e)} index changes 2001-2026; prices on Yahoo for "
          f"{e['ticker'].str.replace('.', '-', regex=False).isin(px['close'].columns[px['close'].notna().any()]).mean():.0%}")
    for hold in (20, 60):
        r = event_returns(e, px, hold)
        for kind, lab in (("del", "H1 buy deletions"), ("add", "H2 short additions")):
            x = r[r["kind"] == kind]
            print(f"{lab} {hold}d vs SPY  IS {stats(x[x['date'] < HOLD])} | HO {stats(x[x['date'] >= HOLD])}")
            print(f"{'':>17}{hold}d vs MDY  IS {stats(x[x['date'] < HOLD], 'ex_mdy')} | HO {stats(x[x['date'] >= HOLD], 'ex_mdy')}")


if __name__ == "__main__":
    main()
