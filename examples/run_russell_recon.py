"""Russell reconstitution day at the ETF level (pre-registered, see
/mnt/project-files/russell-recon/HYPOTHESES.md).

H1: IWC (microcap, where deleted names land) minus IWM from the close of
recon day R to R+10 trading days; H2: the same spread over R-15..R; H3: IWM
minus SPY over R..R+10. Recon day: 4th Friday of June from 2014, last Friday
before. Cost 10 bp per leg. In-sample 2006-15, holdout 2016-25.

    python examples/run_russell_recon.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def recon_day(year):
    fridays = pd.date_range(f"{year}-06-01", f"{year}-06-30", freq="W-FRI")
    return fridays[3] if year >= 2014 else fridays[-1]


def main():
    px = yf.download(["IWC", "IWM", "SPY"], start="2005-09-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    rows = []
    for y in range(2006, 2026):
        i = px.index.searchsorted(recon_day(y))
        if px.index[i] != recon_day(y):
            i -= 1  # holiday: last trading day before
        r = lambda a, b, j0, j1: px[a].iloc[j1] / px[a].iloc[j0] - px[b].iloc[j1] / px[b].iloc[j0] - 0.002
        rows.append({"year": y, "H1 IWC-IWM R..R+10": r("IWC", "IWM", i, i + 10),
                     "H2 IWC-IWM R-15..R": r("IWC", "IWM", i - 15, i),
                     "H3 IWM-SPY R..R+10": r("IWM", "SPY", i, i + 10)})
    d = pd.DataFrame(rows).set_index("year")
    print((d * 100).round(2).to_string())
    for lab, g in (("2006-15", d.loc[:2015]), ("2016-25", d.loc[2016:])):
        print(lab, "  ".join(f"{c.split()[0]} {g[c].mean() * 100:+.2f}% t {g[c].mean() / g[c].std() * np.sqrt(len(g)):+.2f}"
                             for c in d.columns))


if __name__ == "__main__":
    main()
