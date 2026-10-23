"""Alpha hunt batch 3: scheduled and forced flows in liquid ETFs and BTC.

Each test is an event window around a date known in advance (market-on-close
at both edges): per-event return of the asset or spread in 2008-16,
2017-Jun 2024 and the Jul 2024+ holdout, plus a placebo with the dates
shifted 10-15 trading days. Kill rule: t >= 2 in the fit period, same sign
and >= half the size in the holdout, and larger than round-trip costs
(about 6 bp per ETF leg).

C1 GSCI roll (business days 5-9), USO-USL and UNG-UNL
C2 Russell reconstitution, IWM-IWB
C3 2/5/7y, 3y and 20y Treasury auction cycles (IEI/IEF/TLT; 10y/30y is B6)
C4 Corporate tax dates (15th of Apr/Jun/Sep/Dec), SPY and TLT
C5 CME bitcoin futures expiry (last Friday), BTC-USD
C6 MSCI rebalance (last business day of Feb/May/Aug/Nov), EEM-SPY, EFA-SPY
C7 Treasury coupon dates (15th), TLT
C8 VIX futures expiry, VIXY and SPY
C9 Quarter-end window dressing, MTUM/USMV/QUAL minus SPY
"""
import pandas as pd
import yfinance as yf

from backtester import auction_cycle as ac
from backtester import event_windows as ew
from backtester.data.treasury_auctions import load_auctions

TICKERS = "USO USL UNG UNL IWM IWB SPY TLT IEF IEI BIL EEM EFA BTC-USD VIXY MTUM USMV QUAL".split()


def show(name, close, dates, wins, bench=None, how="on_or_after"):
    c = close.dropna()
    b = bench.reindex(c.index).ffill() if bench is not None else None
    an = ew.anchor_positions(c.index, dates, how)
    for s, e in wins:
        r = ew.event_returns(c, an, s, e, b)
        st = ew.split_stats(r)
        pl = ew.placebo(c, an, s, e, bench=b).round(0).tolist()
        print(f"{name:22s} ({s:+d},{e:+d}) " + " ".join(
            f"{i} {st.loc[i, 'mean_bp']:5.0f}bp t{st.loc[i, 't']:5.1f}" for i in st.index) + f"  placebo {pl}")


def main():
    px = yf.download(TICKERS, start="2001-01-01", auto_adjust=True, progress=False)["Close"]
    idx = px["SPY"].dropna().index
    years = range(2001, idx[-1].year + 1)
    lbd = ew.nth_business_day(idx, -1)
    W = [(-5, 0), (-1, 0), (0, 1), (0, 5)]

    d5 = ew.nth_business_day(idx, 5)
    show("C1 USO-USL roll", px["USO"], d5, [(-1, 4)], px["USL"])
    show("C1 UNG-UNL roll", px["UNG"], d5, [(-1, 4)], px["UNL"])

    show("C2 IWM-IWB recon", px["IWM"], [ew.russell_recon(y) for y in years], [(-5, 0), (0, 5)], px["IWB"], "on_or_before")

    a = load_auctions()
    cash = px["BIL"]
    for name, pat, typ, etf in [("2/5/7y", r"(2-Year|5-Year|7-Year)", "Note", "IEF"), ("3y", r"3-Year", "Note", "IEI"),
                                ("20y", r"(19-Year|20-Year)", "Bond", "TLT")]:
        dates = a[(a["type"] == typ) & a["term"].str.match(pat)]["date"].unique()
        c = px[etf].dropna()
        r = ac.window_returns(c, ac.cycles(dates, c.index), 3, cash.reindex(c.index).ffill())
        st = ew.split_stats(r)
        print(f"C3 {name:6s} {etf} 3d cycle " + " ".join(f"{i} {st.loc[i, 'mean_bp']:5.0f}bp t{st.loc[i, 't']:5.1f}" for i in st.index))

    tax = [pd.Timestamp(y, m, 15) for y in years for m in (4, 6, 9, 12)]
    show("C4 tax SPY-BIL", px["SPY"], tax, [(-3, 0), (0, 3)], cash)
    show("C4 tax TLT-BIL", px["TLT"], tax, [(-3, 0), (0, 3)], cash)

    btc = px["BTC-USD"].dropna()
    show("C5 BTC CME expiry", btc, [ew.last_friday(y, m) for y in range(2018, idx[-1].year + 1) for m in range(1, 13)],
         [(-3, 0), (0, 3)])

    msci = [d for d in lbd if d.month in (2, 5, 8, 11)]
    show("C6 EEM-SPY MSCI", px["EEM"], msci, W, px["SPY"], "on_or_before")
    show("C6 EEM-SPY all m-ends", px["EEM"], list(lbd), [(0, 1)], px["SPY"], "on_or_before")
    show("C6 EFA-SPY MSCI", px["EFA"], msci, [(-5, 0)], px["SPY"], "on_or_before")

    show("C7 TLT coupon 15th", px["TLT"], [pd.Timestamp(y, m, 15) for y in years for m in range(1, 13)], [(-3, 0), (0, 3)], cash)

    vexp = [ew.vix_expiry(y, m) for y in range(2011, idx[-1].year + 1) for m in range(1, 13)]
    show("C8 VIXY VIX expiry", px["VIXY"], vexp, [(-5, 0), (0, 1)])
    show("C8 SPY VIX expiry", px["SPY"], vexp, [(-5, 0), (0, 1)], cash)

    qe = [d for d in lbd if d.month in (3, 6, 9, 12)]
    for f in ("MTUM", "USMV", "QUAL"):
        show(f"C9 {f}-SPY q-end", px[f], qe, [(-5, 0), (0, 1)], px["SPY"], "on_or_before")


if __name__ == "__main__":
    main()
