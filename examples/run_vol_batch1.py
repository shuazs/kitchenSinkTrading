"""Volatility-structure scan, batch 1 (daily Yahoo index-vol data).

Indicators (all known at the close of day t, trade at the open of t+1):
VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, VXN, VXD, OVX, GVZ and SPY realised vol.
Each is turned into an expanding-window percentile (min 3 years, no look-ahead)
and every (indicator, target, side, threshold) rule "hold the target while the
percentile is above/below q, else BIL" is scored at 3 bp per side on each flip.

The scan is judged the protocol way (docs/guide/protocol.md): the best rule is
chosen on data before 2024-07 only, then reported on the Jul 2024+ holdout;
the deflated Sharpe uses the number of rules tried. A second part checks
specific stories: VIX spike buys in 3x ETFs, VIXM as a tail hedge sleeve,
FOMC-window vol ETPs, and the VXN/VIX growth-vs-index rotation.

    python examples/run_vol_batch1.py [scan|spikes|hedge|fomc|rotation|all]
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from backtester.ml.contest import deflated_sharpe

warnings.filterwarnings("ignore")
HOLD = pd.Timestamp("2024-07-01")
SPLITS = [("2007", "2012"), ("2013", "2018"), ("2019", "2024-06"), ("2024-07", "2026")]
COST = 3e-4  # per side, ETF presets are 2+2 bp
VOL = "^VIX ^VIX9D ^VIX3M ^VIX6M ^VVIX ^SKEW ^VXN ^VXD ^OVX ^GVZ".split()
ETF = "SPY QQQ IWM TLT GLD USO BIL UPRO TQQQ SVXY VIXY VIXM XLE XLU".split()


def download(syms, start="2005-01-01"):
    d = yf.download(syms, start=start, progress=False, auto_adjust=True, group_by="column")
    return d["Open"], d["Close"]


def sharpe(r, ppy=252):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(ppy) if len(r) > 20 and r.std() > 0 else np.nan


def cagr(r):
    r = r.dropna()
    return (1 + r).prod() ** (252 / len(r)) - 1 if len(r) > 20 else np.nan


def maxdd(r):
    e = (1 + r.dropna()).cumprod()
    return (e / e.cummax() - 1).min()


def pct_rank(s, minp=756):
    """Expanding percentile of today's value among values up to and including today."""
    s = s.dropna()
    return s.expanding(minp).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)


def oo_returns(o):
    """Return from open t+1 to open t+2, indexed at t (signal day)."""
    return (o.shift(-2) / o.shift(-1) - 1)


def excess(r, cash):
    return (r - cash.reindex(r.index)).dropna()


def run_rule(on: pd.Series, tgt_oo: pd.Series, cash_oo: pd.Series) -> pd.Series:
    """on: bool at close t. Position from open t+1 to t+2; flip costs COST per side."""
    on = on.astype(float).fillna(0.0)
    flips = on.diff().abs().fillna(0.0)
    r = on * tgt_oo + (1 - on) * cash_oo - flips * COST * 2  # a flip sells one, buys the other
    return r.dropna()


def build():
    o, c = download(VOL + ETF)
    spy_o = o["SPY"]
    rv = np.log(c["SPY"]).diff().rolling(21).std() * np.sqrt(252) * 100
    ind = pd.DataFrame(index=c.index)
    ind["vix"] = c["^VIX"]
    ind["vvix"] = c["^VVIX"]
    ind["vvix_vix"] = c["^VVIX"] / c["^VIX"]
    ind["skew"] = c["^SKEW"]
    ind["skew_chg5"] = c["^SKEW"].diff(5)
    ind["vix9d_vix"] = c["^VIX9D"] / c["^VIX"]
    ind["vix_vix3m"] = c["^VIX"] / c["^VIX3M"]
    ind["vix3m_vix6m"] = c["^VIX3M"] / c["^VIX6M"]
    ind["vxn_vix"] = c["^VXN"] / c["^VIX"]
    ind["vxd_vix"] = c["^VXD"] / c["^VIX"]
    ind["vix_rv"] = c["^VIX"] - rv
    ind["vix_rv_ratio"] = c["^VIX"] / rv
    ind["ovx"] = c["^OVX"]
    ind["gvz"] = c["^GVZ"]
    ind["vix_chg1"] = c["^VIX"].pct_change()
    ind["vix_z20"] = (c["^VIX"] - c["^VIX"].rolling(20).mean()) / c["^VIX"].rolling(20).std()
    ind = ind.dropna(how="all")
    return o, c, ind


def scan():
    o, c, ind = build()
    oo = oo_returns(o)
    pr = ind.apply(pct_rank).reindex(ind.index)
    targets = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UPRO"]
    rows, rets = [], {}
    for name, t, side, q in itertools.product(pr.columns, targets, ("above", "below"), (0.2, 0.5, 0.8)):
        if name in ("ovx",) and t not in ("USO", "SPY"):
            continue
        if name == "gvz" and t not in ("GLD", "SPY"):
            continue
        on = (pr[name] > q) if side == "above" else (pr[name] < q)
        on = on.where(pr[name].notna())
        r = excess(run_rule(on, oo[t], oo["BIL"]), oo["BIL"])
        r = r["2007":]
        base = excess(oo[t], oo["BIL"]).reindex(r.index)
        if on.dropna().mean() < 0.05:
            continue
        key = (name, t, side, q)
        rets[key] = r
        tr, ho = r[:"2024-06"], r["2024-07":]
        bt, bh = base[:"2024-06"], base["2024-07":]
        rows.append(dict(ind=name, target=t, side=side, q=q, sh_tr=sharpe(tr), bh_tr=sharpe(bt),
                         sh_ho=sharpe(ho), bh_ho=sharpe(bh), exp=on.dropna().mean(), flips=on.dropna().astype(float).diff().abs().sum() / on.notna().sum() * 252))
    df = pd.DataFrame(rows)
    df["gain_tr"] = df.sh_tr - df.bh_tr
    df["gain_ho"] = df.sh_ho - df.bh_ho
    n = len(df)
    print(f"{n} rules tried")
    print("share of rules beating buy&hold Sharpe: train %.2f holdout %.2f" % ((df.gain_tr > 0).mean(), (df.gain_ho > 0).mean()))
    print("corr(train gain, holdout gain) = %.2f" % df[["gain_tr", "gain_ho"]].corr().iloc[0, 1])
    top = df.sort_values("gain_tr", ascending=False).head(15)
    pd.set_option("display.width", 200)
    print(top.round(2).to_string())
    # deflated Sharpe of the train-best rule per target
    for t in ("SPY", "QQQ", "TLT", "GLD", "USO", "UPRO"):
        d = df[df.target == t].sort_values("gain_tr", ascending=False).iloc[0]
        key = (d.ind, d.target, d.side, d.q)
        r = rets[key]
        tr = r[:"2024-06"]
        sub = " ".join(f"{a}:{sharpe(r[a:b]) - sharpe(excess(oo[t], oo["BIL"])[a:b]):+.2f}" for a, b in SPLITS)
        try:
            dsr = deflated_sharpe(tr, n_trials=n)
        except Exception as e:  # signature differs: fall back to raw
            dsr = float("nan")
        print(f"best {t}: {key} train gain {d.gain_tr:+.2f} holdout gain {d.gain_ho:+.2f} (holdout sharpe {d.sh_ho:.2f} vs {d.bh_ho:.2f}) DSR {dsr} subperiod gains {sub}")
    df.to_csv("/tmp/claude/vol_scan.csv", index=False)
    return df


def spikes():
    """Buy after a VIX spike. Entry at the next open, hold h days; 3 bp each way."""
    o, c, ind = build()
    vix = c["^VIX"]
    rows = []
    for tgt in ("SPY", "UPRO", "QQQ", "TQQQ"):
        px = o[tgt]
        for trig, h in itertools.product(("chg30", "z3", "lvl35", "peak"), (1, 3, 5, 10, 21)):
            if trig == "chg30":
                m = vix.pct_change() > 0.30
            elif trig == "z3":
                m = ind["vix_z20"] > 3
            elif trig == "lvl35":
                m = (vix > 35) & (vix.pct_change() < 0)  # >35 and falling from the prior close
            else:
                m = (vix == vix.rolling(60).max()) & (vix > 30)
            idx = np.flatnonzero(m.to_numpy())
            ev, last = [], -h
            for i in idx:
                if i - last < h or i + 1 + h >= len(px):
                    continue
                ev.append((px.index[i], px.iloc[i + 1 + h] / px.iloc[i + 1] - 1 - 2 * COST))
                last = i
            if not ev:
                continue
            e = pd.Series(dict(ev))
            base = (px.shift(-h) / px - 1).dropna()
            rows.append(dict(target=tgt, trigger=trig, hold=h, n=len(e), mean_pct=e.mean() * 100, base_pct=base.mean() * 100,
                             excess_pct=(e.mean() - base.mean()) * 100, t=e.mean() / e.std() * np.sqrt(len(e)) if len(e) > 2 else np.nan,
                             hit=(e > 0).mean(), pre24=e[:"2024-06"].mean() * 100, post24=e["2024-07":].mean() * 100, n_post=len(e["2024-07":])))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string())
    return df


def hedge():
    """Does a small long-vol sleeve (VIXM, VIXY) improve a SPY or Balanced-like mix after costs?"""
    o, c, _ = build()
    r = c[["SPY", "VIXM", "VIXY", "TLT", "GLD", "BIL"]].pct_change().dropna()
    r = r["2011":]
    out = []
    for w in (0.0, 0.03, 0.05, 0.10):
        for v in ("VIXM", "VIXY"):
            # monthly rebalanced SPY/vol mix, costs 3 bp on turnover (approximate, drifting weights ignored)
            wts = pd.DataFrame({"SPY": 1 - w, v: w}, index=r.index)
            port = (wts * r[["SPY", v]]).sum(axis=1)
            port -= COST * 2 * w * (1 / 21) * 2  # monthly rebalance turnover ~ 2w
            out.append(dict(vol=v, w=w, cagr=cagr(port), dd=maxdd(port), sh=sharpe(port),
                            cagr_ho=cagr(port["2024-07":]), sh_ho=sharpe(port["2024-07":])))
    print(pd.DataFrame(out).round(3).to_string())


def fomc():
    """Long vol (VIXY/VIXM) or short vol (SVXY) around FOMC statement days; vol crush into/after the event."""
    from backtester.data.fomc import statement_dates
    o, c, _ = build()
    dates = pd.DatetimeIndex(statement_dates())
    rows = []
    for tgt in ("SVXY", "VIXY", "SPY"):
        px = o[tgt]
        for pre, post in itertools.product((0, 1, 2, 3), (0, 1, 2)):
            ev = []
            for d in dates:
                if d not in px.index:
                    continue
                i = px.index.get_loc(d)
                a, b = i - pre, i + post + 1  # buy at open pre days before, sell at open post+1 days after
                if a < 0 or b >= len(px) or np.isnan(px.iloc[a]) or np.isnan(px.iloc[b]):
                    continue
                ev.append((d, px.iloc[b] / px.iloc[a] - 1 - 2 * COST))
            e = pd.Series(dict(ev))
            if len(e) < 10:
                continue
            e = e["2011":]
            rows.append(dict(target=tgt, pre=pre, post=post, n=len(e), mean_bp=e.mean() * 1e4, t=e.mean() / e.std() * np.sqrt(len(e)),
                             pre24_bp=e[:"2024-06"].mean() * 1e4, post24_bp=e["2024-07":].mean() * 1e4, n_post=len(e["2024-07":])))
    df = pd.DataFrame(rows)
    print(df.round(2).to_string())


def rotation():
    """QQQ vs SPY by the tech-vol premium VXN/VIX; and IWM vs SPY by VXD/VIX (rank on expanding window)."""
    o, c, ind = build()
    oo = oo_returns(o)
    pr = ind.apply(pct_rank).reindex(ind.index)
    rows = []
    for name, a, b in (("vxn_vix", "QQQ", "SPY"), ("vxd_vix", "SPY", "IWM"), ("vxn_vix", "SPY", "QQQ")):
        for q in (0.2, 0.5, 0.8):
            for side in ("above", "below"):
                on = (pr[name] > q) if side == "above" else (pr[name] < q)
                on = on.where(pr[name].notna())
                r = run_rule(on, oo[a], oo[b])["2007":]
                rows.append(dict(sig=name, long_if_on=a, else_=b, side=side, q=q, sh=sharpe(r), sh_tr=sharpe(r[:"2024-06"]), sh_ho=sharpe(r["2024-07":]),
                                 bh_a_ho=sharpe(oo[a]["2024-07":]), bh_b_ho=sharpe(oo[b]["2024-07":])))
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("scan", scan), ("spikes", spikes), ("hedge", hedge), ("fomc", fomc), ("rotation", rotation)):
        if what in (name, "all"):
            print(f"\n=== {name} ===")
            fn()


def price_vs_vol():
    """Is the VIX z>3 snapback just a price drop? Compare with price-only triggers on SPY (hold 5/7 days)."""
    o, c, ind = build()
    spy = o["SPY"]
    cl = c["SPY"]
    rows = []
    for name, m in (("vix_z>3", ind["vix_z20"] > 3),
                    ("spy_3d<-3%", cl.pct_change(3) < -0.03),
                    ("spy_5d<-5%", cl.pct_change(5) < -0.05),
                    ("spy_1d<-2.5%", cl.pct_change() < -0.025),
                    ("vix_z>3 & spy_3d<-3%", (ind["vix_z20"] > 3) & (cl.pct_change(3) < -0.03)),
                    ("vix_z>3 & spy_3d>=-3%", (ind["vix_z20"] > 3) & (cl.pct_change(3) >= -0.03))):
        for h in (5, 7):
            ev, last = [], -h
            for i in np.flatnonzero(m.fillna(False).to_numpy()):
                if i - last < h or i + 1 + h >= len(spy):
                    continue
                ev.append((spy.index[i], spy.iloc[i + 1 + h] / spy.iloc[i + 1] - 1 - 2 * COST))
                last = i
            e = pd.Series(dict(ev))
            rows.append(dict(trigger=name, h=h, n=len(e), mean_pct=e.mean() * 100, t=e.mean() / e.std() * np.sqrt(len(e)),
                             pre24=e[:"2024-06"].mean() * 100, post24=e["2024-07":].mean() * 100, n_post=len(e["2024-07":])))
    print(pd.DataFrame(rows).round(2).to_string())


def weekday_vol_etps():
    """SVXY/VIXY by weekday, split overnight (close->next open) vs day (open->close). 2011+ (SVXY halved to 0.5x in Feb 2018)."""
    d = yf.download(["SVXY", "VIXY", "SPY"], start="2011-01-01", progress=False, auto_adjust=True)
    o, c = d["Open"], d["Close"]
    rows = []
    for s in ("SVXY", "VIXY", "SPY"):
        night = (o[s].shift(-1) / c[s] - 1)  # close t -> open t+1, indexed t
        day = c[s] / o[s] - 1
        for part, r in (("night", night), ("day", day)):
            g = r.groupby(r.index.dayofweek)
            for wd, x in g:
                x = x.dropna()
                rows.append(dict(sym=s, part=part, weekday=wd, n=len(x), mean_bp=x.mean() * 1e4, t=x.mean() / x.std() * np.sqrt(len(x)),
                                 pre24_bp=x[:"2024-06"].mean() * 1e4, post24_bp=x["2024-07":].mean() * 1e4))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("price_vs_vol", "weekday"):
    {"price_vs_vol": price_vs_vol, "weekday": weekday_vol_etps}[sys.argv[1]]()
