"""Single-stock option signals (IV vs HV, smile, put/call, IV changes), Oct 2019 - Jul 2023.

Data: `data/options_iv.load_iv()` (HF gauss314/options-IV-SP500, ~1,000 liquid tickers
with median option volume > 800). Each signal is the cross-sectional rank on day t
(close); positions open at the open of t+1 and close h trading days later (h = 5, 21
with non-overlapping rebalances). Reported per signal: top and bottom decile mean
return, long-short spread, long-only excess vs the equal-weighted universe, t-stats
across rebalances, sub-period split (fit Oct 2019-Dec 2021, test 2022-Jul 2023) and a
10 bp per side cost on the long leg's turnover. Survivorship: Yahoo has no delisted
names, so the universe is today's survivors.

    python examples/run_options_iv.py
"""
import numpy as np
import pandas as pd

from backtester.data.options_iv import load_iv
from backtester.data.yahoo import load_bars
from backtester.data.yahoo_batch import _path, prefetch

SPLIT = pd.Timestamp("2022-01-01")
COST = 10e-4  # per side


def prices(symbols):
    o, c = {}, {}
    for s in symbols:
        try:
            b = load_bars(s, "2019-01-01", "2023-12-31")
        except Exception:
            continue
        if len(b) > 500:
            o[s], c[s] = b["open"], b["close"]
    return pd.DataFrame(o), pd.DataFrame(c)


def signals(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = lambda col: d.pivot(index="date", columns="symbol", values=col)
    iv, hv20, hv60 = p("ATM_IV"), p("hv_20"), p("hv_60")
    cv, pv = p("calls_contracts_traded"), p("puts_contracts_traded")
    coi, poi = p("calls_open_interest"), p("puts_open_interest")
    dotm, ditm = p("DOTM_IV"), p("DITM_IV")
    vol = cv + pv
    out = {
        "iv_minus_hv20": iv - hv20,
        "iv_over_hv20": iv / hv20,
        "iv_minus_hv60": iv - hv60,
        "iv_chg5": iv / iv.shift(5) - 1,
        "iv_chg1": iv / iv.shift(1) - 1,
        "iv_rank250": iv.rolling(250, min_periods=120).rank(pct=True),
        "wing_dotm": dotm / iv,
        "wing_ditm": ditm / iv,
        "pc_vol": pv / vol.where(vol > 0),
        "pc_oi": poi / (poi + coi),
        "put_vol_abn": np.log1p(pv) - np.log1p(pv.rolling(20, min_periods=10).mean().shift(1)),
        "call_vol_abn": np.log1p(cv) - np.log1p(cv.rolling(20, min_periods=10).mean().shift(1)),
        "vol_over_oi": vol / (coi + poi).where((coi + poi) > 0),
        "pc_vol_chg": (pv / vol.where(vol > 0)) - (pv / vol.where(vol > 0)).rolling(20, min_periods=10).mean().shift(1),
    }
    return out


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 3 and x.std() > 0 else np.nan


def evaluate(sig: pd.DataFrame, o: pd.DataFrame, h: int, q: float = 0.1, min_names: int = 200) -> pd.DataFrame:
    """Rebalance every h days. Row per rebalance: top, bottom, universe mean returns (open t+1 -> open t+1+h)."""
    sig = sig.reindex(columns=o.columns)
    dates = [d for d in sig.index if d in o.index]
    rows = []
    for k in range(0, len(dates) - 1, h):
        d = dates[k]
        i = o.index.get_loc(d)
        if i + 1 + h >= len(o):
            break
        r = o.iloc[i + 1 + h] / o.iloc[i + 1] - 1
        s = sig.loc[d]
        ok = s.notna() & r.notna()
        if ok.sum() < min_names:
            continue
        s, r = s[ok], r[ok]
        rk = s.rank(pct=True)
        rows.append(dict(date=d, top=r[rk > 1 - q].mean(), bot=r[rk <= q].mean(), uni=r.mean(), n=ok.sum()))
    df = pd.DataFrame(rows).set_index("date")
    df["ls"] = df.top - df.bot
    return df


def report():
    d = load_iv()
    syms = sorted(d[(d.calls_contracts_traded + d.puts_contracts_traded).groupby(d.symbol).transform("median") > 800].symbol.unique())
    prefetch(syms)  # one threaded pass; symbols Yahoo lacks (delisted) are skipped
    syms = [s for s in syms if _path(s, "1d").exists()]
    o, c = prices(syms)
    print(f"{o.shape[1]} tickers with prices; IV rows {len(d):,}")
    sg = signals(d)
    rows = []
    for name, s in sg.items():
        for h in (5, 21):
            e = evaluate(s, o, h)
            if len(e) < 10:
                continue
            side = {}
            for lab, part in (("all", e), ("fit", e[e.index < SPLIT]), ("test", e[e.index >= SPLIT])):
                side[f"ls_{lab}"] = part.ls.mean() * 1e4
                side[f"t_{lab}"] = tstat(part.ls)
                side[f"topx_{lab}"] = (part.top - part.uni).mean() * 1e4
                side[f"botx_{lab}"] = (part.bot - part.uni).mean() * 1e4
            # cost on the long top leg: full turnover each rebalance
            side["topx_net_all"] = ((e.top - e.uni) - 2 * COST).mean() * 1e4
            rows.append(dict(signal=name, h=h, n_reb=len(e), **side))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.round(1).to_string())
    return out




ETFS = "SPY QQQ IWM EEM HYG EFA SLV XLF GLD TLT XLE XLK XLU XLV XLI XLP XLY XLB XLRE XBI KRE SMH XOP GDX GDXJ USO UNG FXI EWZ EWJ LQD IEF TQQQ SQQQ UVXY SVXY VXX ARKK XRT ITB JETS KWEB IBB".split()


def etf_sleeve():
    """Cross-sectional IV signals inside ~40 liquid ETFs (min 20 names per rebalance)."""
    d = load_iv()
    d = d[d.symbol.isin(ETFS)]
    prefetch(sorted(d.symbol.unique()))
    o, _ = prices(sorted(d.symbol.unique()))
    sg = signals(d)
    rows = []
    for name, s in sg.items():
        for h in (5, 21):
            e = evaluate(s, o, h, q=0.2, min_names=20)
            if len(e) < 10:
                continue
            rows.append(dict(signal=name, h=h, n_reb=len(e), ls_bp=e.ls.mean() * 1e4, t=tstat(e.ls),
                             fit_t=tstat(e.ls[e.index < SPLIT]), test_t=tstat(e.ls[e.index >= SPLIT]),
                             topx_bp=(e.top - e.uni).mean() * 1e4))
    print(pd.DataFrame(rows).round(1).to_string())


def market_level():
    """Market-wide aggregates from the panel as SPY timing signals (expanding percentile, min 120d)."""
    d = load_iv()
    d = d[~d.symbol.isin(ETFS)]
    g = d.groupby("date")
    m = pd.DataFrame({
        "pc_vol": g.puts_contracts_traded.sum() / (g.puts_contracts_traded.sum() + g.calls_contracts_traded.sum()),
        "pc_oi": g.puts_open_interest.sum() / (g.puts_open_interest.sum() + g.calls_open_interest.sum()),
        "med_iv_hv": g.apply(lambda x: (x.ATM_IV - x.hv_20).median()),
        "med_iv_over_vix": g.apply(lambda x: x.ATM_IV.median() / x.VIX.iloc[0]),
        "med_smile": g.apply(lambda x: (x.DOTM_IV / x.ATM_IV).median()),
        "vol_oi": g.apply(lambda x: (x.calls_contracts_traded + x.puts_contracts_traded).sum() / (x.calls_open_interest + x.puts_open_interest).sum()),
    })
    import yfinance as yf

    px = yf.download(["SPY", "BIL"], start="2019-06-01", end="2023-09-01", progress=False, auto_adjust=True)
    spy_o, bil_o = px["Open"]["SPY"], px["Open"]["BIL"]
    f = lambda o: o.shift(-2) / o.shift(-1) - 1
    rows = []
    for name in m.columns:
        pr = m[name].expanding(120).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
        for side in ("above", "below"):
            for q in (0.2, 0.5, 0.8):
                on = ((pr > q) if side == "above" else (pr < q)).where(pr.notna()).reindex(spy_o.index).ffill()
                on = on.astype(float)
                r = on * f(spy_o) + (1 - on) * f(bil_o) - on.diff().abs().fillna(0) * 6e-4
                r = r.dropna()
                base = f(spy_o).reindex(r.index)
                sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
                rows.append(dict(signal=name, side=side, q=q, exposure=on.mean(), sh_fit=sh(r[r.index < SPLIT]), bh_fit=sh(base[base.index < SPLIT]),
                                 sh_test=sh(r[r.index >= SPLIT]), bh_test=sh(base[base.index >= SPLIT])))
    df = pd.DataFrame(rows)
    df["gain_fit"] = df.sh_fit - df.bh_fit
    df["gain_test"] = df.sh_test - df.bh_test
    print(df.round(2).sort_values("gain_fit", ascending=False).to_string())
    print("share of rules beating buy&hold: fit %.2f test %.2f; corr %.2f" % ((df.gain_fit > 0).mean(), (df.gain_test > 0).mean(), df[["gain_fit", "gain_test"]].corr().iloc[0, 1]))


if __name__ == "__main__":
    import sys

    for name, fn in (("stocks", report), ("etfs", etf_sleeve), ("market", market_level)):
        if len(sys.argv) < 2 or sys.argv[1] in (name, "all"):
            print(f"\n=== {name} ===")
            fn()
