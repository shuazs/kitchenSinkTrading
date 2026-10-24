"""Alpha hunt batch 11: ideas built on this repo's own keepers and near misses.

Each test prints sub-periods / holdouts so the verdicts in
docs/guide/strategies.md can be re-checked:

- Polymarket replication of two Kalshi near misses (HF SII-WANGZJ fills,
  2024+): "will X say Y" markets, buy No at the first fill with YES at
  20-80c; tennis match winners, buy the favourite at 80-95c. Gross, net of
  the Polymarket US fee 5% p(1-p), and t-stats clustered by event;
- Polymarket theme jumps -> assets (the Iran -> oil rule from jump_trades.py
  on new pairs): Ukraine ceasefire -> EZU / wheat, Taiwan invasion -> SMH /
  TSM, US shutdown -> TLT / SPY, tariffs -> FXI / KWEB / EWW / EWC;
- target-date-fund rebalancing: SPY after stocks lag bonds (bottom decile
  of SPY-AGG over 5/10/20 days);
- January days 6-10 prior-year winners minus losers in country ETFs,
  sector ETFs and crypto (and the commodity futures re-check);
- BTC into month-end after a down month, before and after the spot ETFs;
- single-stock LETF reversal sharpened to top-decile pressure (threshold
  frozen on Jan 2024-Jun 2025).
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from run_alpha_batch8 import HTTPFile  # noqa: E402
from run_alpha_batch10 import LETF, fmt, tstat  # noqa: E402

from backtester.data.cache import cache_dir  # noqa: E402
from backtester.data.polymarket import HF_REPO, escalation_index, hourly_prices, load_markets  # noqa: E402
from backtester.data.yahoo import load_hourly  # noqa: E402
from backtester.jump_trades import jump_events, jump_trades, summarize  # noqa: E402

HALVES = [("2024-01", "2025-06"), ("2025-07", "2026-12")]
THEMES = {"ukraine": ({"EZU": 1, "ZW=F": -1}, r"ukraine|russia x ukraine"), "taiwan": ({"SMH": -1, "TSM": -1}, r"taiwan"),
          "shutdown": ({"TLT": 1, "SPY": -1}, r"government shutdown|shutdown"), "tariff": ({"FXI": -1, "KWEB": -1, "EWW": -1, "EWC": -1}, r"tariff")}


def theme_sign(theme: str, q: str) -> int:
    """+1 when YES means peace (Ukraine), conflict (Taiwan), a shutdown, or higher tariffs; -1 for tariff cuts."""
    q = q.lower()
    if theme == "ukraine":
        return int(bool(re.search(r"ceasefire|ends? .*war|peace|mineral deal", q)))
    if theme == "taiwan":
        return int(bool(re.search(r"invade|blockade|clash", q)))
    if theme == "shutdown":
        return int(bool(re.search(r"shutdown (saturday|by|in 20)|will there be a us government shutdown|government shutdown on", q)))
    if "between" in q or "less than" in q or "fart" in q:
        return 0
    if re.search(r"lower|reduc|remove|agreement|deal|pause|cut", q):
        return -1
    return int(bool(re.search(r"impose|increase|in effect|large tariffs|rules in favor|go into effect", q)))


def pm_selection() -> pd.DataFrame:
    """Mention, tennis match-winner and theme markets, with grp and the YES outcome (y1)."""
    m = load_markets()
    q, es, et = m.question.str.lower(), m.event_slug.fillna("").str.lower(), m.event_title.fillna("").str.lower()
    m["grp"] = ""
    m.loc[q.str.contains(r"\bsay\b|\bmention", regex=True) & (m.volume >= 5000), "grp"] = "mention"
    ten = ((es.str.match(r"^(atp|wta)-") | et.str.contains(r"\batp\b|\bwta\b|wimbledon|roland garros|australian open", regex=True))
           & (m.volume >= 5000) & q.str.contains(" vs") & ~m.answer1.isin(["Yes", "No", "Over", "Under"])
           & ~q.str.contains("handicap|o/u|total|set "))
    m.loc[ten, "grp"] = "tennis"
    for k, (_, pat) in THEMES.items():
        m.loc[(m.grp == "") & q.str.contains(pat, regex=True) & (m.volume >= 2e5), "grp"] = k
    s = m[m.grp != ""].copy()
    s["id"] = s.id.astype(str)
    op = s.outcome_prices.apply(lambda v: [float(x) for x in ast.literal_eval(v)] if isinstance(v, str) else [])
    s["y1"] = op.apply(lambda v: v[0] if len(v) == 2 and v[0] in (0.0, 1.0) else np.nan)
    s.loc[s.closed != 1, "y1"] = np.nan
    return s


def pm_fills(ids, name: str = "fills_batch11") -> pd.DataFrame:
    """Fills since 2024 for the selected markets (one ~25-minute range-read pass over quant.parquet, cached)."""
    path = cache_dir("polymarket") / f"{name}.parquet"
    if not path.exists():
        ids = pa.array(sorted(ids))
        pf = pq.ParquetFile(HTTPFile(HF_REPO + "quant.parquet"))
        t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
        parts = []
        for i in range(pf.metadata.num_row_groups):
            st = pf.metadata.row_group(i).column(0).statistics
            if st is not None and st.max < t0:
                continue
            mask = pc.is_in(pf.read_row_group(i, columns=["market_id"])["market_id"], value_set=ids)
            if pc.any(mask).as_py():
                parts.append(pf.read_row_group(i, columns=["timestamp", "market_id", "price", "usd_amount", "side"]).filter(mask))
        pq.write_table(pa.concat_tables(parts), path)
    f = pd.read_parquet(path)
    f["market_id"] = f.market_id.astype(str)
    f["time"] = pd.to_datetime(f.pop("timestamp"), unit="s")
    return f.sort_values("time", ignore_index=True)


def report(name: str, r: pd.DataFrame):
    r = r.assign(net=r.pnl - 0.05 * r.px * (1 - r.px))
    ev = r.groupby("event_slug").net.mean()
    print(f"{name}: gross {fmt(r.set_index('time').pnl.sort_index(), HALVES)} | net of US fee {fmt(r.set_index('time').net.sort_index(), HALVES)}"
          f" | net clustered by event {ev.mean() * 100:.2f}c t{tstat(ev):.2f} events {len(ev)}")


def pm_near_misses(S: pd.DataFrame, F: pd.DataFrame):
    f = F[F.usd_amount >= 20].merge(S[["id", "grp", "y1", "question", "event_slug"]].dropna(subset=["y1"]), left_on="market_id", right_on="id")
    # entries are taker fills on the side we buy: side SELL = the taker sold YES at the bid, i.e. bought No at 1 - price
    m = f[(f.grp == "mention") & f.price.between(0.2, 0.8) & (f.side == "SELL")].groupby("market_id").first().reset_index()
    m["px"], m["pnl"] = 1 - m.price, m.y1.rsub(1) - (1 - m.price)
    report("Polymarket mentions, buy No at YES 20-80c", m)
    trump = m.question.str.contains("Trump")
    report("  Trump markets", m[trump])
    report("  other speakers", m[~trump])
    for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8)):
        report(f"  No price {lo:.1f}-{hi:.1f}", m[m.px.between(lo, hi)])
    t = f[(f.grp == "tennis") & ((f.price.between(0.8, 0.95) & (f.side == "BUY")) | (f.price.between(0.05, 0.2) & (f.side == "SELL")))]
    t = t.groupby("market_id").first().reset_index()
    fav = t.price >= 0.5
    t["px"] = np.where(fav, t.price, 1 - t.price)
    t["pnl"] = np.where(fav, t.y1, 1 - t.y1) - t.px
    report("Polymarket tennis, buy the favourite at 80-95c", t)


def pm_theme_jumps(S: pd.DataFrame, F: pd.DataFrame):
    S = S.assign(sign=[theme_sign(g, q) if g in THEMES else 0 for g, q in zip(S.grp, S.question)])
    pooled = []
    for th, (assets, _) in THEMES.items():
        s = S[(S.grp == th) & (S.sign != 0)]
        p, v = hourly_prices(F[F.market_id.isin(s.id)])
        idx = escalation_index(p, v, s.set_index("id").sign)
        for thr in (0.03, 0.05, 0.08):
            ev = jump_events(idx, thr)
            for a, d in assets.items():
                bars = load_hourly(a)
                e = ev.assign(direction=ev.direction * d)
                e = e[e.time >= bars.index.min()]
                if len(jump_trades(bars, e, hold=3, delay=1)) < 2:
                    print(f"Theme {th} -> {a}, jump >= {thr:.2f}: fewer than 2 events")
                    continue
                now, late = summarize(jump_trades(bars, e, hold=3)), summarize(jump_trades(bars, e, hold=3, delay=1))
                print(f"Theme {th} -> {a}, jump >= {thr:.2f}, 3h hold: n{now.trades:.0f} {now.mean_bp:+.1f}bp t{now.t_stat:.2f}"
                      f" | 1h late {late.mean_bp:+.1f}bp t{late.t_stat:.2f}")
                if thr == 0.05:
                    pooled.append(jump_trades(bars, e, hold=3).ret)
    r = pd.concat(pooled)
    print(f"Theme jumps pooled (>= 5 pts): n{len(r)} {r.mean() * 1e4:+.1f}bp t{tstat(r):.2f}")


def tdf_rebalancing():
    c = yf.download("SPY AGG".split(), start="2004-01-01", auto_adjust=True, progress=False)["Close"].dropna()
    for lb in (5, 10, 20):
        sp = c.SPY / c.SPY.shift(lb) - c.AGG / c.AGG.shift(lb)
        sig = sp < sp.expanding(500).quantile(0.1).shift(1)
        for h in (1, 5):
            fwd = c.SPY.shift(-h) / c.SPY - 1
            ex = (fwd - fwd.expanding(500).mean().shift(h + 1))[sig]
            keep, last = [], -h
            for d in ex.index:                      # one signal per h days
                if c.index.get_loc(d) - last >= h:
                    keep.append(d)
                    last = c.index.get_loc(d)
            print(f"TDF: stocks lag bonds (bottom decile {lb}d), SPY next {h}d excess: "
                  f"{fmt(ex.loc[keep] - 0.0002, [('2006', '2016'), ('2017', '2024-06'), ('2024-07', '2026')], 1e4, 'bp')}")


def january_cross_asset():
    groups = {"country ETFs": ("EWA EWC EWG EWH EWI EWJ EWK EWL EWM EWN EWO EWP EWQ EWS EWU EWW EWD EWT EWY EWZ", 0.002),
              "sector ETFs": ("XLK XLF XLE XLV XLI XLY XLP XLB XLU", 0.001),
              "crypto": ("BTC-USD ETH-USD XRP-USD LTC-USD BCH-USD ADA-USD DOGE-USD XLM-USD LINK-USD TRX-USD EOS-USD XMR-USD "
                         "DASH-USD ETC-USD NEO-USD ZEC-USD BNB-USD XTZ-USD ATOM-USD VET-USD", 0.004),
              "commodity futures": ("CL=F NG=F GC=F SI=F HG=F ZC=F ZW=F ZS=F KC=F SB=F CT=F LE=F HE=F", 0.001)}
    for name, (tk, cost) in groups.items():
        c = yf.download(tk.split(), start="1999-01-01", auto_adjust=True, progress=False)["Close"]
        c = c[c.index.dayofweek < 5]
        for a, b in ((5, 10), (1, 5), (10, 15)):
            out = {}
            for y in range(c.index[0].year + 1, 2027):
                jan, prev = c.index[(c.index.year == y) & (c.index.month == 1)], c[c.index.year == y - 1]
                if len(jan) <= b or len(prev) < 200:
                    continue
                perf = prev.iloc[-1] / prev.iloc[0] - 1
                ok = perf.notna() & c.loc[jan[a - 1]].notna() & c.loc[jan[b - 1]].notna()
                if ok.sum() < 6:
                    continue
                q = perf[ok].rank(pct=True)
                r = c.loc[jan[b - 1]] / c.loc[jan[a - 1]] - 1
                out[y] = r[q[q > 2 / 3].index].mean() - r[q[q <= 1 / 3].index].mean() - cost
            s = pd.Series(out)
            print(f"January {name}, days {a + 1}-{b} winners-losers net: {s.mean() * 100:.2f}% t{tstat(s):.2f} n{len(s)}"
                  f" positive {(s > 0).mean():.0%} | 2024-26 {s.loc[2024:].round(4).tolist()}")


def btc_month_end():
    b = yf.download("BTC-USD", start="2018-01-01", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    rows = []
    for m, g in b.groupby(b.index.to_period("M")):
        if len(g) >= 25:
            rows.append({"m": m, "down": g.iloc[-3] < g.iloc[0], "q": m.month % 3 == 0, "r": g.iloc[-1] / g.iloc[-3] - 1 - 0.002})
    d = pd.DataFrame(rows)
    for lab, x in (("pre-ETF 2018-2023", d[d.m < pd.Period("2024-01")]), ("post-ETF 2024+", d[d.m >= pd.Period("2024-01")])):
        print(f"BTC last 2 days of month, {lab}: after down month {x.r[x.down].mean() * 100:+.2f}% t{tstat(x.r[x.down]):.2f} n{x.down.sum()}"
              f" | after up month {x.r[~x.down].mean() * 100:+.2f}% t{tstat(x.r[~x.down]):.2f}"
              f" | quarter-end after down {x.r[x.down & x.q].mean() * 100:+.2f}% n{(x.down & x.q).sum()}")


def letf_sharpened():
    tick = sorted(set(LETF) | {s for v in LETF.values() for s in v} | {"SPY"})
    d = yf.download(tick, start="2022-01-01", auto_adjust=True, progress=False)
    c, o, v = d["Close"], d["Open"], d["Volume"]
    spy_oc = c.SPY.shift(-1) / o.SPY.shift(-1) - 1
    ev = []
    for u, legs in LETF.items():
        legs = {s: lv for s, lv in legs.items() if s in c and c[s].notna().sum() > 50}
        dv = sum(abs(lv * (lv - 1)) * (c[s] * v[s]).rolling(20, min_periods=5).mean().shift(1).fillna(0) for s, lv in legs.items())
        r = c[u].pct_change()
        ev.append(pd.DataFrame({"r": r, "sd": r.rolling(20).std().shift(1), "p": dv / (c[u] * v[u]).rolling(20).mean().shift(1),
                                "oc": c[u].shift(-1) / o[u].shift(-1) - 1 - spy_oc - 0.0005}).dropna())
    e = pd.concat(ev).sort_index()
    e = e[e.r < -2 * e.sd]
    fit, hold = e["2024-01-01":"2025-06-30"], e["2025-07-01":]
    th = fit.p.quantile(0.9)                       # frozen on the fit period
    for lab, x in (("fit 2024-01..2025-06", fit), ("holdout 2025-07+", hold)):
        a, b = x.oc[x.p >= th], x.oc[x.p < th]
        print(f"Sharpened LETF (pressure >= {th:.3f}, move < -2x 20d vol), {lab}: {a.mean() * 1e4:.0f}bp t{tstat(a):.2f} n{len(a)}"
              f" | rest {b.mean() * 1e4:.0f}bp t{tstat(b):.2f} n{len(b)}")


def main():
    S = pm_selection()
    F = pm_fills(S.id)
    pm_near_misses(S, F)
    pm_theme_jumps(S, F)
    tdf_rebalancing()
    january_cross_asset()
    btc_month_end()
    letf_sharpened()


if __name__ == "__main__":
    main()
