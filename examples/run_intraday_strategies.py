"""Intraday rule families on free 1-minute bars, with walk-forward selection.

    python examples/run_intraday_strategies.py              # design period, through 2025
    python examples/run_intraday_strategies.py --holdout    # also score Jan-Oct 2026 once

Instruments: a SPY stand-in (S&P 500 index, Jun 2021 on), a QQQ stand-in
(NQ then QQQ, 2024 on), BTC (Binance spot, Jun 2021 on) in two sessions, the
UTC day and US stock hours, and ETH (Dec 2024 to Nov 2025) as a cross-check
for BTC's picks. See backtester/data/minute_mirrors.py.

Costs per unit traded (one side): ETFs 1.5 bp (penny spread on SPY/QQQ plus
a minute of slippage beyond the one-minute fill delay), stress 3 bp; micro
index futures (MES/MNQ) 0.6 bp. Spot crypto 10 bp (low-fee venue or maker)
and 25 bp (US retail taker), long only; CME micro bitcoin futures 4 bp.
Every rule runs at 1x exposure with fractional shares, so results are per
dollar of equity; one MES contract is ~$34k, about 7x a $5,000 account.

Trend families also run on high-volatility days only (trailing vol above
its 1-year median or 75th percentile). "break-even bp" is the one-way cost
per unit traded that would zero a variant's gross profit through 2025.

Walk-forward: each test year trades the variant with the best net Sharpe on
all earlier days. Results are saved to ~/.cache/kitchensink/intraday_lab/.
"""
import sys

import numpy as np
import pandas as pd

from backtester import intraday_lab as L
from backtester.data.cache import cache_dir
from backtester.data.minute_mirrors import load_minute, rth_bars

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 90)

HOLDOUT = pd.Timestamp("2026-01-01")
show_holdout = "--holdout" in sys.argv
out_dir = cache_dir("intraday_lab")


def utc_naive(sym: str) -> pd.DataFrame:
    d = load_minute(sym)
    d.index = d.index.tz_localize(None)
    return d


def ny_naive(sym: str) -> pd.DataFrame:
    d = load_minute(sym)
    d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    return d


INSTRUMENTS = {
    # name: (bars, session start, length, walk-forward test years, pick source for a cross-check)
    "SPY": (lambda: rth_bars("SPY"), 570, 390, [2023, 2024, 2025], None),
    "QQQ": (lambda: rth_bars("QQQ"), 570, 390, [2025], "SPY"),
    "BTC UTC day": (lambda: utc_naive("BTC"), 0, 1440, [2023, 2024, 2025], None),
    "BTC US hours": (lambda: ny_naive("BTC"), 570, 390, [2023, 2024, 2025], None),
    "ETH UTC day": (lambda: utc_naive("ETH"), 0, 1440, [], "BTC UTC day"),
}
# Cost tiers: (name, bps per unit traded one way, long only). Micro futures: MES/MNQ about $0.62 fees
# plus half a tick ($0.62) on ~$34k notional; CME micro bitcoin about $2.50 fees plus a tick on ~$11k.
TIERS = {
    False: [("gross", 0.0, False), ("ETF 1.5bp", 1.5, False), ("ETF 3bp", 3.0, False), ("micro futures 0.6bp", 0.6, False)],
    True: [("gross", 0.0, False), ("spot 10bp", 10.0, True), ("spot 25bp", 25.0, True), ("micro futures 4bp", 4.0, False)],
}


def family_params(name: str, params: dict, length: int) -> dict:
    p = dict(params)
    if length > 390 and name == "end-of-day momentum":
        p["signal_end"] = [30, 60, 120, 360, 720, 1380]
    if length > 390 and name == "VWAP trend":
        p["check_every"] = [5, 15, 60]
    return p


rows, picks_all, finals, n_variants = [], [], {}, 0
for inst, (load, start, length, test_years, cross) in INSTRUMENTS.items():
    g = L.Grid(load(), start=start, length=length)
    crypto = length == 1440 or inst.startswith(("BTC", "ETH"))
    design = g.days < HOLDOUT
    families = dict(L.FAMILIES, **(L.CRYPTO_FAMILIES if length == 1440 else {}))
    print(f"== {inst}: {g.days[0].date()} to {g.days[-1].date()}, {len(g)} days")
    for fam, (fn, params) in families.items():
        R = {t: {} for t in TIERS[crypto]}
        be = {}
        for k, x in L.family_positions(g, fam, fn, family_params(fam, params, length)):
            n_variants += 1
            for t in R:
                if not (t[2] and "long_only=False" in k):
                    sim = L.simulate(g, x, t[1])
                    R[t][k] = sim["ret"]
            s_d = sim[design]
            be[k] = s_d["gross"].mean() / s_d["turnover"].mean() * 1e4 if s_d["turnover"].mean() > 0 else np.nan
        be = pd.Series(be)
        for t, r in R.items():
            r = pd.DataFrame(r)
            r.to_parquet(out_dir / f"{inst}_{fam}_{t[0]}.parquet".replace(" ", "_"))
            rd = r[design]
            sh = L.sharpe(rd)
            row = {"instrument": inst, "family": fam, "costs": t[0], "variants": r.shape[1],
                   "share positive": (sh > 0).mean(), "best in-sample sharpe": sh.max(), "median sharpe": sh.median(),
                   "break-even bp median": be.median(), "break-even bp best": be.max()}
            wf, picks = L.walk_forward(rd, test_years)
            if len(wf):
                st = L.stats(wf)
                row.update({"WF years": f"{test_years[0]}-{test_years[-1]}", "WF cagr": st["cagr"],
                            "WF sharpe": st["sharpe"], "WF t": st["t"], "WF max_dd": st["max_dd"]})
                picks_all += [(inst, fam, t[0], y, b) for y, b in picks]
            final = sh.idxmax() if sh.notna().any() else None
            finals[(inst, fam, t[0])] = final
            if cross and finals.get((cross, fam, t[0])) in r:
                c = L.stats(rd[finals[(cross, fam, t[0])]])
                row.update({"cross pick cagr": c["cagr"], "cross pick sharpe": c["sharpe"]})
            row["final pick"] = final
            if show_holdout and final and (~design).any():
                h = L.stats(r.loc[~design, final])
                row.update({"holdout cagr": h["cagr"], "holdout sharpe": h["sharpe"], "holdout max_dd": h["max_dd"]})
            rows.append(row)
    if inst == "SPY":
        for name, c, _ in TIERS[False][1:]:
            ss = L.session_split(g, c)
            print(f"session split, {name} per fill, 2 fills a day, through 2025:")
            print(ss[design].apply(L.stats).T, "\n")
            if show_holdout:
                print("same, 2026 holdout:")
                print(ss[~design].apply(L.stats).T, "\n")

res = pd.DataFrame(rows)
res.to_csv(out_dir / ("summary_holdout.csv" if show_holdout else "summary.csv"), index=False)
pd.DataFrame(picks_all, columns=["instrument", "family", "costs", "year", "pick"]).to_csv(
    out_dir / "walk_forward_picks.csv", index=False)
print(res.drop(columns="final pick").to_string(index=False))
print(f"\n{n_variants} rule variants tried across {len(INSTRUMENTS)} instrument/sessions, each at "
      f"4 cost levels. Results in {out_dir}")
