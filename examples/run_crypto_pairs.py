"""Crypto pairs trading / stat-arb: BTC/ETH, a 19-coin universe, daily and intraday.

Five tests, all with per-fill costs from `crypto_pairs.COST_BPS` (spot taker
55 bp, spot maker 30 bp, futures 7 bp), design 2018-06..2023 and holdout
2024+ (intraday: design 2021-06..2024-06, holdout 2024-07..2026-02):

1. BTC/ETH daily z-score mean reversion (rolling hedge or log ratio), as a
   long/short pair and as a long-only rotation (hold the cheap coin).
2. The same spread traded the other way (ETH/BTC ratio momentum).
3. Rolling pair selection over 19 coins (Engle-Granger cointegration or
   Gatev distance), long/short and long-only cheap leg.
4. Walk-forward: each year, pick the test-3 config with the best Sharpe over
   the previous 2 years and trade it for the next year.
5. Intraday 5m / 15m / 1h z-score pairs on six Binance coins: gross bp per
   round trip against the round-trip cost (2 x per-fill cost).

Data: `daily_close.parquet` (yfinance, 2017-11+) and `bars_5m.parquet`
(Binance spot 1m from HF Torch-Trade/*usdt_spot_1m_05_2021_to_03_2026,
resampled) in /mnt/project-files/crypto-pairs/ (`--data` overrides).
Run `--fetch` once to rebuild the daily file from Yahoo.
Takes ~15 min; `--skip-intraday` for the daily part only (~6 min).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.crypto_pairs import (
    COST_BPS,
    band_positions,
    rolling_pairs,
    rolling_spread_z,
    rotation_returns,
    simulate_pair,
    stats,
    trade_list,
)

COINS = "BTC ETH SOL BNB XRP ADA DOGE LTC BCH LINK DOT AVAX XLM ETC TRX ATOM ALGO FIL XMR".split()
DESIGN, HOLD = slice("2018-06-01", "2023-12-31"), slice("2024-01-01", None)
I_DESIGN, I_HOLD = ("2021-06-01", "2024-06-30"), ("2024-07-01", "2026-03-01")


def fetch_daily(path: Path) -> None:
    from backtester.data.yahoo import load_market

    data = load_market([c + "-USD" for c in COINS], start="2017-11-09")
    close = data.field("close")
    close.columns = [c.replace("-USD", "") for c in close.columns]
    close.to_parquet(path)


def per_period(net: pd.Series, ppy: float = 365) -> dict:
    d, h = stats(net[DESIGN], ppy), stats(net[HOLD], ppy)
    return {"sharpe_design": d["sharpe"], "cagr_design": d["cagr"], "dd_design": d["max_dd"],
            "sharpe_hold": h["sharpe"], "cagr_hold": h["cagr"], "dd_hold": h["max_dd"]}


def btc_eth(close: pd.DataFrame) -> pd.DataFrame:
    r, lp = close.pct_change(), np.log(close)
    rows = []
    grid = itertools.product([60, 120, 250], [20, 30, 60, 90], [1.5, 2.0, 2.5], [0.0, 0.5], [True, False])
    for hw, zw, ent, ex, ratio in grid:
        if ratio and hw != 60:
            continue
        h = rolling_spread_z(lp.ETH, lp.BTC, hw, zw, ratio=ratio)
        for direction, sign in [("reversion", 1), ("momentum", -1)]:
            if direction == "momentum" and not ratio:
                continue
            pos = sign * band_positions(h.z.values, ent, ex if sign == 1 else 0.0, 4.0 if sign == 1 else None)
            for cost in ["futures", "spot_taker"]:
                ls = simulate_pair(r.ETH, r.BTC, pos, h.beta.values, COST_BPS[cost])
                rot = rotation_returns(r.ETH, r.BTC, pos, COST_BPS[cost])
                for kind, res in [("long_short", ls), ("rotation", rot)]:
                    rows.append({"direction": direction, "kind": kind, "cost": cost, "hedge_window": hw,
                                 "z_window": zw, "entry": ent, "exit": ex, "ratio": ratio,
                                 **per_period(res["net"])})
    return pd.DataFrame(rows)


def universe(close: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, series, cache = [], {}, {}
    grid = itertools.product(["coint", "distance"], [90, 180, 365], [14, 30, 60], [3, 5, 10], [1.5, 2.0, 2.5])
    for method, form, trd, n, ent in grid:
        for cost, long_only in [("futures", False), ("spot_taker", False), ("spot_taker", True)]:
            out = rolling_pairs(close, form, trd, n, ent, 0.0 if method == "distance" else 0.5, 4.0, method,
                                COST_BPS[cost], long_only=long_only, cache=cache)
            kind = "long_only" if long_only else "long_short"
            t = out["trades"]
            key = (method, form, trd, n, ent, cost, kind)
            series[key] = out["returns"]
            rows.append({"method": method, "formation": form, "trading": trd, "n_pairs": n, "entry": ent,
                         "cost": cost, "kind": kind, "trades": len(t),
                         "gross_per_trade": t["gross"].mean() if len(t) else np.nan,
                         "win_rate": (t["gross"] > 0).mean() if len(t) else np.nan,
                         **per_period(out["returns"])})
    return pd.DataFrame(rows), series


def walk_forward(series: dict, cost: str, kind: str, train_years: int = 2) -> pd.Series:
    """Each calendar year, trade the config with the best Sharpe over the
    previous `train_years` years. Only unseen years are reported."""
    keys = [k for k in series if k[5] == cost and k[6] == kind]
    panel = pd.DataFrame({k: series[k] for k in keys}).fillna(0.0)
    years = sorted(set(panel.index.year))
    out = []
    for y in years:
        train = panel[(panel.index.year >= y - train_years) & (panel.index.year < y)]
        if train.index.year.nunique() < train_years:
            continue
        sh = train.mean() / train.std().replace(0, np.nan)
        out.append(panel.loc[panel.index.year == y, sh.idxmax()])
    return pd.concat(out)


def intraday(bars: pd.DataFrame) -> pd.DataFrame:
    c5 = bars.pivot(index="ts", columns="symbol", values="close").ffill()
    pairs = [("ETH", "BTC"), ("SOL", "ETH"), ("SOL", "BTC"), ("BNB", "BTC"), ("XRP", "BTC"),
             ("ADA", "BTC"), ("ADA", "XRP"), ("ADA", "ETH")]
    rows = []
    for freq, k in [("5min", 1), ("15min", 3), ("1h", 12)]:
        c = c5 if k == 1 else c5.resample(freq, label="right", closed="right").last().dropna(how="all")
        r, lp = c.pct_change(), np.log(c)
        for (y, x), hw, zw, ent in itertools.product(pairs, [288, 2016], [48, 288, 1152], [2.0, 3.0]):
            # windows in 5-min bars (1 day, 1 week; 4 h, 1 day, 4 days)
            h = rolling_spread_z(lp[y], lp[x], max(hw // k, 20), max(zw // k, 12))
            pos = band_positions(h.z.values, ent, 0.0, ent + 3)
            t = trade_list(simulate_pair(r[y], r[x], pos, h.beta.values, 0.0))
            start = pd.to_datetime(t["start"])
            for period, (a, b) in [("design", I_DESIGN), ("hold", I_HOLD)]:
                tt = t[(start >= pd.Timestamp(a, tz="UTC")) & (start <= pd.Timestamp(b, tz="UTC"))]
                rows.append({"freq": freq, "pair": f"{y}/{x}", "hedge_window": hw, "z_window": zw, "entry": ent,
                             "period": period, "trades": len(tt), "gross_bp": tt["gross"].mean() * 1e4,
                             "win_rate": (tt["gross"] > 0).mean(), "bars_held": tt["bars"].mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/project-files/crypto-pairs")
    ap.add_argument("--out", default=None, help="results folder (default <data>/results)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--skip-intraday", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    out = Path(a.out) if a.out else data / "results"
    out.mkdir(parents=True, exist_ok=True)
    if a.fetch:
        fetch_daily(data / "daily_close.parquet")
    close = pd.read_parquet(data / "daily_close.parquet")
    close = close[close.index < close.index[-1]]  # drop today's partial bar
    pd.set_option("display.width", 200)

    r = close.pct_change()
    bench = {"BTC": r.BTC, "50/50 BTC+ETH": 0.5 * r.BTC + 0.5 * r.ETH,
             "equal-weight 19 coins": r.mean(axis=1)}
    bench_tab = pd.DataFrame({k: per_period(v.fillna(0)) for k, v in bench.items()}).T
    bench_tab.to_csv(out / "benchmarks.csv")
    print("Buy and hold\n", bench_tab.round(2))

    be = btc_eth(close)
    be.to_csv(out / "btc_eth_daily.csv", index=False)
    print("\n1-2. BTC/ETH daily, median over the grid")
    print(be.groupby(["direction", "kind", "cost"])[["sharpe_design", "sharpe_hold", "cagr_hold"]].median().round(2))

    uni, series = universe(close)
    uni.to_csv(out / "universe_pairs.csv", index=False)
    print("\n3. 19-coin rolling pairs, median over 162 configs")
    print(uni.groupby(["kind", "cost", "method"])[["sharpe_design", "sharpe_hold", "gross_per_trade", "win_rate"]]
          .median().round(3))
    best = uni[(uni.kind == "long_short") & (uni.cost == "futures")].nlargest(5, "sharpe_design")
    print("Top 5 by design Sharpe (futures costs)\n", best.round(3).to_string(index=False))

    wf = {}
    for cost, kind in [("futures", "long_short"), ("spot_taker", "long_short"), ("spot_taker", "long_only")]:
        s = walk_forward(series, cost, kind)
        wf[f"{kind} {cost}"] = {**stats(s, 365), "start": s.index[0].date()}
    wf_tab = pd.DataFrame(wf).T
    wf_tab.to_csv(out / "walk_forward.csv")
    print("\n4. Walk-forward (2y train, 1y test)\n", wf_tab)

    if not a.skip_intraday:
        intr = intraday(pd.read_parquet(data / "bars_5m.parquet"))
        intr.to_csv(out / "intraday.csv", index=False)
        print("\n5. Intraday gross bp per round trip (median over pairs and windows)")
        print(intr.groupby(["freq", "period"])[["gross_bp", "trades", "win_rate", "bars_held"]].median().round(2))
        rt = {k: 2 * v for k, v in COST_BPS.items()}
        piv = intr.pivot_table(index=["freq", "pair", "hedge_window", "z_window", "entry"], columns="period",
                               values="gross_bp")
        for name, c in rt.items():
            print(f"round trip {name} {c:.0f} bp: configs clearing it in design {(piv.design > c).mean():.1%}, "
                  f"in design and holdout {((piv.design > c) & (piv.hold > c)).mean():.1%}")


if __name__ == "__main__":
    main()
