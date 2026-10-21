"""Can a neural net combine our signals better than a linear blend?

Five models (equal weight, ridge, LightGBM, MLP, LSTM Deep Momentum Network)
plus an always-long baseline compete on the same weekly feature panel, the
same purged walk-forward folds, the same position sizing and the same costs,
on three universes: 14 multi-asset ETFs, ~57 large coins and today's S&P 100
(the last two are survivor-biased). The design doc's pass criteria are
printed for each universe.

    pip install -e ".[ml]"
    python examples/run_signal_blend.py                 # walk-forward only
    python examples/run_signal_blend.py --holdout       # also score the locked holdout, once
    python examples/run_signal_blend.py --universe etf --quick
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.data.cache import cache_dir
from backtester.data.fred import load_series, tbill_rate
from backtester.data.yahoo import load_market
from backtester.data.yahoo_batch import prefetch
from backtester.ml import build_panel, holdout_fold, walk_forward_folds
from backtester.ml.contest import (
    deflated_sharpe,
    information_coefficient,
    out_of_sample_positions,
    out_of_sample_seed_positions,
    pooled_correlation,
    positions_frame,
    yearly_sharpe,
)
from backtester.ml.models import LADDER, LSTMMomentum, MLP
from backtester.strategies import BuyAndHold, EnsembleTrend, HedgedMomentum, MultiAssetTrend, SignalBlend
from backtester.strategies.multi_asset_trend import TREND_UNIVERSE
from backtester.strategies.stock_momentum import SP100

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

COINS = ("BTC ETH XRP LTC BCH ADA DOGE LINK XLM TRX EOS XMR ETC DASH ZEC NEO XEM WAVES QTUM OMG BAT ZRX "
         "LSK ICX VET XTZ ATOM ALGO DOT SOL AVAX AAVE FIL NEAR BNB SHIB HBAR ICP EGLD THETA MKR SNX SAND "
         "MANA AXS KSM DCR SC DGB RVN ONT ZIL ENJ CHZ IOTA").split()
MARKET = ["SPY", "TLT", "^VIX", "^VIX3M"]
HOLDOUT = "2024-07-01"
# Configurations run while building this script, before the final run, so the
# Deflated Sharpe ratio charges for them too (a smoke test of all five
# models on the ETF universe).
EXPLORATORY_TRIALS = 5


def universes():
    etfs = [s for s in TREND_UNIVERSE if s != "BTC-USD"]
    return {
        "etf": dict(symbols=etfs, start="2004-01-01", first_test="2012-01-01", costs=COSTS["etf"],
                    blend=dict(mode="risk_parity", max_gross=1.5), max_leverage=1.5,
                    reference=("MultiAssetTrend, long or cash", MultiAssetTrend(universe=etfs, short="cash"))),
        "crypto": dict(symbols=[f"{c}-USD" for c in COINS], start="2017-11-09", first_test="2020-01-01",
                       costs=COSTS["crypto"], blend=dict(mode="risk_parity", max_gross=1.0), max_leverage=1.0,
                       reference=("EnsembleTrend", EnsembleTrend())),
        "sp100": dict(symbols=SP100, start="2004-01-01", first_test="2012-01-01", costs=COSTS["large_cap"],
                      blend=dict(mode="top_n", top_n=10), max_leverage=1.0,
                      reference=("HedgedMomentum top 10, unhedged", HedgedMomentum())),
    }


def model_factories(quick: bool):
    f = {k: v for k, v in LADDER.items()}
    if quick:
        f["mlp"] = lambda: MLP(seeds=3)
        f["lstm"] = lambda: LSTMMomentum(seeds=2)
    return f


def run_universe(name, cfg, quick, holdout, verbose):
    t0 = time.time()
    symbols = prefetch(cfg["symbols"] + MARKET)
    data = load_market(symbols, start=cfg["start"], curve=load_series("T10Y2Y"))
    tradable = [s for s in cfg["symbols"] if s in data.bars]
    panel = build_panel(data, tradable)
    dates = panel.index.get_level_values("date")
    folds = walk_forward_folds(dates, cfg["first_test"], pd.Timestamp(HOLDOUT) - pd.Timedelta(days=1))
    start, end = folds[0].test[0], folds[-1].test[-1]
    rf = tbill_rate()
    bt = Backtester(5_000, costs=cfg["costs"], cash_rate=rf, max_leverage=cfg["max_leverage"])
    print(f"\n=== {name}: {len(tradable)} symbols, {len(panel):,} asset-weeks, test {start.date()} to {end.date()}")

    def run(positions, s=start, e=end):
        return bt.run(SignalBlend(positions=positions_frame(positions), **cfg["blend"]), data, start=s, end=e)

    results, positions = {}, {}
    always = pd.Series(1.0, index=panel.index[(dates >= start) & (dates <= end)])
    positions["always_long"] = always
    seed_positions = {}
    for m, make in model_factories(quick).items():
        t = time.time()
        if m in ("mlp", "lstm"):
            # Each seed is scaled like the ensemble, so their mean is the ensemble.
            seed_positions[m] = out_of_sample_seed_positions(panel, make, folds)
            positions[m] = seed_positions[m].mean(axis=1)
        else:
            positions[m] = out_of_sample_positions(panel, make, folds, verbose=verbose)
        print(f"  {m}: {time.time() - t:.0f}s", flush=True)
    for m, pos in positions.items():
        results[m] = run(pos)
    ref_name, ref = cfg["reference"]
    results[ref_name] = bt.run(ref, data, start=start, end=end)
    results["buy and hold, equal weight"] = bt.run(BuyAndHold(symbols=tradable), data, start=start, end=end)

    stats = pd.DataFrame({k: r.stats(rf=rf) for k, r in results.items()}).T
    table = stats[["cagr", "volatility", "sharpe", "max_drawdown", "annual_turnover", "cost_drag_pct_per_year"]].astype(float)
    table["ic"] = pd.Series({m: information_coefficient(p, panel) for m, p in positions.items() if m != "always_long"})
    table["pooled_corr"] = pd.Series({m: pooled_correlation(p, panel) for m, p in positions.items() if m != "always_long"})
    yearly = pd.DataFrame({k: yearly_sharpe(r.equity) for k, r in results.items()})
    table["years_beating_ridge"] = (yearly.sub(yearly["ridge"], axis=0) > 0).mean()

    # Seed spread for the nets: each seed traded on its own.
    for m, seeds in seed_positions.items():
        sh = [run(seeds[c]).stats(rf=rf)["sharpe"] for c in seeds]
        table.loc[m, "worst_seed_sharpe"] = min(sh)
        table.loc[m, "best_seed_sharpe"] = max(sh)

    model_names = list(model_factories(quick))
    trial_sr = []
    for m in model_names:
        r = results[m].equity.pct_change().dropna()
        trial_sr.append(r.mean() / r.std())
    return dict(name=name, data=data, panel=panel, bt=bt, cfg=cfg, table=table, yearly=yearly, results=results,
                trial_sr=trial_sr, model_names=model_names, run=run, folds=folds, seconds=time.time() - t0)


def score_holdout(u, quick):
    panel, data, cfg = u["panel"], u["data"], u["cfg"]
    f = holdout_fold(panel.index.get_level_values("date"), HOLDOUT)
    rf = tbill_rate()
    out = {}
    for m, make in model_factories(quick).items():
        pos = make().fit(panel, f).predict(panel, f.test)
        out[m] = u["run"](pos, f.test[0], None).stats(rf=rf)
    out["always_long"] = u["run"](pd.Series(1.0, index=panel.index[panel.index.get_level_values("date") >= f.test[0]]),
                                  f.test[0], None).stats(rf=rf)
    ref_name, ref = cfg["reference"]
    out[ref_name] = u["bt"].run(ref, data, start=f.test[0]).stats(rf=rf)
    return pd.DataFrame(out).T[["cagr", "sharpe", "max_drawdown"]].astype(float)


def verdict(u, n_trials, all_trial_sr, holdout_table=None):
    t = u["table"]
    rows = []
    for net in ("mlp", "lstm"):
        r = u["results"][net].equity.pct_change().dropna()
        checks = {
            "beats ridge by 0.2": t.loc[net, "sharpe"] - t.loc["ridge", "sharpe"] >= 0.2,
            "beats equal weight by 0.2": t.loc[net, "sharpe"] - t.loc["equal_weight", "sharpe"] >= 0.2,
            "beats ridge in 2/3 of years": t.loc[net, "years_beating_ridge"] >= 2 / 3,
            "deflated Sharpe >= 0.95": deflated_sharpe(r, n_trials, all_trial_sr) >= 0.95,
            "worst seed >= ridge": t.loc[net, "worst_seed_sharpe"] >= t.loc["ridge", "sharpe"],
            "cost drag <= 2x ridge": t.loc[net, "cost_drag_pct_per_year"] <= 2 * t.loc["ridge", "cost_drag_pct_per_year"],
        }
        if holdout_table is not None:
            checks["holdout Sharpe >= ridge"] = holdout_table.loc[net, "sharpe"] >= holdout_table.loc["ridge", "sharpe"]
        rows.append(pd.Series(checks, name=net))
    dsr = {m: deflated_sharpe(u["results"][m].equity.pct_change().dropna(), n_trials, all_trial_sr) for m in u["model_names"]}
    return pd.DataFrame(rows).T, pd.Series(dsr, name="deflated_sharpe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["etf", "crypto", "sp100", "all"], default="all")
    ap.add_argument("--holdout", action="store_true", help="score the locked holdout (do this once, at the end)")
    ap.add_argument("--quick", action="store_true", help="fewer seeds, for a fast check")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    cfgs = universes()
    names = list(cfgs) if args.universe == "all" else [args.universe]
    runs = [run_universe(n, cfgs[n], args.quick, args.holdout, args.verbose) for n in names]
    n_trials = EXPLORATORY_TRIALS + sum(len(u["model_names"]) for u in runs)
    all_sr = [s for u in runs for s in u["trial_sr"]]
    log = {}
    for u in runs:
        print(f"\n--- {u['name']} walk-forward, net of costs ({u['seconds'] / 60:.1f} min)")
        print(u["table"].to_string())
        print("\nSharpe by year:")
        print(u["yearly"][["always_long", *u["model_names"]]].round(2).to_string())
        hold = score_holdout(u, args.quick) if args.holdout else None
        if hold is not None:
            print(f"\nHoldout from {HOLDOUT}:")
            print(hold.to_string())
        checks, dsr = verdict(u, n_trials, all_sr, hold)
        print(f"\nPass criteria for the nets ({n_trials} trials counted):")
        print(checks.to_string())
        print("\nDeflated Sharpe (probability the true Sharpe is above zero):")
        print(dsr.round(3).to_string())
        log[u["name"]] = {"table": u["table"].round(4).to_dict(), "yearly": u["yearly"].round(3).to_dict(),
                          "checks": checks.astype(bool).to_dict(), "dsr": dsr.round(4).to_dict(),
                          "holdout": None if hold is None else hold.round(4).to_dict()}
    path = cache_dir("ml") / f"signal_blend_{'holdout' if args.holdout else 'walkforward'}.json"
    path.write_text(json.dumps(log, default=str, indent=1))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
