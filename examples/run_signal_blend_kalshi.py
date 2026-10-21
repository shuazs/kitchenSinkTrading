"""Do Kalshi prediction-market features help the signal combiner?

Same harness as run_signal_blend.py, run twice per model: once on the 20
strategy features and once with the 9 weekly Kalshi features added
(backtester.data.kalshi). Kalshi history starts late 2021, so the fair test
is the two walk-forward years before the locked holdout (2023 and the first
half of 2024), trained on everything earlier. The holdout period through
January 2026 (where the Kalshi file ends) is also shown with --extended, as
a second, descriptive look: the holdout was already scored once, and
nothing here is tuned on it.

    python examples/run_signal_blend_kalshi.py --kalshi /mnt/project-files/kalshi-leading/data/daily_features.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_signal_blend import HOLDOUT, MARKET, universes  # noqa: E402

from backtester import Backtester  # noqa: E402
from backtester.data.fred import load_series, tbill_rate  # noqa: E402
from backtester.data.kalshi import KALSHI_FEATURES, load_daily_features, weekly_features  # noqa: E402
from backtester.data.yahoo import load_market  # noqa: E402
from backtester.data.yahoo_batch import prefetch  # noqa: E402
from backtester.ml import FEATURES, build_panel, holdout_fold, walk_forward_folds  # noqa: E402
from backtester.ml.contest import information_coefficient, out_of_sample_positions, pooled_correlation, positions_frame  # noqa: E402
from backtester.ml.models import LADDER, LSTMMomentum, MLP  # noqa: E402
from backtester.strategies import SignalBlend  # noqa: E402
from backtester.strategies.multi_asset_trend import week_ends  # noqa: E402

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

KALSHI_END = "2026-01-30"  # last full week in the Kalshi file
WITH = FEATURES + KALSHI_FEATURES


def factory(name, features, quick):
    if name == "mlp":
        return lambda: MLP(seeds=3 if quick else 10, features=features)
    if name == "lstm":
        return lambda: LSTMMomentum(seeds=2 if quick else 5, features=features)
    return lambda: LADDER[name](features=features)


def run(name, cfg, daily, quick, extended):
    symbols = prefetch(cfg["symbols"] + MARKET)
    data = load_market(symbols, start=cfg["start"], curve=load_series("T10Y2Y"))
    tradable = [s for s in cfg["symbols"] if s in data.bars]
    kal = weekly_features(daily, week_ends(data.index))
    panel = build_panel(data, tradable, extra=kal)
    dates = panel.index.get_level_values("date")
    rf = tbill_rate()
    bt = Backtester(5_000, costs=cfg["costs"], cash_rate=rf, max_leverage=cfg["max_leverage"])

    def trade(pos, s, e):
        return bt.run(SignalBlend(positions=positions_frame(pos), **cfg["blend"]), data, start=s, end=e).stats(rf=rf)

    # Which Kalshi features line up with next week's vol-scaled return, before the holdout?
    pre = panel[(dates >= "2022-01-01") & (dates < HOLDOUT)]
    uni = pre[KALSHI_FEATURES].corrwith(pre["target"]).rename(name)

    periods = {"walk-forward 2023 to mid-2024": walk_forward_folds(dates, "2023-01-01", pd.Timestamp(HOLDOUT) - pd.Timedelta(days=1))}
    if extended:
        f = holdout_fold(dates, HOLDOUT)
        f.test = f.test[f.test <= pd.Timestamp(KALSHI_END)]
        periods[f"holdout {HOLDOUT} to {KALSHI_END} (second look)"] = [f]
    rows = []
    for label, folds in periods.items():
        s, e = folds[0].test[0], folds[-1].test[-1]
        test_rows = panel.index[(dates >= s) & (dates <= e)]
        base = trade(pd.Series(1.0, index=test_rows), s, e)
        rows.append({"period": label, "model": "always_long", "features": "-", "sharpe": base["sharpe"], "cagr": base["cagr"],
                     "pooled_corr": np.nan, "ic": np.nan})
        for m in ["equal_weight", "ridge", "lightgbm", "mlp", "lstm"]:
            for tag, feats in (("base", FEATURES), ("+kalshi", WITH)):
                if m == "equal_weight" and tag == "+kalshi":
                    continue  # equal weight only combines the per-asset strategy signals
                pos = out_of_sample_positions(panel, factory(m, feats, quick), folds)
                st = trade(pos, s, e)
                rows.append({"period": label, "model": m, "features": tag, "sharpe": st["sharpe"], "cagr": st["cagr"],
                             "pooled_corr": pooled_correlation(pos, panel), "ic": information_coefficient(pos, panel)})
            print(f"  {name} {label[:12]} {m} done", flush=True)
    return pd.DataFrame(rows), uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kalshi", required=True, help="path to the Kalshi daily_features.parquet")
    ap.add_argument("--universe", choices=["etf", "crypto", "sp100", "all"], default="all")
    ap.add_argument("--extended", action="store_true", help="also show the holdout period through Jan 2026 (descriptive)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    daily = load_daily_features(args.kalshi)
    cfgs = universes()
    names = list(cfgs) if args.universe == "all" else [args.universe]
    unis = []
    for n in names:
        table, uni = run(n, cfgs[n], daily, args.quick, args.extended)
        unis.append(uni)
        print(f"\n=== {n}")
        for period, t in table.groupby("period", sort=False):
            print(f"\n{period}")
            wide = t.pivot_table(index="model", columns="features", values=["sharpe", "pooled_corr"], sort=False)
            print(wide.to_string())
    print("\nCorrelation of each Kalshi feature with next week's vol-scaled return, Jan 2022 to Jun 2024 (pooled rows):")
    print(pd.concat(unis, axis=1).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
