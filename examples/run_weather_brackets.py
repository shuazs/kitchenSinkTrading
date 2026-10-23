"""Kalshi temperature brackets vs NWP consensus (alpha hunt #3).
Kill rule: edge under 3c/contract after fees, or not positive in every month."""
import numpy as np
import pandas as pd

from backtester.weather_brackets import brier, edge_trades, load_rows


def main():
    rows = load_rows()
    print(f"{len(rows)} bracket snapshots, {rows.day.min()} to {rows.day.max()}")
    print("Brier (lower is better):", {k: round(v, 4) for k, v in brier(rows).items()})
    for prob in ("p_model", "p_model_raw"):
        for edge in (0.05, 0.10, 0.15):
            t = edge_trades(rows, prob, edge)
            daily = t.groupby("day").pnl.sum()
            month = pd.to_datetime(t.day).dt.month
            print(f"{prob:11s} edge {edge:.2f}: n {len(t):4d}, {t.pnl.mean()*100:+.2f}c/contract, "
                  f"t(day) {daily.mean()/daily.std()*np.sqrt(len(daily)):.2f}, by month "
                  f"{(t.pnl.groupby(month.values).mean()*100).round(1).to_dict()}")


if __name__ == "__main__":
    main()
