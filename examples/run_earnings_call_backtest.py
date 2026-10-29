"""Backtest earnings-call text features (features.parquet from run_earnings_call_text.py).

Per call: event = first open after the call date (entry at open of earnings_date+1 trading day, exit at close
H days later); return is excess vs SPY. Signals are z-scored cross-sectionally per calendar quarter (no
look-ahead: only the current quarter's calls). Also a walk-forward ridge on the lexical features + change-vs-prior-call.
Usage: python examples/run_earnings_call_backtest.py FEATURES.parquet
"""
import sys
import numpy as np
import pandas as pd
from backtester.data.yahoo import load_bars

H = [5, 20, 60]
LAG = 1  # entry open = LAG trading days after earnings_date (call may be after the close)


def build(feat):
    f = pd.read_parquet(feat)
    f["date"] = pd.to_datetime(f.earnings_date)
    spy = load_bars("SPY", "2012-01-01", "2026-10-01")
    rows = []
    for tk, g in f.groupby("ticker"):
        try:
            b = load_bars(tk.replace(".", "-"), "2012-01-01", "2026-10-01")
        except Exception:
            continue
        if b is None or len(b) < 300:
            continue
        idx = b.index
        for _, r in g.iterrows():
            i = idx.searchsorted(r.date, side="right") - 1 + LAG  # trading day LAG after earnings_date
            if i < 0 or i + max(H) >= len(idx) or abs((idx[i] - r.date).days) > 7:
                continue
            e = {"entry": idx[i], "ticker": tk}
            o = b["open"].iloc[i]
            for h in H:
                x = b["close"].iloc[i + h - 1] / o - 1
                j = spy.index.searchsorted(idx[i]); k = spy.index.searchsorted(idx[i + h - 1])
                m = spy["close"].iloc[k] / spy["open"].iloc[j] - 1
                e[f"r{h}"] = x - m
            e["gap"] = o / b["close"].iloc[i - 1] - 1  # entry-day gap vs prior close (info, not a signal)
            rows.append({**r.to_dict(), **e})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build(sys.argv[1])
    d.to_parquet("/tmp/ec/events.parquet")
    print(len(d))
