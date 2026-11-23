"""Kalshi ladder as the consensus: do bonds/stocks drift after CPI and jobs surprises?

Surprise per release (`kalshi_macro.ladder_surprises`): settled value minus
Kalshi's implied median 24 h before the release, in strike steps, signed so
> 0 is hawkish (higher CPI / payrolls, lower unemployment). Same-day series
are averaged. Pre-registered (kalshi-macro/HYPOTHESES.md part 2):
H5 TLT open->close on release day against the surprise (short after hawkish),
H6 SPY the same, H7 TLT next-day close->close. |surprise| > 0.5 step to
trade; 4 bp per side. Holdout from Jul 2025.

    python examples/run_kalshi_surprise_drift.py
"""
import numpy as np
import pandas as pd

from backtester import kalshi_macro as km

D = "/mnt/project-files/kalshi-macro/"
HOLD = "2025-07-01"
COST = 2 * 0.0004


def tstat(x):
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def main():
    s = km.ladder_surprises(
        km.release_events(pd.read_parquet(D + "markets.parquet")), pd.read_parquet(D + "prints.parquet"))
    s["date"] = s["release"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    d = s.groupby("date")["surprise"].mean()
    px = pd.read_parquet(D + "etf_daily.parquet")
    o, c = px["Open"], px["Close"]
    rows = []
    for name, ret in [("H5 TLT open->close", c["TLT"] / o["TLT"] - 1), ("H6 SPY open->close", c["SPY"] / o["SPY"] - 1),
                      ("H7 TLT next day", c["TLT"].shift(-1) / c["TLT"] - 1)]:
        x = pd.DataFrame({"s": d}).join(ret.rename("r"), how="inner").dropna()
        x = x[x["s"].abs() > 0.5]
        x["pnl"] = -np.sign(x["s"]) * x["r"] - COST  # hawkish -> short
        for lab, g in [("in-sample", x[x.index < HOLD]), ("holdout", x[x.index >= HOLD])]:
            rows.append({"test": name, "period": lab, "n": len(g), "mean_bp": round(g["pnl"].mean() * 1e4, 1),
                         "t": round(tstat(g["pnl"]), 2), "hit": round((g["pnl"] > 0).mean(), 2),
                         "corr_surprise_ret": round(np.corrcoef(g["s"], g["r"])[0, 1], 2) if len(g) > 2 else np.nan})
    print(f"{d.size} release dates with a Kalshi surprise, {d.index.min():%Y-%m} to {d.index.max():%Y-%m}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
