"""Short both legs of 3x leveraged ETF pairs to harvest volatility decay (pre-registered).

A 3x bull and a 3x bear ETF on the same index both lose value in choppy
markets, so shorting equal dollars of both earns that decay, minus borrow and
the big losses when one leg trends hard between rebalances.
Rule (kalshi-macro/HYPOTHESES.md part 4): gross 1x equity, 50% short each
leg, rebalance monthly (H10) or weekly (H11), borrow 5%/yr per leg (10%
sensitivity), no short rebate, 3 bp per side on traded dollars. Holdout from
Jul 2024. Data: yfinance adjusted closes.

    python examples/run_letf_pair_decay.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

PAIRS = [("TQQQ", "SQQQ"), ("UPRO", "SPXU"), ("SOXL", "SOXS"), ("TNA", "TZA"), ("FAS", "FAZ"), ("LABU", "LABD")]
HOLD = "2024-07-01"
COST = 0.0003


def pair_returns(px: pd.DataFrame, bull: str, bear: str, freq: str, borrow: float) -> pd.Series:
    """Daily returns of a 50/50 short of both legs, rebalanced at `freq` period ends."""
    p = px[[bull, bear]].dropna()
    r = p.pct_change().fillna(0.0)
    w = np.array([-0.5, -0.5])
    out = []
    period = p.index.to_period(freq)
    for i in range(len(p)):
        day = (w * r.iloc[i].values).sum() - borrow / 252 * np.abs(w).sum()
        w = w * (1 + r.iloc[i].values)
        eq = 1 + day
        w = w / eq
        turnover = 0.0
        if i + 1 < len(p) and period[i + 1] != period[i]:
            turnover = np.abs(np.array([-0.5, -0.5]) - w).sum()
            w = np.array([-0.5, -0.5])
        out.append(day - turnover * COST)
    return pd.Series(out, index=p.index).iloc[1:]


def stats(r: pd.Series) -> dict:
    if len(r) < 50:
        return {}
    eq = (1 + r).cumprod()
    return {"cagr": round(eq.iloc[-1] ** (252 / len(r)) - 1, 3), "vol": round(r.std() * np.sqrt(252), 3),
            "sharpe": round(r.mean() / r.std() * np.sqrt(252), 2), "maxdd": round((eq / eq.cummax() - 1).min(), 3)}


def main():
    tick = sorted({t for pr in PAIRS for t in pr})
    px = yf.download(tick, start="2010-01-01", auto_adjust=True, progress=False)["Close"]
    for freq, lab in [("M", "H10 monthly"), ("W", "H11 weekly")]:
        for borrow in (0.05, 0.10):
            rets = {f"{a}/{b}": pair_returns(px, a, b, freq, borrow) for a, b in PAIRS}
            ew = pd.DataFrame(rets).mean(axis=1, skipna=True).dropna()
            print(f"\n{lab}, borrow {borrow:.0%}/yr per leg")
            rows = {k: {"IS": stats(v[v.index < HOLD]), "HO": stats(v[v.index >= HOLD])} for k, v in rets.items()}
            rows["equal-weight"] = {"IS": stats(ew[ew.index < HOLD]), "HO": stats(ew[ew.index >= HOLD])}
            for k, v in rows.items():
                print(f"  {k:<13} IS {v['IS']}  HO {v['HO']}")
            if freq == "M" and borrow == 0.05:
                y = ew.groupby(ew.index.year).apply(lambda x: (1 + x).prod() - 1).round(3)
                print("  equal-weight by year:", y.to_dict())


if __name__ == "__main__":
    main()
