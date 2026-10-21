"""Four machine-learning framings that use the repo's research as inputs,
each tested walk-forward with costs on $5,000. They differ from the weekly
signal-combination contest (run_signal_blend.py), which asked models to
forecast every asset's next-week return.

1. Meta-labeling: a classifier sees each Donchian breakout trade the primary
   rule proposes (32 ETFs, and 55 coins) with the repo's signals at entry,
   and skips the half it rates worst. Controls: the primary alone, and
   skipping a random half of trades.
2. Regimes: a statistical jump model on SPY returns (Shu, Yu and Mulvey 2024)
   switches SPY and the UPRO vol-target keeper to cash in its bear regime.
   Control: the 200-day trend filter.
3. Volatility forecasts: HAR, HAR+VIX, ridge and LightGBM forecasts of next
   month's SPY volatility size the UPRO vol-target keeper instead of the
   trailing 20-day volatility.
4. Monthly ranking: ridge and LambdaMART rank the S&P 100 by next month's
   return from the repo's price signals; top 10 held. Controls: the 12-1
   momentum sort and equal weight. Today's S&P 100 is survivor-biased.

    pip install -e ".[ml]"
    python examples/run_ml_research.py             # walk-forward to 2024-06-30
    python examples/run_ml_research.py --holdout   # also score Jul 2024 onward, once
    python examples/run_ml_research.py --only meta
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtester import COSTS, Backtester
from backtester.data.cache import cache_dir
from backtester.data.fred import tbill_rate
from backtester.data.yahoo import load_market
from backtester.data.yahoo_batch import prefetch
from backtester.ml.metalabel import GBM, Logistic, MetaFiltered, decide, event_table, walk_forward_probs
from backtester.ml.rank import TopN, monthly_panel, rank_ic, walk_forward_scores
from backtester.ml.regime import RegimeSwitch, walk_forward_regimes
from backtester.ml.vol_forecast import ForecastVolTarget, future_log_var, qlike, vol_features, walk_forward_vol
from backtester.strategies import Breakout, BuyAndHold, TrendFilter, VolTarget
from backtester.strategies.stock_momentum import SP100

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

KEY = ["cagr", "volatility", "sharpe", "max_drawdown", "n_trades", "annual_turnover"]
WF_END = "2024-06-30"
HOLDOUT = "2024-07-01"
ETFS = ("SPY QQQ IWM EFA EEM TLT IEF GLD SLV DBC USO UUP FXE FXY XLK XLF XLE XLV XLI XLY XLP XLU XLB VNQ "
        "EWJ EWZ EWG FXI HYG LQD TIP DBA").split()
COINS = ("BTC ETH XRP LTC BCH ADA DOGE LINK XLM TRX EOS XMR ETC DASH ZEC NEO XEM WAVES QTUM OMG BAT ZRX "
         "LSK ICX VET XTZ ATOM ALGO DOT SOL AVAX AAVE FIL NEAR BNB SHIB HBAR ICP EGLD THETA MKR SNX SAND "
         "MANA AXS KSM DCR SC DGB RVN ONT ZIL ENJ CHZ IOTA").split()
MARKET = ["SPY", "TLT", "^VIX", "^VIX3M", "^TNX", "^IRX"]


def periods(holdout: bool):
    out = [("walk-forward", None, WF_END)]
    return out + [("holdout", HOLDOUT, None)] if holdout else out


def curve_of(data):
    # 10y minus 3m Treasury yield from Yahoo, so the run does not depend on FRED.
    return data.bars["^TNX"]["close"] - data.bars["^IRX"]["close"]


def welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    return float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)))


def meta_labeling(rf, holdout: bool, results: dict) -> None:
    end = None if holdout else WF_END
    books = {
        "etf": dict(universe=ETFS, start="2004-01-01", first_test="2010-01-01", cost=0.0008,
                    costs=COSTS["etf"], slots=10, primary=Breakout(entry=20, exit=10)),
        "crypto": dict(universe=[f"{c}-USD" for c in COINS], start="2017-11-09", first_test="2019-01-01",
                       cost=0.004, costs=COSTS["crypto"], slots=10, primary=Breakout(entry=20, exit=10)),
    }
    for name, b in books.items():
        data = load_market(b["universe"] + MARKET, start=b["start"])
        trades = event_table(data, b["primary"], b["universe"], cost=b["cost"], curve=curve_of(data))
        print(f"\n== 1. Meta-labeling, {name}: {len(trades)} breakout trades, "
              f"{trades['label'].mean():.0%} win after costs")
        bt = Backtester(5_000, costs=b["costs"], cash_rate=rf, min_trade_value=50)
        for label, p_start, p_end in periods(holdout):
            lo, hi = p_start or b["first_test"], p_end or str(data.index[-1].date())
            rows, trade_rows = {}, {}
            for make in (Logistic, GBM):
                probs = walk_forward_probs(trades, make, b["first_test"], end or hi)
                probs = probs[(probs["entry"] >= lo) & (probs["entry"] <= hi)]
                lab = probs.dropna(subset=["label"])
                kept = lab[lab["p"] >= lab["p_q50"]]["ret"]
                skipped = lab[lab["p"] < lab["p_q50"]]["ret"]
                trade_rows[make.name] = {
                    "trades": len(lab), "AUC": roc_auc_score(lab["label"], lab["p"]),
                    "kept mean": kept.mean(), "skipped mean": skipped.mean(), "Welch t": welch_t(kept, skipped),
                    "AUC>0.5 years": f"{(lab.groupby(lab['entry'].dt.year).apply(lambda g: roc_auc_score(g['label'], g['p']) if g['label'].nunique() > 1 else np.nan) > 0.5).sum()}/{lab['entry'].dt.year.nunique()}",
                }
                for rule in ("all", "skip_q50", "size"):
                    if rule == "all" and make is GBM:
                        continue
                    t = probs.assign(mult=decide(probs, rule))
                    s = MetaFiltered(primary=b["primary"], universe=b["universe"], trades=t, slots=b["slots"])
                    key = "primary (all trades)" if rule == "all" else f"{make.name} {rule}"
                    rows[key] = bt.run(s, data, start=lo, end=hi).stats()[KEY]
            rng = np.random.default_rng(0)
            rand = []
            for _ in range(20):
                t = probs.assign(mult=(rng.random(len(probs)) >= 0.5).astype(float))
                s = MetaFiltered(primary=b["primary"], universe=b["universe"], trades=t, slots=b["slots"])
                rand.append(bt.run(s, data, start=lo, end=hi).stats()["sharpe"])
            print(f"-- {label} ({lo} to {hi}), per trade")
            print(pd.DataFrame(trade_rows).T.to_string())
            tab = pd.DataFrame(rows).T
            print(tab.to_string())
            print(f"random half skipped (20 draws): Sharpe {np.mean(rand):.3f} +- {np.std(rand):.3f}")
            results[f"meta_{name}_{label}"] = {"trades": pd.DataFrame(trade_rows).T.to_dict(), "book": tab.to_dict(),
                                               "random_sharpe": [float(np.mean(rand)), float(np.std(rand))]}


def regimes(rf, holdout: bool, results: dict) -> None:
    data = load_market(["SPY", "UPRO", "SHY"], start="1995-01-01")
    end = str(data.index[-1].date()) if holdout else WF_END
    states, picks = walk_forward_regimes(data.bars["SPY"]["close"], rf, "2007-01-01", end)
    print("\n== 2. Statistical jump model regimes on SPY")
    print("lambda picked per year:", picks["lam"].to_dict())
    print("share of days in bear regime:", states.groupby(states.index.year).mean().round(2).to_dict())
    bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf, min_trade_value=50)
    vt = VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20)
    runs = {
        "SPY hold": BuyAndHold(symbols=["SPY"]),
        "SPY, cash below 200d": TrendFilter(risk_on="SPY", signal="SPY", risk_off=None, ma=200),
        "SPY, cash in JM bear": RegimeSwitch(states=states, bull=BuyAndHold(symbols=["SPY"])),
        "UPRO vol target 20% (keeper)": vt,
        "UPRO vol target, cash below 200d": RegimeSwitch(
            states=(data.bars["SPY"]["close"] < data.bars["SPY"]["close"].rolling(200).mean()).astype(int), bull=vt),
        "UPRO vol target, cash in JM bear": RegimeSwitch(states=states, bull=vt),
    }
    for label, lo, hi in periods(holdout):
        lo = lo or "2009-07-01"
        tab = pd.DataFrame({k: bt.run(s, data, start=lo, end=hi).stats()[KEY] for k, s in runs.items()}).T
        print(f"-- {label} ({lo} to {hi or 'now'})")
        print(tab.to_string())
        results[f"regime_{label}"] = tab.to_dict()


def vol_forecasts(rf, holdout: bool, results: dict) -> None:
    data = load_market(["SPY", "UPRO", "SHY", "^VIX", "^VIX3M"], start="1995-01-01")
    spy = data.bars["SPY"]
    X = vol_features(spy, data.bars["^VIX"]["close"], data.bars["^VIX3M"]["close"])
    y = future_log_var(spy, 21)
    end = str(data.index[-1].date()) if holdout else WF_END
    base = X.drop(columns=["vix_ratio"])
    fc = {m: walk_forward_vol(base, y, m, "2009-01-01", end) for m in ("trailing", "har", "har_vix", "ridge", "lightgbm")}
    bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf, min_trade_value=50)
    print("\n== 3. Volatility forecasts sizing the UPRO vol-target keeper (20% target)")
    for label, lo, hi in periods(holdout):
        lo = lo or "2009-07-01"
        rows = {}
        for m, f in fc.items():
            fs = f.loc[lo:hi]
            rows[m] = {"QLIKE": qlike(fs, spy), **bt.run(
                ForecastVolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20, forecast=f),
                data, start=lo, end=hi).stats()[KEY].to_dict()}
        rows["keeper VolTarget(20d)"] = bt.run(VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20),
                                               data, start=lo, end=hi).stats()[KEY].to_dict()
        tab = pd.DataFrame(rows).T
        print(f"-- {label} ({lo} to {hi or 'now'})")
        print(tab.to_string())
        results[f"vol_{label}"] = tab.to_dict()


def ranking(rf, holdout: bool, results: dict) -> None:
    prefetch(SP100 + ["SPY"])
    data = load_market(SP100 + ["SPY"], start="2004-01-01")
    panel = monthly_panel(data, SP100)
    end = str(data.index[-1].date()) if holdout else WF_END
    # $10 minimum: the equal-weight benchmark holds ~100 names at ~$50 each.
    bt = Backtester(5_000, costs=COSTS["large_cap"], cash_rate=rf, min_trade_value=10)
    ew = panel["ret"].groupby(level="date").mean()
    scores = {m: walk_forward_scores(panel, m, "2010-01-01", end) for m in ("momentum", "ridge", "lambdarank", "gbm_reg")}
    scores["highest 20d vol (survivor check)"] = panel["vol_20"].loc["2010-01-01":]
    print("\n== 4. Monthly S&P 100 ranking, top 10 (today's members: survivor-biased)")
    for label, lo, hi in periods(holdout):
        lo = lo or "2010-01-01"
        rows = {}
        for m, s in scores.items():
            s = s[(s.index.get_level_values("date") >= lo) & (s.index.get_level_values("date") <= (hi or end))]
            ic = rank_ic(s, panel).dropna()
            top = s.groupby(level="date").rank(ascending=False) <= 10
            spread = panel["ret"].reindex(s.index)[top].groupby(level="date").mean() - ew
            spread = spread.dropna()
            rows[m] = {"IC": ic.mean(), "IC t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                       "top10 - EW /month": spread.mean(), **bt.run(TopN(scores=s, n=10), data, start=lo, end=hi).stats()[KEY].to_dict()}
        rows["equal weight all"] = bt.run(TopN(scores=pd.Series(1.0, index=panel.index), n=200), data, start=lo, end=hi).stats()[KEY].to_dict()
        rows["SPY"] = bt.run(BuyAndHold(symbols=["SPY"]), data, start=lo, end=hi).stats()[KEY].to_dict()
        tab = pd.DataFrame(rows).T
        print(f"-- {label} ({lo} to {hi or 'now'})")
        print(tab.to_string())
        results[f"rank_{label}"] = tab.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="also score July 2024 onward (do this once)")
    ap.add_argument("--only", choices=["meta", "regime", "vol", "rank"])
    a = ap.parse_args()
    rf = tbill_rate()
    results = {}
    steps = {"meta": meta_labeling, "regime": regimes, "vol": vol_forecasts, "rank": ranking}
    for k, f in steps.items():
        if a.only in (None, k):
            f(rf, a.holdout, results)
    out = cache_dir("ml_research") / f"results{'_holdout' if a.holdout else ''}.json"
    out.write_text(json.dumps(results, default=str, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
