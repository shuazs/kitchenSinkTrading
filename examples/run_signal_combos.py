"""Turn the repo's strategies into signals and test every combination of them.

    python examples/run_signal_combos.py [out_dir]

Three searches, each over every subset (not a sample):
  A. SPY timing: 15 on/off signals taken from the strategies (trend filter,
     VIX term structure, vol target, turn of month, pension month-end, RSI(2),
     dual momentum, multi-asset trend, breakout, seasonality, mood index,
     credit, sector breadth, macro switch) x rules AND / OR / majority /
     share-on x vehicles SPY, SSO, UPRO, each flat or 20% vol-targeted.
  B. BTC+ETH timing: 9 signals (breakout, ensemble trend, 200d, vol, x-sectional
     momentum, plus SPY trend / VIX / mood as cross-asset filters) x the same
     rules, flat or 50% vol-targeted, 0.4% retail crypto fees.
  C. Strategy sleeves: every subset of the strategies' own $5k backtests
     (net of costs), equal- or inverse-vol-weighted; once over 2019+ with
     crypto sleeves, once over 2012+ without.

Selection is walk-forward (half-yearly, expanding window); Jul 2024+ is a
holdout that no choice sees. Deflated Sharpe and PBO say how much of the best
backtest is luck. Results: docs/README.md ("Signal combination search").
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import COSTS, Backtester
from backtester.combo_search import SleeveSearch, TimingSearch, banded_scale, evaluate, open_to_open, stats, unit_cost
from backtester.core import MarketData
from backtester.data.fred import tbill_rate, unemployment_rate
from backtester.data.sentiment_indices import MOOD_SYMBOLS, mood_components, mood_index
from backtester.data.yahoo import load_market
from backtester.strategies import (
    Breakout, BuyAndHold, CrossSectionalMomentum, DualMomentum, EnsembleTrend, FundamentalAllocator, MultiAssetTrend,
    PensionRebalance, RiskParity, RSIMeanReversion, SameMonthSeasonality, SectorRotation, Sleeves, TrendFilter,
    TurnOfMonth, VixTermStructure, VolTarget, VolTargetBreakout,
)
from backtester.strategies.fundamental_allocator import macro_stress
from backtester.strategies.multi_asset_trend import INVERSE, TREND_UNIVERSE
from backtester.strategies.sector_rotation import SECTOR_SPDRS

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)
HOLDOUT = "2024-07-01"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if out:
    out.mkdir(parents=True, exist_ok=True)
rf = tbill_rate()


def sub(d: MarketData, syms) -> MarketData:
    return MarketData({s: d.bars[s] for s in syms if s in d.bars}, d.extra)


def on(strategy, d: MarketData, syms, col) -> pd.Series:
    """True where `strategy`, run on `syms`, holds a positive weight in `col`."""
    w = strategy.generate_targets(sub(d, syms))[col]
    return w.reindex(d.index).ffill().fillna(0.0) > 0


def daily_cash(index) -> pd.Series:
    annual = rf.reindex(index, method="ffill").fillna(0.0)
    days = index.to_series().diff().dt.days.fillna(1).clip(lower=1)
    return (1 + annual) ** (days / 365.0) - 1


def report(label: str, search, res: dict, bench: dict, ppy: float = 252.0) -> dict:
    """Print and return the headline numbers for one search."""
    cash = search.cash
    hb = res["blocks"] >= res["holdout_block"]
    wf = res["wf_returns"]
    best = res["best_returns"]
    print(f"\n==== {label}: {len(search):,} combinations, {search.index[0].date()} to {search.index[-1].date()}")
    print("\nTop 10 on all pre-holdout data (in-sample), with their holdout Sharpe:")
    print(res["board"].head(10).to_string(index=False))
    print("\nWalk-forward picks (best on all earlier data, traded the next half-year):")
    print(res["wf_picks"].to_string(index=False))
    rows = {"walk-forward pick": stats(wf, cash, ppy), "walk-forward top-20 blend": stats(res["wf_top_returns"], cash, ppy)}
    for k, r in bench.items():
        rows[f"{k} (same days)"] = stats(r.reindex(wf.index), cash, ppy)
    rows["in-sample best, holdout"] = stats(best[hb], cash, ppy)
    for k, r in bench.items():
        rows[f"{k}, holdout"] = stats(r[hb], cash, ppy)
    table = pd.DataFrame(rows).T
    print("\nOut of sample:")
    print(table)
    print("\nDeflated Sharpe of the in-sample best:", {k: round(v, 3) for k, v in res["dsr"].items()})
    print("PBO (CSCV over pre-holdout half-years):", {k: round(v, 3) for k, v in res["pbo"].items()})
    pre = res["board"]["sharpe_pre"]
    print(f"Share of trials beating {next(iter(bench))} in-sample: "
          f"{(pre > stats(next(iter(bench.values()))[~hb], cash, ppy)['sharpe']).mean():.1%}")
    if out:
        slug = "_".join(re.findall(r"[a-z0-9]+", label.lower()))
        res["board"].head(2000).to_csv(out / f"{slug}_top.csv", index=False)
        res["wf_picks"].to_csv(out / f"{slug}_walk_forward.csv", index=False)
        table.to_csv(out / f"{slug}_summary.csv")
    return {"table": table, "dsr": res["dsr"], "pbo": res["pbo"], "n": len(search)}


summaries = {}

# ------------------------------------------------------------------ A. SPY timing
t0 = time.time()
d = load_market(["SPY", "SSO", "UPRO", "TLT", "SHY", "^VIX", "^VIX3M", "HYG", "IEF", *SECTOR_SPDRS], start="2003-01-01")
try:
    d.extra["macro"] = pd.DataFrame({"UNRATE": unemployment_rate()})
except Exception as e:  # FRED and its mirrors down
    print("no unemployment data:", e)
idx = d.index
close = d.field("close").ffill()
pension = PensionRebalance()
pension.generate_targets(sub(d, ["SPY", "TLT", "SSO"]))
spy_vol = close["SPY"].pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
sector_above = (close[SECTOR_SPDRS] > close[SECTOR_SPDRS].rolling(200).mean()).where(close[SECTOR_SPDRS].rolling(200).count() == 200)
sig = pd.DataFrame({
    "trend_200d": on(TrendFilter(risk_on="SPY", signal="SPY"), d, ["SPY"], "SPY"),
    "vix_contango": on(VixTermStructure(symbol="SPY"), d, ["SPY", "^VIX", "^VIX3M"], "SPY"),
    "low_vol": spy_vol < 0.15,  # VolTarget at 15% would be fully invested
    "turn_of_month": on(TurnOfMonth(), d, ["SPY"], "SPY"),
    "pension_bonds_led": pension.state.reindex(idx) == -1,
    "pension_not_stocks_led": pension.state.reindex(idx) != 1,
    "rsi2_dip": on(RSIMeanReversion(), d, ["SPY"], "SPY"),
    "abs_momentum": on(DualMomentum(universe=["SPY"], safe="SHY"), d, ["SPY", "SHY"], "SPY"),
    "tsmom": MultiAssetTrend().signal(d, ["SPY"])["SPY"].reindex(idx).ffill() > 0,
    "breakout_20_10": on(Breakout(entry=20, exit=10), d, ["SPY"], "SPY"),
    "seasonal_month": on(SameMonthSeasonality(universe=["SPY"], top_n=1, demean=False), d, ["SPY"], "SPY"),
    "mood_fear": mood_index(close[MOOD_SYMBOLS]).reindex(idx) < 25,
    "credit_improving": mood_components(close[MOOD_SYMBOLS])["junk"] > 0,
    "sector_breadth": sector_above.mean(axis=1) > 0.5,
    "macro_ok": ~macro_stress(d, idx, signals=("unemployment",)) if "macro" in d.extra else pd.Series(True, index=idx),
}).fillna(False)
ev = sig.loc["2010-01-01":].iloc[:-2]  # last two rows have no next-open return yet
print("SPY signals, share of days on (2010+):")
print(ev.mean().round(2).to_string())
etf_cost = unit_cost(COSTS["etf"])
vehicles = {}
for sym, lev in (("SPY", 1), ("SSO", 2), ("UPRO", 3)):
    r = open_to_open(d.bars[sym], idx)
    vehicles[sym] = (r, etf_cost, None)
    vehicles[f"{sym} vt20"] = (r, etf_cost, banded_scale((0.20 / (lev * spy_vol)).clip(upper=1.0)))
cash = daily_cash(ev.index)
search = TimingSearch(ev, vehicles, cash)
res = evaluate(search, HOLDOUT)
always = pd.DataFrame({"always": True}, index=ev.index)
base = TimingSearch(always, {k: vehicles[k] for k in ("SPY", "UPRO vt20")}, cash)
bench = {"SPY hold": base.returns(0), "UPRO vt20 (keeper)": base.returns(1)}
summaries["A. SPY timing"] = report("A. SPY timing", search, res, bench)
print(f"[{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ B. BTC+ETH timing
t0 = time.time()
COINS = ["BTC-USD", "ETH-USD"]
dc = load_market(COINS, start="2017-11-09")
cidx = dc.index
cclose = dc.field("close").ffill()
ens = EnsembleTrend(top_n=2, min_history=60).signal(cclose)
xs = CrossSectionalMomentum(top_n=1, lookback=21, trend_symbol=None).generate_targets(dc).reindex(cidx).ffill().fillna(0)
cvol = cclose.pct_change(fill_method=None).rolling(30).std() * np.sqrt(365)
eq_cal = lambda s: s.reindex(cidx, method="ffill")  # stock-market signals carried over weekends
cross = {
    "spy_trend_200d": eq_cal(sig["trend_200d"]),
    "vix_contango": eq_cal(sig["vix_contango"]),
    "mood_not_fear": eq_cal(mood_index(close[MOOD_SYMBOLS]) >= 25),
}
crypto_cost = unit_cost(COSTS["crypto"])
start_c = "2019-01-01"
coin_res, coin_search = {}, {}
book_sig = []
for coin in COINS:
    s = pd.DataFrame({
        "breakout_20_10": on(Breakout(entry=20, exit=10), dc, [coin], coin),
        "ensemble_trend": ens[coin] >= 0.5,
        "ma_200d": on(TrendFilter(risk_on=coin, signal=coin, ma=200), dc, [coin], coin),
        "low_vol": cvol[coin] < 0.6,
        "xs_momentum": xs[coin] > 0,
        "tsmom": MultiAssetTrend().signal(dc, [coin])[coin].reindex(cidx).ffill() > 0,
        **cross,
    }).fillna(False).loc[start_c:].iloc[:-2]
    book_sig.append(s)
# One book: each coin gets half the account and its own copy of every rule.
ret = {c: open_to_open(dc.bars[c], cidx) for c in COINS}
cash_c = daily_cash(book_sig[0].index)


class BookSearch(TimingSearch):
    """TimingSearch over two coins at once: same rule on each coin's own signals, 50/50."""

    def __init__(self, sigs, vehicles, cash):
        super().__init__(sigs[0], vehicles, cash)
        self.sig2 = sigs[1].fillna(False).astype(bool)
        self.specs = self.specs[self.specs.vehicle.str.startswith(COINS[0])].copy()
        self.specs["vehicle"] = self.specs.vehicle.str.replace(COINS[0], "BTC+ETH", regex=False)
        self.specs = self.specs.reset_index(drop=True)

    def _net(self, vehicle, mask_idx, rules):
        from backtester.combo_search import combine, timing_returns
        tag = vehicle.replace("BTC+ETH", "")
        total = 0
        for c, s in zip(COINS, (self.signals, self.sig2)):
            r, cost, scale = self.vehicles[c + tag]
            w = combine(s.to_numpy(), self.masks[mask_idx], rules) * 0.5
            if scale is not None:
                w = w * scale[:, None].astype(np.float32)
            # timing_returns parks (1 - w) in cash; this coin's half only owns 0.5 - w of it.
            total = total + timing_returns(w, r, self.cash.to_numpy(), cost) - 0.5 * self.cash.to_numpy()[:, None]
        return total

    def moments(self, blocks, names, chunk=6000, periods_per_year=365.0):
        from backtester.combo_search import Moments
        parts, cash = [], self.cash.to_numpy()[:, None]
        for v in self.specs.vehicle.unique():
            sp = self.specs[self.specs.vehicle == v]
            for a in range(0, len(sp), chunk):
                c = sp.iloc[a:a + chunk]
                net = self._net(v, c["mask"].to_numpy(), c["rule"].to_numpy())
                parts.append(Moments.from_returns(net - cash, blocks, names, periods_per_year))
        return Moments.concat(parts)


cveh = {}
for c in COINS:
    cveh[c] = (ret[c], crypto_cost, None)
    cveh[f"{c} vt50"] = (ret[c], crypto_cost, banded_scale((0.5 / cvol[c]).clip(upper=1.0)))
bsearch = BookSearch(book_sig, cveh, cash_c)
bres = evaluate(bsearch, HOLDOUT, min_train_years=3, periods_per_year=365)
always = [pd.DataFrame({"a": True}, index=book_sig[0].index)] * 2
cb = BookSearch(always, cveh, cash_c)
bench_c = {"BTC+ETH 50/50 hold": cb.returns(0)}
summaries["B. BTC+ETH timing"] = report("B. BTC+ETH timing", bsearch, bres, bench_c, ppy=365)
print(f"[{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ C. strategy sleeves
t0 = time.time()
ETFS = ("SPY QQQ IWM EFA EEM TLT IEF GLD SLV DBC USO UUP FXE FXY XLK XLF XLE XLV XLI XLY XLP XLU XLB VNQ "
        "EWJ EWZ EWG FXI HYG LQD TIP DBA").split()
trend_etfs = [s for s in TREND_UNIVERSE if s != "BTC-USD"]
syms = sorted(set(ETFS + trend_etfs + list(INVERSE.values()) + SECTOR_SPDRS + [
    "UPRO", "SSO", "SHY", "TQQQ", "SVXY", "^VIX", "^VIX3M", "DIA", "AGG", "TMF", "UGL", "IJR", "BIL"]))
de = load_market(syms, start="2003-01-01", **d.extra)
etf_bt = Backtester(5_000, costs=COSTS["etf"], cash_rate=rf, min_trade_value=10)
runs = {}


def run(name, strategy, symbols, bt=etf_bt, data=de):
    try:
        runs[name] = bt.run(strategy, sub(data, symbols)).equity
    except Exception as e:
        print(f"sleeve {name} skipped: {e}")


run("upro_vt", VolTarget(symbol="UPRO", signal="SPY", leverage=3, target_vol=0.20), ["SPY", "UPRO"])
run("trend_1x", MultiAssetTrend(universe=trend_etfs, max_gross=1.0), trend_etfs + list(INVERSE.values()))
run("pension", PensionRebalance(), ["SPY", "TLT", "SSO"])
run("tom_tqqq", TurnOfMonth(symbol="TQQQ", risk_off="QQQ"), ["TQQQ", "QQQ"])
run("sso_200d", TrendFilter(risk_on="SSO", signal="SPY", risk_off="SHY"), ["SSO", "SPY", "SHY"])
run("svxy_vix", VixTermStructure(), ["SVXY", "^VIX", "^VIX3M"])
run("sector_rotation", SectorRotation(safe="SHY", market="SPY", ma=200), SECTOR_SPDRS + ["SPY", "SHY"])
run("rsi2", RSIMeanReversion(), ["SPY", "QQQ", "IWM", "DIA"])
run("dual_momentum", DualMomentum(universe=["SPY", "EFA"], safe="AGG"), ["SPY", "EFA", "AGG"])
run("risk_parity", RiskParity(), ["UPRO", "TMF", "UGL", "DBC", "SPY", "TLT", "GLD"])
run("seasonality", SameMonthSeasonality(universe=SECTOR_SPDRS, safe="SHY"), SECTOR_SPDRS + ["SHY"])
run("etf_breakout", Breakout(entry=20, exit=10, max_positions=10), ETFS)
run("qqq_allocator", FundamentalAllocator(macro="macro" in d.extra),
    ["QQQ", "EFA", "EEM", "IJR", "SPY", "IEF", "BIL", "HYG", "SSO"])
try:
    from backtester.core import CostModel
    from backtester.data.fx import load_fx_market
    from backtester.strategies import FXCarry
    fx = FXCarry()
    runs["fx_carry"] = Backtester(5_000, costs=CostModel(spread_bps=2, slippage_bps=1), allow_short=True,
                                  max_leverage=fx.p["max_gross"], cash_rate=rf).run(fx, load_fx_market(start="2004-01-01")).equity
except Exception as e:
    print("sleeve fx_carry skipped:", e)
try:
    from backtester.strategies import CreditSpreads, Wheel
    from backtester.strategies.spreads import load_spread_inputs
    from backtester.strategies.wheel import load_wheel_inputs
    runs["put_spreads"] = CreditSpreads(take_profit=1.0, exit_dte=None).run(load_spread_inputs(), 5_000).equity
    runs["wheel_xlf"] = Wheel(symbol="XLF").run(load_wheel_inputs("XLF"), 5_000).equity
except Exception as e:
    print("option sleeves skipped:", e)
COINS10 = ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "LINK-USD", "XLM-USD", "TRX-USD"]
dk = load_market(COINS10, start="2017-11-09")
crypto_bt = Backtester(5_000, costs=COSTS["crypto"], cash_rate=rf, min_trade_value=10)
run("crypto_ensemble", Sleeves(sleeves=[(EnsembleTrend(top_n=2, min_history=60, target_vol=0.5), COINS, 1.0)]),
    COINS, crypto_bt, dk)
run("crypto_breakout", Breakout(entry=20, exit=10), COINS, crypto_bt, dk)
run("crypto_vt_breakout", VolTargetBreakout(entry=20, exit=10), COINS, crypto_bt, dk)
run("crypto_xs_momentum", CrossSectionalMomentum(top_n=3, lookback=21), COINS10, crypto_bt, dk)
runs["spy_hold"] = etf_bt.run(BuyAndHold(symbols=["SPY"]), sub(de, ["SPY"])).equity

days = de.bars["SPY"].index
eq = pd.DataFrame({k: v.reindex(days, method="ffill") for k, v in runs.items()})
sleeve_ret = eq.pct_change(fill_method=None)
if out:
    eq.to_csv(out / "sleeve_equity.csv")
print("\nSleeves (first day with returns):")
print(sleeve_ret.apply(lambda s: s.first_valid_index()).astype(str).to_string())
spy_ret = sleeve_ret.pop("spy_hold")

for label, cols, start, min_train in [
    ("C1. sleeves incl. crypto", list(sleeve_ret.columns), "2019-01-01", 2.5),
    ("C2. sleeves without crypto", [c for c in sleeve_ret.columns if not c.startswith("crypto")], "2012-01-01", 4),
]:
    r = sleeve_ret[cols].loc[start:].dropna(axis=1, how="all")
    late = r.columns[r.iloc[0].isna()]
    if len(late):
        print(f"{label}: dropping sleeves that start late: {list(late)}")
        r = r.drop(columns=late)
    r = r.fillna(0.0)
    c = daily_cash(r.index)
    s = SleeveSearch(r, c)
    rs = evaluate(s, HOLDOUT, min_train_years=min_train)
    # The no-search control: hold every sleeve in equal weight.
    summaries[label] = report(label, s, rs, {"SPY hold": spy_ret.reindex(r.index).fillna(0.0),
                                             "all sleeves equal weight": r.mean(axis=1)})
print(f"[{time.time() - t0:.0f}s]")

print("\n==== Summary")
for k, v in summaries.items():
    t = v["table"]
    print(f"{k}: {v['n']:,} trials; walk-forward Sharpe {t.iloc[0]['sharpe']:.2f} (top-20 {t.iloc[1]['sharpe']:.2f}) "
          f"vs {t.index[2]} {t.iloc[2]['sharpe']:.2f}; "
          f"deflated Sharpe {v['dsr']['dsr']:.2f}; PBO {v['pbo']['pbo']:.2f}")
