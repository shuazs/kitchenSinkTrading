import numpy as np
import pandas as pd

from backtester import Backtester, CostModel, MarketData
from backtester.data.synthetic import synthetic_bars, synthetic_market
from backtester.research import event_study, summarize_events
from backtester.strategies import EventHold, HedgeFundClone, MultiAssetTrend, PensionRebalance, RiskParity, best_ideas
from backtester.strategies.turn_of_month import month_position

FREE = CostModel(spread_bps=0, slippage_bps=0)


def trending(n=800, drift=0.001, seed=0, start="2015-01-01"):
    """Prices with a steady drift (up for drift>0) and small noise."""
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(drift + 0.005 * rng.standard_normal(n)))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6}, index=idx)


def test_trend_goes_long_uptrend_and_into_inverse_etf_for_downtrend():
    data = MarketData({"SPY": trending(drift=0.001), "TLT": trending(drift=-0.001, seed=1), "TBF": trending(seed=2, drift=0.0)})
    w = MultiAssetTrend(universe=["SPY", "TLT"]).generate_targets(data)
    late = w.iloc[-1]
    assert late["SPY"] > 0 and late["TLT"] == 0 and late["TBF"] > 0
    assert w.abs().sum(axis=1).max() <= 1.5 + 1e-9


def test_trend_cash_mode_never_shorts_and_respects_vol_target():
    data = synthetic_market(["SPY", "TLT", "GLD"], n=900, sigma=0.15)
    w = MultiAssetTrend(short="cash", target_vol=0.10, band=0).generate_targets(data)
    assert (w >= 0).all().all()
    res = Backtester(costs=FREE, max_leverage=1.5).run(MultiAssetTrend(short="cash", target_vol=0.10, band=0), data)
    vol = res.equity.pct_change().iloc[300:].std() * np.sqrt(252)
    assert vol < 0.16


def test_trend_rebalances_only_at_week_ends():
    data = synthetic_market(["SPY", "TLT"], n=600)
    w = MultiAssetTrend(short="cash").generate_targets(data)
    # Business-day synthetic data: Fridays, plus the last bar of a partial final week.
    assert set(w.index[:-1].dayofweek) == {4}


def test_pension_signal_windows_and_side():
    n = 1200
    idx = pd.bdate_range("2015-01-01", periods=n)
    stock = trending(n, drift=0.0005)
    bond = trending(n, drift=0.0005, seed=7)
    data = MarketData({"SPY": stock, "TLT": bond, "SSO": stock, "SH": stock})
    st = PensionRebalance(days=4, min_months=12, when_stocks_lead="SH")
    w = st.generate_targets(data)
    gap = st.gaps(data)["gap"]
    _, from_end = month_position(idx)
    for side, sym in ((-1, "SSO"), (1, "SH")):
        on = (w[sym] > 0).to_numpy()
        assert on.any()
        # Positions sit only from the signal close to the day before month end.
        assert set(from_end[on]) == {5, 4, 3, 2}
        # And only in months whose gap is in the right tail of earlier months.
        for m in pd.DatetimeIndex(idx[on]).to_period("M").unique():
            g = gap[gap.index.to_period("M") == m].iloc[0]
            prior = gap[gap.index.to_period("M") < m]
            assert (g < prior.quantile(1 / 3)) if side == -1 else (g > prior.quantile(2 / 3))
    both = (w["SSO"] > 0) & (w["SH"] > 0)
    assert not both.any()
    assert (w.loc[(w["SSO"] == 0) & (w["SH"] == 0), "SPY"] == 1).all()


def test_risk_parity_inverse_vol_and_cap():
    data = MarketData({"A": synthetic_bars(sigma=0.10, seed=1, n=600), "B": synthetic_bars(sigma=0.30, seed=2, n=600)})
    w = RiskParity(assets={"A": ("A", 1), "B": ("B", 1)}, target_vol=None, max_gross=1.0).generate_targets(data)
    last = w.iloc[-1]
    assert abs(last.sum() - 1.0) < 1e-9 and last["A"] > 2 * last["B"]
    w2 = RiskParity(assets={"A": ("A", 3), "B": ("B", 1)}, target_vol=0.5, max_gross=1.0).generate_targets(data)
    assert w2.sum(axis=1).max() <= 1.0 + 1e-9


def test_event_hold_buys_next_open_and_exits_after_hold_days():
    data = synthetic_market(["AAA", "BBB"], n=200)
    day = data.index[50]
    ev = pd.DataFrame({"date": [day], "symbol": ["AAA"]})
    data = MarketData(data.bars, extra={"events": ev})
    res = Backtester(costs=FREE).run(EventHold(hold_days=10, min_dollar_volume=0, min_price=0), data)
    held = res.positions["AAA"] > 0
    assert held[held].index[0] == data.index[51]
    assert held.sum() == 10


def test_event_study_entry_and_excess():
    data = MarketData({"X": trending(100, drift=0.01), "B": trending(100, drift=0.0, seed=3)})
    s = event_study(data, pd.DataFrame({"date": [data.index[10]], "symbol": ["X"]}), horizons=(5,), benchmark="B")
    o, c = data.bars["X"]["open"], data.bars["X"]["close"]
    assert np.isclose(s["r5"].iloc[0], c.iloc[15] / o.iloc[11] - 1)
    assert s["x5"].iloc[0] > 0
    assert summarize_events(s).loc["r5", "n"] == 1


def _holdings():
    rows = []
    period = pd.Timestamp("2020-03-31")
    for m in range(4):
        for r, cusip in enumerate(["C1", "C2", "C3", "C4"][m % 2 :][:3], start=1):
            rows.append(dict(manager_cik=f"M{m}", period=period, filed=pd.Timestamp("2020-05-10"), total_usd=1e9,
                             n_positions=20, top10_share=0.7, rank=r, cusip=cusip, issuer=cusip, weight=0.2 / r))
    # A diversified giant whose votes must not count.
    rows.append(dict(manager_cik="BIG", period=period, filed=pd.Timestamp("2020-05-10"), total_usd=1e12,
                     n_positions=3000, top10_share=0.2, rank=1, cusip="C9", issuer="C9", weight=0.05))
    # A late filer (after the 45-day deadline) is not seen on the signal date.
    rows.append(dict(manager_cik="LATE", period=period, filed=pd.Timestamp("2020-06-30"), total_usd=1e9,
                     n_positions=20, top10_share=0.7, rank=1, cusip="C9", issuer="C9", weight=0.5))
    return pd.DataFrame(rows)


def test_best_ideas_counts_concentrated_on_time_managers():
    ideas = best_ideas(_holdings(), top_k=3)
    assert "C9" not in set(ideas["cusip"])
    assert ideas.iloc[0]["cusip"] in ("C2", "C3") and ideas.iloc[0]["votes"] == 4
    assert ideas["date"].iloc[0] == pd.Timestamp("2020-05-16")


def test_hedge_fund_clone_holds_top_names_from_signal_day():
    ideas = best_ideas(_holdings(), top_k=3).assign(symbol=lambda d: d["cusip"])
    data = synthetic_market(["C1", "C2", "C3", "C4"], n=600, start="2019-06-03")
    data = MarketData(data.bars, extra={"best_ideas": ideas})
    w = HedgeFundClone(n_stocks=2, min_votes=2).generate_targets(data)
    assert len(w) == 1 and w.index[0] == pd.Timestamp("2020-05-18")  # the 16th is a Saturday
    assert set(w.columns[w.iloc[0] > 0]) == {"C2", "C3"}
