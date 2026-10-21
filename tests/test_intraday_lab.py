import numpy as np
import pandas as pd
import pytest

from backtester import intraday_lab as L


def _day(day: str, path: np.ndarray, start: str = "09:30", volume: float = 1000.0) -> pd.DataFrame:
    t = pd.date_range(f"{day} {start}", periods=len(path), freq="1min")
    return pd.DataFrame({"open": path, "high": path, "low": path, "close": path, "volume": volume}, index=t)


def _grid(paths: list[np.ndarray]) -> L.Grid:
    days = pd.bdate_range("2024-01-01", periods=len(paths))
    return L.Grid(pd.concat([_day(str(d.date()), p) for d, p in zip(days, paths)]))


def _noise(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out, px = [], 100.0
    for _ in range(n):
        p = px * (1 + np.cumsum(rng.normal(0, 0.0003, 390)))
        out.append(p)
        px = p[-1]
    return out


def test_simulate_fills_one_minute_late_and_charges_both_sides():
    g = _grid([np.linspace(100, 101, 390)])
    pos = np.zeros((1, 390))
    pos[0, 100:200] = 1.0
    out = L.simulate(g, pos, cost_bps=0.0)
    px = g.px[0]
    assert out["ret"].iloc[0] == pytest.approx(np.sum(px[102:202] / px[101:201] - 1))
    costly = L.simulate(g, pos, cost_bps=10.0)
    assert out["ret"].iloc[0] - costly["ret"].iloc[0] == pytest.approx(2 * 10 / 1e4)


def test_simulate_flattens_at_the_close():
    g = _grid([np.linspace(100, 101, 390)])
    out = L.simulate(g, np.ones((1, 390)), cost_bps=1.0)
    assert out["turnover"].iloc[0] == pytest.approx(2.0)


def test_opening_range_breakout_goes_long_on_a_trend_day():
    trend = np.r_[np.full(30, 100.0) + np.tile([0.1, -0.1], 15), np.linspace(100.2, 103, 360)]
    g = _grid(_noise(15) + [trend])
    pos = L.opening_range(g, minutes=30, mode="breakout", stop="range")
    assert pos[-1, 29] == 0 and pos[-1, -1] == 1
    assert L.simulate(g, pos, 0.0)["ret"].iloc[-1] > 0.02


def test_long_only_rules_never_short():
    g = _grid(_noise(30, seed=3))
    for fn, kw in [(L.opening_range, {}), (L.vwap_trend, {}), (L.vwap_reversion, {"k": 0.1}),
                   (L.noise_area, {}), (L.momentum_window, {}), (L.gap, {"threshold": 0.0})]:
        assert fn(g, long_only=True, **kw).min() >= 0, fn.__name__


def test_vwap_trend_is_long_above_vwap_and_short_below():
    g = _grid([np.linspace(100, 101, 390), np.linspace(101, 100, 390)])
    pos = L.vwap_trend(g, check_every=1, first=5)
    assert (pos[0, 10:] == 1).all() and (pos[1, 10:] == -1).all()


def test_gap_fades_a_big_gap_up():
    paths = _noise(15)
    paths.append(paths[-1][-1] * 1.03 * np.ones(390))
    g = _grid(paths)
    pos = L.gap(g, threshold=0.5, direction="fade", entry=1, exit=None)
    assert pos[-1, 5] == -1 and pos[-1, 0] == 0


def test_hour_of_day_picks_the_hour_that_always_rises():
    days = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(2)
    frames, px = [], 100.0
    for d in days:
        r = rng.normal(0, 0.0002, 1440)
        r[600:660] += 0.0001  # hour 10 drifts up
        p = px * np.cumprod(1 + r)
        frames.append(_day(str(d.date()), p, start="00:00"))
        px = p[-1]
    g = L.Grid(pd.concat(frames), start=0, length=1440)
    pos = L.hour_of_day(g, lookback=180, rule="top2")
    assert pos[-1, 610] == 1


def test_walk_forward_trades_last_years_best_column():
    idx = pd.bdate_range("2021-01-01", "2023-12-31")
    rng = np.random.default_rng(0)
    good = np.where(idx.year < 2023, 0.002, -0.002) + rng.normal(0, 0.001, len(idx))
    bad = rng.normal(0, 0.001, len(idx))
    rets = pd.DataFrame({"good": good, "bad": bad}, index=idx)
    wf, picks = L.walk_forward(rets, [2022, 2023])
    assert picks == [(2022, "good"), (2023, "good")]
    assert wf.index.min().year == 2022 and wf[wf.index.year == 2023].mean() < 0


def test_session_split_sums_to_buy_and_hold_without_costs():
    g = _grid(_noise(5))
    s = L.session_split(g, cost_bps=0.0).iloc[1:]
    both = (1 + s["overnight only"]) * (1 + s["session only"]) - 1
    assert np.allclose(both, s["buy and hold"])


def test_high_vol_days_keeps_only_the_volatile_regime():
    calm, wild = _noise(150, seed=4), []
    rng = np.random.default_rng(5)
    px = calm[-1][-1]
    for _ in range(60):
        p = px * (1 + np.cumsum(rng.normal(0, 0.0015, 390)))
        wild.append(p)
        px = p[-1]
    g = _grid(calm + wild)
    keep = L.high_vol_days(g, 0.75)
    assert keep[-40:].mean() > 0.9 and keep[:150].mean() < 0.1
    assert L.high_vol_days(g, 0.0).all()


def test_family_positions_labels_vol_filtered_copies():
    g = _grid(_noise(20))
    labels = [k for k, _ in L.family_positions(g, "VWAP trend", L.vwap_trend, {"check_every": [5], "long_only": [True]})]
    assert labels == [f"check_every=5, long_only=True, vol_q={q}" for q in L.VOL_QUANTILES]
