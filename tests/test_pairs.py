import numpy as np
import pandas as pd
import pytest

from backtester.pairs import (asset_leg_trades, fade_contract_spread, half_life, kalshi_fee, rolling_hedge,
                              zscore)


def test_kalshi_fee_peaks_at_50_cents():
    assert kalshi_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_fee(0.1) == pytest.approx(0.0063)


def test_half_life_recovers_ar1():
    rng = np.random.default_rng(0)
    phi, n = 0.8, 20000
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    assert half_life(pd.Series(s)) == pytest.approx(np.log(0.5) / np.log(phi), rel=0.1)


def test_rolling_hedge_finds_beta_without_lookahead():
    rng = np.random.default_rng(1)
    x = pd.Series(np.cumsum(rng.normal(size=500)))
    y = 2.0 + 3.0 * x + rng.normal(scale=0.1, size=500)
    h = rolling_hedge(y, x, 100)
    assert h.beta.iloc[:100].isna().all() and h.beta.iloc[100] == pytest.approx(3.0, abs=0.05)
    assert h.spread.dropna().abs().max() < 1


def test_zscore_uses_only_past():
    s = pd.Series([0.0, 1, 0, 1, 0, 1, 0, 1, 10])
    z = zscore(s, 8)
    assert z.iloc[-1] > 10 and np.isnan(z.iloc[0])


def test_asset_leg_trades_shorts_rich_asset_and_charges_costs():
    idx = pd.date_range("2025-01-01", periods=6, freq="h")
    lp = pd.Series(np.log([100, 100, 100, 99, 99, 99]), index=idx)
    z = pd.Series([0, 3, 0, 0, 0, 0.0], index=idx)
    tr = asset_leg_trades(lp, z, entry=2, hold=2, cost_bp=4, delay=1)
    # signal at bar 1, enter at bar 2's close, hold 2 bars: 100 -> 99 short
    assert len(tr) == 1 and tr.side.iloc[0] == -1
    assert tr.gross_bp.iloc[0] == pytest.approx(-np.log(0.99) * 1e4)
    assert tr.net_bp.iloc[0] == pytest.approx(tr.gross_bp.iloc[0] - 4)


def _prints(rows, result="no"):
    t0 = pd.Timestamp("2025-01-02 15:00", tz="UTC")
    df = pd.DataFrame(rows, columns=["sec", "p", "F0", "yes"])
    df["created_time"] = t0 + pd.to_timedelta(df.sec, unit="s")
    df["close_time"] = t0 + pd.Timedelta("1h")
    df["ticker"], df["result"] = "X", result
    df["e"] = df.p - df.F0
    return df


def test_fade_fills_on_opposite_side_so_bounce_is_not_profit():
    # bid 0.45 / ask 0.55 around fair 0.50: YES-taker prints at 0.55 look rich,
    # but selling fills at the bid and buying back at the ask loses the spread.
    rows = [(0, 0.55, 0.50, 1), (60, 0.45, 0.50, 0), (400, 0.55, 0.50, 1), (500, 0.45, 0.50, 0)]
    tr = fade_contract_spread(_prints(rows), threshold=0.04, hold_min=5)
    first = tr.iloc[0]
    assert first.side == -1 and first.entry == 0.45 and first.exit == 0.55
    assert first.gross == pytest.approx(-0.10)
    assert first.net == pytest.approx(-0.10 - kalshi_fee(0.45) - kalshi_fee(0.55))


def test_fade_holds_to_settlement_without_exit_fee():
    rows = [(0, 0.70, 0.50, 1), (60, 0.68, 0.50, 0)]
    tr = fade_contract_spread(_prints(rows, result="no"), threshold=0.1, hold_min=None)
    assert len(tr) == 1 and tr.settled.iloc[0]
    assert tr.net.iloc[0] == pytest.approx(0.68 - kalshi_fee(0.68))
