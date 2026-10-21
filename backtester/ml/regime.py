"""Market regimes from a statistical jump model, and a strategy that switches
between keepers by regime.

Statistical jump models (Nystrup, Lindstrom and Madsen 2020; Shu, Yu and
Mulvey 2024, "Downside Risk Reduction Using Regime-Switching Signals: A
Statistical Jump Model Approach", J. Asset Management) cluster days by a few
return features, with a fixed penalty `lam` for every change of regime, so
regimes persist instead of flickering like a hidden Markov model's. Shu et
al. find that switching an equity index to cash in the model's bear regime
cut drawdowns and raised Sharpe out of sample in the US, Germany and Japan
from 1990 to 2023, with costs and a one-day delay.

Fit: alternate between centroids (means of each regime's days) and the
regime path (dynamic programming over squared distance + lam per jump).
Online use: with the centroids fixed, the forward pass's cheapest state at
day t depends only on data up to t, so one pass gives a regime for every day
without lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy


def jump_features(close: pd.Series, rf: pd.Series | None = None, halflives=(5, 10, 21)) -> pd.DataFrame:
    """Shu et al.'s features: EWM mean excess return, log EWM downside
    deviation and their ratio (an EWM Sortino ratio) at several half-lives."""
    r = np.log(close).diff()
    if rf is not None:
        r = r - (rf.reindex(r.index).ffill().fillna(0.0) / 252)
    out = {}
    for h in halflives:
        mean = r.ewm(halflife=h).mean()
        dd = np.sqrt((r.clip(upper=0) ** 2).ewm(halflife=h).mean())
        out[f"ret_{h}"] = mean
        out[f"dd_{h}"] = np.log(dd + 1e-6)
        out[f"sortino_{h}"] = mean / (dd + 1e-6)
    return pd.DataFrame(out).iloc[max(halflives) :]


def _path(X: np.ndarray, mu: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal regime path (backtracked) and the online states (argmin of
    the forward cost at each day)."""
    loss = 0.5 * ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    T, K = loss.shape
    cost = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    cost[0] = loss[0]
    pen = lam * (1 - np.eye(K))
    for t in range(1, T):
        tot = cost[t - 1][:, None] + pen  # from i to j
        back[t] = tot.argmin(axis=0)
        cost[t] = loss[t] + tot.min(axis=0)
    online = cost.argmin(axis=1)
    s = np.empty(T, dtype=int)
    s[-1] = cost[-1].argmin()
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s, online


class JumpModel:
    """Two-state (or K-state) statistical jump model on z-scored features."""

    def __init__(self, lam: float = 50.0, n_states: int = 2, n_init: int = 5, max_iter: int = 30, seed: int = 0):
        self.lam, self.K, self.n_init, self.max_iter, self.seed = lam, n_states, n_init, max_iter, seed

    def fit(self, feats: pd.DataFrame, ret: pd.Series) -> "JumpModel":
        self.mu_, self.sd_ = feats.mean(), feats.std().replace(0, 1)
        X = self._z(feats)
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None, None)
        for _ in range(self.n_init):
            mu = X[rng.choice(len(X), self.K, replace=False)]
            s = None
            for _ in range(self.max_iter):
                s_new, _ = _path(X, mu, self.lam)
                if s is not None and (s_new == s).all():
                    break
                s = s_new
                mu = np.array([X[s == k].mean(axis=0) if (s == k).any() else mu[k] for k in range(self.K)])
            obj = 0.5 * ((X - mu[s]) ** 2).sum() + self.lam * (np.diff(s) != 0).sum()
            if obj < best[0]:
                best = (obj, mu, s)
        _, self.centroids_, s = best
        # Order states from best to worst average return on their own days,
        # so state 0 is "bull" whatever order clustering found them in.
        same_day = ret.reindex(feats.index).to_numpy()
        means = [np.nanmean(same_day[s == k]) if (s == k).any() else -np.inf for k in range(self.K)]
        order = np.argsort(means)[::-1]
        self.centroids_ = self.centroids_[order]
        self.train_states_ = pd.Series(np.argsort(order)[s], index=feats.index)
        return self

    def _z(self, feats):
        return ((feats - self.mu_) / self.sd_).clip(-5, 5).fillna(0.0).to_numpy()

    def online_states(self, feats: pd.DataFrame) -> pd.Series:
        """Regime at each day using only data up to that day."""
        _, online = _path(self._z(feats), self.centroids_, self.lam)
        return pd.Series(online, index=feats.index)


def _switch_sharpe(states: pd.Series, ret: pd.Series, rf: pd.Series, cost: float) -> float:
    """Sharpe of holding the asset in state 0 (bull), cash otherwise, with a
    one-day delay and `cost` per switch."""
    on = (states == 0).astype(float).shift(1).fillna(0.0)
    r = ret.reindex(on.index).fillna(0.0)
    cash = rf.reindex(on.index).ffill().fillna(0.0) / 252
    pnl = on * r + (1 - on) * cash - on.diff().abs().fillna(0.0) * cost - cash
    return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else -np.inf


def walk_forward_regimes(
    close: pd.Series,
    rf: pd.Series,
    first_test: str,
    test_end: str,
    lams=(0.0, 10.0, 30.0, 60.0, 100.0),
    val_years: int = 4,
    cost: float = 0.0005,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Out-of-sample regimes, refit each calendar year (expanding window).

    For each test year, `lam` is picked by fitting on data up to `val_years`
    before it and scoring the bull-or-cash switch on the following
    `val_years` (online states), as Shu et al. do; the model is then refit on
    everything before the test year with that `lam`. Returns the daily
    regime (0 = bull) and the `lam` picked per year.
    """
    feats = jump_features(close, rf).dropna()
    ret = close.pct_change(fill_method=None)
    parts, picks = [], []
    for year in range(pd.Timestamp(first_test).year, pd.Timestamp(test_end).year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        val_start = pd.Timestamp(f"{year - val_years}-01-01")
        fit_v = feats[feats.index < val_start]
        val = feats[(feats.index >= val_start) & (feats.index < start)]
        scores = {}
        for lam in lams:
            m = JumpModel(lam=lam, seed=seed).fit(fit_v, ret)
            st = m.online_states(feats[feats.index < start]).loc[val.index]
            scores[lam] = _switch_sharpe(st, ret, rf, cost)
        lam = max(scores, key=scores.get)
        m = JumpModel(lam=lam, seed=seed).fit(feats[feats.index < start], ret)
        end = min(pd.Timestamp(test_end), pd.Timestamp(f"{year}-12-31"))
        st = m.online_states(feats[feats.index <= end])
        parts.append(st[(st.index >= start) & (st.index <= end)])
        picks.append({"year": year, "lam": lam, **{f"val_sharpe_{k:g}": v for k, v in scores.items()}})
    return pd.concat(parts), pd.DataFrame(picks).set_index("year")


class RegimeSwitch(Strategy):
    """Hold `bull`'s targets in regime 0 and `bear`'s otherwise.

    states: daily Series of regimes (0 = bull), known at each close.
    bull / bear: Strategies run on the same data; bear=None means cash.
    """

    name = "regime_switch"
    params = {"states": None, "bull": None, "bear": None}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        idx = data.index
        st = self.p["states"].reindex(idx).ffill()
        bull = self.p["bull"].generate_targets(data).reindex(idx).ffill().reindex(columns=data.symbols).fillna(0.0)
        if self.p["bear"] is not None:
            bear = self.p["bear"].generate_targets(data).reindex(idx).ffill().reindex(columns=data.symbols).fillna(0.0)
        else:
            bear = bull * 0.0
        on = (st == 0).to_numpy()[:, None]
        w = pd.DataFrame(np.where(on, bull, bear), index=idx, columns=data.symbols)
        return w[st.notna()]
