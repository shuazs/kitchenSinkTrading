"""Exhaustive search over combinations of strategy signals, read honestly.

Each strategy is reduced to a daily 0/1 signal ("risk on") or to a sleeve's
daily net return. Every subset of signals is combined with a simple rule
(AND, OR, majority vote, share of signals on) and traded in a vehicle, or
every subset of sleeves is held equal- or inverse-vol-weighted. With
hundreds of thousands of trials the best in-sample result is mostly luck,
so the search is judged by:

- walk-forward selection: at each half-year, pick the best combination on
  all data before it and trade it on the next half-year only;
- a locked holdout (data after `holdout`) that no choice looks at;
- the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) of the best
  in-sample combination given the number and spread of trials;
- the probability of backtest overfitting, PBO (Bailey et al., 2017,
  combinatorially symmetric cross-validation over half-year blocks).

Everything is vectorised: a combination is a column of a (days x combos)
array, not a Backtester run. Statistics are kept as per-block sums so any
window's Sharpe is a cheap sum. Signal timing matches the engine: a signal
at close t trades at open t+1 (`open_to_open`).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from .core import CostModel

RULES = ("and", "or", "majority", "share")
EULER = 0.5772156649


def unit_cost(cm: CostModel) -> float:
    """Cost of trading 100% of equity once under a CostModel (fee + half spread + slippage)."""
    return cm.pct_fee + (cm.spread_bps / 2 + cm.slippage_bps) / 1e4


def open_to_open(bars: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Return earned by a weight set at close t: open t+1 to open t+2, on `index`."""
    o = bars["open"].reindex(index).ffill()
    return (o.shift(-2) / o.shift(-1) - 1).rename(None)


def subset_masks(n: int, max_size: int | None = None) -> np.ndarray:
    """Every non-empty subset of n items as a (subsets x n) bool array."""
    rows = []
    for k in range(1, (max_size or n) + 1):
        for c in combinations(range(n), k):
            m = np.zeros(n, bool)
            m[list(c)] = True
            rows.append(m)
    return np.array(rows)


def rule_specs(masks: np.ndarray) -> list[tuple[int, str]]:
    """(mask row, rule) pairs that give distinct exposures: one rule for a
    single signal, and/or/share for pairs, all four rules from three up."""
    sizes = masks.sum(axis=1)
    out = []
    for i, k in enumerate(sizes):
        rules = ("single",) if k == 1 else ("and", "or", "share") if k == 2 else RULES
        out += [(i, r) for r in rules]
    return out


def combine(signals: np.ndarray, masks: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """(T, n) 0/1 signals, (m, n) masks, (m,) rule names -> (T, m) exposure in [0, 1].

    and: all on; or: any on; majority: at least half on; share: fraction on.
    """
    on = signals.astype(np.float32) @ masks.T.astype(np.float32)
    size = masks.sum(axis=1).astype(np.float32)
    out = np.empty_like(on)
    for rule in np.unique(rules):
        c = rules == rule
        o, s = on[:, c], size[c]
        if rule in ("and", "single"):
            out[:, c] = o >= s - 0.5
        elif rule == "or":
            out[:, c] = o > 0.5
        elif rule == "majority":
            out[:, c] = o >= s / 2
        elif rule == "share":
            out[:, c] = o / s
        else:
            raise ValueError(rule)
    return out


def timing_returns(weight: np.ndarray, asset_ret: np.ndarray, cash_ret: np.ndarray, cost: float) -> np.ndarray:
    """Net daily returns of holding `weight` (T, m) of an asset, rest in cash.

    weight[t] is set at close t and earns asset_ret[t] (open t+1 to open t+2);
    each change pays `cost` per unit of weight traded.
    """
    turn = np.abs(np.diff(weight, axis=0, prepend=0.0))
    return weight * asset_ret[:, None] + (1 - weight) * cash_ret[:, None] - cost * turn


def banded_scale(raw: pd.Series, band: float = 0.1) -> pd.Series:
    """Hold a sizing series fixed except on each week's last day when it moved by `band` or more."""
    checks = raw.index.isin(raw.groupby(raw.index.to_period("W")).tail(1).index)
    held, cur = np.full(len(raw), np.nan), np.nan
    for i, (x, chk) in enumerate(zip(raw.to_numpy(float), checks)):
        if chk and not np.isnan(x) and (np.isnan(cur) or abs(x - cur) >= band):
            cur = x
        held[i] = cur
    return pd.Series(held, index=raw.index)


# ---------------------------------------------------------------- block statistics


def half_year_blocks(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Label each day with its half-year (block number, block names)."""
    lab = pd.Series(index.year.astype(str) + np.where(index.month <= 6, "H1", "H2"), index=index)
    names = list(dict.fromkeys(lab))
    return lab.map({n: i for i, n in enumerate(names)}).to_numpy(), names


@dataclass
class Moments:
    """Per-block sums of excess returns for every combination."""

    s1: np.ndarray  # (B, m) sum of daily excess returns
    s2: np.ndarray  # (B, m) sum of squares
    n: np.ndarray  # (B,) days per block
    names: list[str]
    periods_per_year: float

    @classmethod
    def from_returns(cls, excess: np.ndarray, blocks: np.ndarray, names, periods_per_year=252.0) -> Moments:
        starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
        x = np.nan_to_num(excess.astype(np.float64))
        return cls(np.add.reduceat(x, starts, axis=0), np.add.reduceat(x * x, starts, axis=0),
                   np.diff(np.r_[starts, len(blocks)]).astype(float), list(names), periods_per_year)

    @classmethod
    def concat(cls, parts: list[Moments]) -> Moments:
        p = parts[0]
        return cls(np.hstack([q.s1 for q in parts]), np.hstack([q.s2 for q in parts]), p.n, p.names, p.periods_per_year)

    def sharpe(self, sel) -> np.ndarray:
        """Annualised Sharpe of every combination over the blocks in `sel` (bool mask or indices)."""
        sel = np.asarray(sel)
        if sel.dtype != bool:
            sel = np.isin(np.arange(len(self.n)), sel)
        w = sel.astype(float)
        n = w @ self.n
        mu = (w @ self.s1) / n
        var = (w @ self.s2) / n - mu * mu
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = mu / np.sqrt(np.maximum(var, 1e-18)) * np.sqrt(self.periods_per_year)
        return np.where(var > 1e-14, sr, 0.0)


def walk_forward_picks(mom: Moments, first_test: int, last_test: int, top: int = 1) -> dict[int, np.ndarray]:
    """For each test block b in [first_test, last_test], the `top` combinations
    with the best Sharpe on blocks 0..b-1 (expanding window)."""
    picks = {}
    for b in range(first_test, last_test + 1):
        sr = mom.sharpe(np.arange(b))
        picks[b] = np.argsort(-sr)[:top]
    return picks


def pbo(mom: Moments, blocks, n_splits: int = 400, seed: int = 0) -> dict:
    """Probability of backtest overfitting by CSCV on the given blocks.

    Each split puts half the blocks in-sample; the in-sample winner's
    out-of-sample Sharpe is ranked among all combinations. PBO is the share
    of splits where it lands in the bottom half. Also returns the median
    out-of-sample percentile of the winner and how often its OOS Sharpe > 0.
    """
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks)
    pct, pos = [], []
    for _ in range(n_splits):
        is_ = rng.choice(blocks, len(blocks) // 2, replace=False)
        oos = np.setdiff1d(blocks, is_)
        best = int(np.argmax(mom.sharpe(is_)))
        sr_oos = mom.sharpe(oos)
        pct.append((sr_oos < sr_oos[best]).mean())
        pos.append(sr_oos[best] > 0)
    pct = np.array(pct)
    return {"pbo": float((pct <= 0.5).mean()), "median_oos_percentile": float(np.median(pct)),
            "winner_oos_positive": float(np.mean(pos)), "splits": n_splits}


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray, n_trials: int | None = None,
                    periods_per_year: float = 252.0) -> dict:
    """Deflated Sharpe ratio of the chosen strategy's daily excess `returns`.

    trial_sharpes: annualised Sharpe of every trial on the same window.
    Probability that the true Sharpe is above zero after allowing for the
    best-of-N selection, skew and fat tails. 0.95 is the usual bar.
    """
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    t = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd
    z = (r - r.mean()) / sd
    skew, kurt = (z ** 3).mean(), (z ** 4).mean()
    n = n_trials or len(trial_sharpes)
    var_sr = np.var(np.asarray(trial_sharpes) / np.sqrt(periods_per_year))
    nd = NormalDist()
    sr0 = 0.0 if n <= 1 else np.sqrt(var_sr) * ((1 - EULER) * nd.inv_cdf(1 - 1 / n) + EULER * nd.inv_cdf(1 - 1 / (n * np.e)))
    stat = (sr - sr0) * np.sqrt(t - 1) / np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return {"sharpe": sr * np.sqrt(periods_per_year), "expected_max_sharpe_of_noise": sr0 * np.sqrt(periods_per_year),
            "dsr": nd.cdf(stat), "n_trials": n, "days": t}


def stats(net: pd.Series, cash: pd.Series, periods_per_year: float = 252.0) -> dict:
    """CAGR, vol, Sharpe (vs cash), max drawdown and t-stat of a daily net return series."""
    net = net.dropna()
    ex = net - cash.reindex(net.index).fillna(0.0)
    eq = (1 + net).cumprod()
    years = len(net) / periods_per_year
    sr = ex.mean() / ex.std() * np.sqrt(periods_per_year) if ex.std() > 0 else 0.0
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1, "vol": net.std() * np.sqrt(periods_per_year), "sharpe": sr,
            "max_dd": (eq / eq.cummax() - 1).min(), "t": sr * np.sqrt(years), "years": years}


# ---------------------------------------------------------------- searches


class TimingSearch:
    """Every rule x subset of 0/1 signals, traded in every vehicle.

    signals: bool frame (days x signals), the state at each close.
    vehicles: name -> (asset_ret, cost, scale): asset_ret from `open_to_open`,
        cost from `unit_cost`, scale an optional sizing series (e.g. a vol
        target) multiplied into the exposure, or None.
    cash: daily cash return on the same index.
    """

    def __init__(self, signals: pd.DataFrame, vehicles: dict, cash: pd.Series, max_size: int | None = None):
        self.signals = signals.fillna(False).astype(bool)
        self.index = signals.index
        self.cash = cash.reindex(self.index).fillna(0.0)
        self.masks = subset_masks(signals.shape[1], max_size)
        self.rules = rule_specs(self.masks)
        self.vehicles = {k: (r.reindex(self.index).fillna(0.0).to_numpy(), c,
                             None if s is None else s.reindex(self.index).ffill().fillna(0.0).to_numpy())
                         for k, (r, c, s) in vehicles.items()}
        cols = list(signals.columns)
        self.specs = pd.DataFrame([
            {"vehicle": v, "rule": rule, "size": int(self.masks[i].sum()),
             "signals": "+".join(c for c, on in zip(cols, self.masks[i]) if on), "mask": i}
            for v in self.vehicles for i, rule in self.rules])

    def __len__(self):
        return len(self.specs)

    def _net(self, vehicle: str, mask_idx: np.ndarray, rules: np.ndarray) -> np.ndarray:
        ret, cost, scale = self.vehicles[vehicle]
        w = combine(self.signals.to_numpy(), self.masks[mask_idx], rules)
        if scale is not None:
            w = w * scale[:, None].astype(np.float32)
        return timing_returns(w, ret, self.cash.to_numpy(), cost)

    def moments(self, blocks, names, chunk: int = 6000, periods_per_year=252.0) -> Moments:
        parts, cash = [], self.cash.to_numpy()[:, None]
        for v in self.vehicles:
            sp = self.specs[self.specs.vehicle == v]
            for a in range(0, len(sp), chunk):
                c = sp.iloc[a:a + chunk]
                net = self._net(v, c["mask"].to_numpy(), c["rule"].to_numpy())
                parts.append(Moments.from_returns(net - cash, blocks, names, periods_per_year))
        return Moments.concat(parts)

    def returns(self, i: int) -> pd.Series:
        """Daily net returns of trial i."""
        r = self.specs.iloc[i]
        return pd.Series(self._net(r.vehicle, np.array([r["mask"]]), np.array([r.rule]))[:, 0], index=self.index)


class SleeveSearch:
    """Every subset of strategy sleeves, equal- or inverse-volatility weighted.

    rets: daily net returns of each sleeve (days x sleeves, no gaps).
    Weights are reset daily (an approximation of a monthly rebalance; the
    cost of rebalancing between sleeves is not charged). Inverse-vol uses each
    sleeve's trailing `vol_days` volatility, known at the prior close and
    refreshed at month ends.
    """

    def __init__(self, rets: pd.DataFrame, cash: pd.Series, vol_days: int = 126, max_size: int | None = None):
        self.rets = rets
        self.index = rets.index
        self.cash = cash.reindex(self.index).fillna(0.0)
        self.masks = subset_masks(rets.shape[1], max_size)
        vol = rets.rolling(vol_days, min_periods=21).std().shift()
        vol = vol.where(vol.index.to_series().dt.month.diff().ne(0)).ffill().bfill()
        self.inv = (1 / vol.clip(lower=1e-4)).to_numpy()
        cols = list(rets.columns)
        self.specs = pd.DataFrame([
            {"weighting": wt, "size": int(m.sum()), "signals": "+".join(c for c, on in zip(cols, m) if on), "mask": i}
            for wt in ("equal", "inverse_vol") for i, m in enumerate(self.masks)])

    def __len__(self):
        return len(self.specs)

    def _net(self, weighting: str, mask_idx: np.ndarray) -> np.ndarray:
        m = self.masks[mask_idx].T.astype(np.float64)
        r = self.rets.to_numpy()
        if weighting == "equal":
            return (r @ m) / m.sum(axis=0)
        return ((r * self.inv) @ m) / (self.inv @ m)

    def moments(self, blocks, names, chunk: int = 20000, periods_per_year=252.0) -> Moments:
        parts, cash = [], self.cash.to_numpy()[:, None]
        for wt in ("equal", "inverse_vol"):
            sp = self.specs[self.specs.weighting == wt]
            for a in range(0, len(sp), chunk):
                net = self._net(wt, sp["mask"].to_numpy()[a:a + chunk])
                parts.append(Moments.from_returns(net - cash, blocks, names, periods_per_year))
        return Moments.concat(parts)

    def returns(self, i: int) -> pd.Series:
        r = self.specs.iloc[i]
        return pd.Series(self._net(r.weighting, np.array([r["mask"]]))[:, 0], index=self.index)


def evaluate(search, holdout: str, min_train_years: float = 5.0, periods_per_year: float = 252.0,
             pbo_splits: int = 400, top: int = 20) -> dict:
    """Score a TimingSearch or SleeveSearch without trusting its best backtest.

    Returns walk-forward picks and their stitched out-of-sample returns (the
    single best, and an equal-weight blend of the `top` best, which is less
    greedy), the
    pick made on all pre-holdout data with its holdout returns, the deflated
    Sharpe of that pick, PBO over pre-holdout half-years, and the in-sample
    leaderboard.
    """
    blocks, names = half_year_blocks(search.index)
    mom = search.moments(blocks, names, periods_per_year=periods_per_year)
    hold_b = next(i for i, n in enumerate(names) if pd.Timestamp(holdout) <= _block_start(n))
    pre = np.arange(hold_b)
    first = int(np.searchsorted(np.cumsum(mom.n) / periods_per_year, min_train_years - 1e-9) + 1)

    picks = walk_forward_picks(mom, first, hold_b - 1, top)
    wf = [search.returns(int(i[0]))[blocks == b] for b, i in picks.items()]
    wf_top = [pd.concat([search.returns(int(j)) for j in i], axis=1).mean(axis=1)[blocks == b] for b, i in picks.items()]
    wf_table = pd.DataFrame([{"test": names[b], **search.specs.iloc[int(i[0])].drop("mask").to_dict(),
                              "train_sharpe": mom.sharpe(np.arange(b))[int(i[0])],
                              "test_sharpe": mom.sharpe([b])[int(i[0])]} for b, i in picks.items()])

    sr_pre = mom.sharpe(pre)
    best = int(np.argmax(sr_pre))
    best_ret = search.returns(best)
    ex = (best_ret - search.cash)[blocks < hold_b].to_numpy()
    board = search.specs.drop(columns="mask").assign(sharpe_pre=sr_pre, sharpe_holdout=mom.sharpe(np.arange(hold_b, len(names))))
    return {
        "moments": mom, "names": names, "blocks": blocks, "holdout_block": hold_b,
        "wf_picks": wf_table, "wf_returns": pd.concat(wf), "wf_top_returns": pd.concat(wf_top),
        "best": best, "best_returns": best_ret,
        "dsr": deflated_sharpe(ex, sr_pre, len(sr_pre), periods_per_year),
        "pbo": pbo(mom, pre, pbo_splits),
        "board": board.sort_values("sharpe_pre", ascending=False),
    }


def _block_start(name: str) -> pd.Timestamp:
    return pd.Timestamp(f"{name[:4]}-{'01' if name.endswith('H1') else '07'}-01")
