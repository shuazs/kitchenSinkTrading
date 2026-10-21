"""Meta-labeling: a secondary model that decides whether to take, skip or
resize each trade a primary strategy proposes (Lopez de Prado 2018, ch. 3;
Joubert 2022, "Meta-Labeling: Theory and Framework", JFDS).

The primary model keeps choosing the side and the exit. The secondary model
only sees the trades the primary would take, with the repo's strategy
signals measured at the entry close, and predicts the chance the trade pays.
That gives it an easier question than forecasting returns every week (the
framing of the signal-combination contest), and it can only cut risk the
primary would have taken, never add a new kind of bet.

    trades = event_table(data, Breakout(entry=20, exit=10), universe)
    probs = walk_forward_probs(trades, make_model, first_test="2019-01-01")
    strategy = MetaFiltered(primary=..., universe=..., mult=decide(probs, ...))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData, Strategy
from .features import asset_features, market_features
from .labels import entries_and_exits, trade_returns, triple_barrier

EVENT_FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252", "macd_8_24", "macd_16_48", "macd_32_96",
    "chan_55", "xs_rank", "vol_20", "vol_ratio", "rsi_2", "season",
    "pension_gap", "days_to_month_end", "vix_ratio", "vix_change", "spy_trend", "curve",
    "brk_strength", "vol_surge", "breadth", "past_hit",
]


def held_mask(primary: Strategy, data: MarketData, universe) -> pd.DataFrame:
    """Where the primary has a position (signal-close dates), on its universe."""
    sub = MarketData({s: data.bars[s] for s in universe}, data.extra)
    w = primary.generate_targets(sub).reindex(data.index).ffill()
    return w.fillna(0.0) > 0


def event_table(
    data: MarketData,
    primary: Strategy,
    universe,
    cost: float = 0.0,
    min_ret: float = 0.0,
    tb: dict | None = None,
    curve: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per primary trade: entry, exit, label columns and features at entry.

    cost: round-trip cost subtracted from the trade return before labelling.
    tb: triple-barrier settings (pt, sl, horizon) for the alternative label.
    curve: optional yield-curve series for the `curve` feature (else
        data.extra['curve']).
    """
    universe = [s for s in universe if s in data.bars]
    held = held_mask(primary, data, universe)
    trades = entries_and_exits(held)
    opens = data.field("open")[universe]
    close = data.field("close")[universe].ffill()
    trades["ret"] = trade_returns(trades, opens, cost)
    trades["label"] = (trades["ret"] > min_ret).astype(float).where(trades["ret"].notna())

    per_year = 365 if (data.index.dayofweek >= 5).mean() > 0.1 else 252
    dvol = close.pct_change(fill_method=None).rolling(max(20, round(20 * per_year / 252))).std()
    trades = trades.join(triple_barrier(trades, close, dvol, **(tb or {})))
    trades["end"] = trades[["exit", "tb_end"]].max(axis=1)

    feats = asset_features(data, universe)
    entry_hi = close.rolling(getattr(primary, "p", {}).get("entry", 20)).max().shift(1)
    feats["brk_strength"] = (close / entry_hi - 1) / dvol
    if "volume" in next(iter(data.bars.values())):
        vol = data.field("volume")[universe].replace(0, np.nan)
        feats["vol_surge"] = np.log(vol.rolling(5).mean() / vol.rolling(60).mean())
    feats["breadth"] = pd.DataFrame(
        np.repeat(held.mean(axis=1).to_numpy()[:, None], len(universe), axis=1), index=held.index, columns=universe
    )
    for name, f in feats.items():
        vals = f.reindex(data.index).ffill(limit=3)
        trades[name] = [vals.at[d, s] if s in vals else np.nan for d, s in zip(trades["entry"], trades["symbol"])]
    trades["past_hit"] = _past_hit_rate(trades)
    if curve is not None:
        data = MarketData(data.bars, {**data.extra, "curve": curve})
    mkt = market_features(data, data.index)
    trades = trades.join(mkt, on="entry")
    trades["season"] = trades["season"].fillna(0.0)
    for c in EVENT_FEATURES:
        if c not in trades:
            trades[c] = 0.0
    return trades.replace([np.inf, -np.inf], np.nan)


def _past_hit_rate(trades: pd.DataFrame, n: int = 20) -> pd.Series:
    """Win rate of the last `n` closed trades in the whole book at each entry
    (a "strategy is working" feature; uses only trades already exited)."""
    done = trades.dropna(subset=["ret"]).sort_values("exit")
    exits, wins = done["exit"].to_numpy(), (done["ret"] > 0).to_numpy(float)
    out = np.full(len(trades), np.nan)
    for k, e in enumerate(trades["entry"].to_numpy()):
        m = np.searchsorted(exits, e, side="left")
        if m >= 5:
            out[k] = wins[max(0, m - n) : m].mean()
    return pd.Series(out, index=trades.index)


class Logistic:
    """L2 logistic regression on z-scored features (the linear baseline)."""

    name = "logistic"

    def __init__(self, features=EVENT_FEATURES, C: float = 0.1):
        self.features, self.C = list(features), C

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Logistic":
        from sklearn.linear_model import LogisticRegression

        self.mu, self.sd = X[self.features].mean(), X[self.features].std().replace(0, 1).fillna(1)
        self.m = LogisticRegression(C=self.C, max_iter=1000).fit(self._z(X), y)
        return self

    def _z(self, X):
        return ((X[self.features] - self.mu) / self.sd).clip(-5, 5).fillna(0.0).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict_proba(self._z(X))[:, 1]


class GBM:
    """Small, heavily regularised LightGBM classifier."""

    name = "lightgbm"

    def __init__(self, features=EVENT_FEATURES, seed: int = 0, **kw):
        self.features = list(features)
        self.kw = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                       subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                       random_state=seed, verbose=-1, **kw)

    def fit(self, X, y):
        from lightgbm import LGBMClassifier

        self.m = LGBMClassifier(**self.kw).fit(X[self.features], y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X[self.features])[:, 1]


def walk_forward_probs(
    trades: pd.DataFrame,
    make_model,
    first_test: str,
    test_end: str,
    label: str = "label",
    purge_days: int = 5,
    min_train: int = 150,
) -> pd.DataFrame:
    """Out-of-sample probability for every trade entered from `first_test`
    to `test_end`, refitting each calendar year on trades whose labels ended
    at least `purge_days` before that year starts (expanding window).

    Adds `p` (test probability) and `p_cut` (the training predictions'
    quantiles, for thresholds set without looking at the test year).
    """
    out = []
    start, end = pd.Timestamp(first_test), pd.Timestamp(test_end)
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, pd.Timestamp(f"{year}-01-01")), min(end, pd.Timestamp(f"{year}-12-31"))
        test = trades[(trades["entry"] >= lo) & (trades["entry"] <= hi)]
        cutoff = lo - pd.Timedelta(days=purge_days)
        train = trades[(trades["end"] < cutoff) & trades[label].notna()]
        if test.empty or len(train) < min_train or train[label].nunique() < 2:
            continue
        m = make_model().fit(train, train[label])
        p_train = m.predict_proba(train)
        t = test.copy()
        t["p"] = m.predict_proba(test)
        t["base_rate"] = train[label].mean()
        for q in (0.25, 0.33, 0.5):
            t[f"p_q{int(q * 100)}"] = np.quantile(p_train, q)
        t["p_median"] = np.median(p_train)
        out.append(t)
    return pd.concat(out) if out else trades.iloc[:0]


def decide(probs: pd.DataFrame, rule: str) -> pd.Series:
    """Per-trade size multiplier from out-of-sample probabilities.

    "all": take every trade (the primary alone, on the same dates).
    "skip_q25" / "skip_q33" / "skip_q50": skip trades below that quantile of
        the training predictions.
    "size": p over the training median, clipped to [0, 2].
    """
    if rule == "all":
        return pd.Series(1.0, index=probs.index)
    if rule.startswith("skip_q"):
        return (probs["p"] >= probs[f"p_q{rule[6:]}"]).astype(float)
    if rule == "size":
        return (probs["p"] / probs["p_median"]).clip(0, 2)
    raise ValueError(rule)


class MetaFiltered(Strategy):
    """The primary strategy with each trade scaled by a meta-label multiplier.

    primary: the Strategy whose trades `trades` lists (same params).
    trades: rows from event_table (entry, exit, symbol) with a `mult` column;
        trades not listed (outside the test years) get `default`.
    Each open position gets mult / (number of open positions with mult > 0),
    so a pure filter (mult 0 or 1) is the primary run on the kept trades
    only. With `slots`, each position gets mult / slots instead, so weights
    change only on entries and exits (equal splitting rebalances every open
    position whenever one opens or closes). The book's gross is capped at
    max_gross.
    """

    name = "meta_filtered"
    params = {"primary": None, "universe": None, "trades": None, "default": 0.0, "slots": None, "max_gross": 1.0}

    def generate_targets(self, data: MarketData) -> pd.DataFrame:
        uni = [s for s in self.p["universe"] if s in data.bars]
        held = held_mask(self.p["primary"], data, uni)
        mult = pd.DataFrame(np.where(held, self.p["default"], 0.0), index=held.index, columns=uni)
        idx = held.index
        for t in self.p["trades"].itertuples():
            i = idx.get_loc(t.entry)
            j = len(idx) if getattr(t, "open", False) else idx.get_loc(t.exit)
            mult.iloc[i:j, uni.index(t.symbol)] = t.mult
        n = (mult > 0).sum(axis=1).clip(lower=1) if self.p["slots"] is None else self.p["slots"]
        w = mult.div(n, axis=0)
        gross = w.sum(axis=1)
        w = w.div((gross / self.p["max_gross"]).clip(lower=1.0), axis=0)
        return w.reindex(columns=data.symbols, fill_value=0.0)


__all__ = ["EVENT_FEATURES", "GBM", "Logistic", "MetaFiltered", "decide", "event_table", "held_mask", "walk_forward_probs"]
