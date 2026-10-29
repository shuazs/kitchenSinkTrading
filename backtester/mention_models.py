"""Models that price Kalshi "mention" markets (will speaker X say word Y).

Every model outputs q = P(Yes) for a market at a fixed decision time t_s,
using only what is known at t_s. The backtest then buys Yes (or No) at the
first *taker print of that side* after t_s when q beats that price by more
than the Kalshi taker fee plus a threshold, and holds to settlement.

Pieces:
- `variants`, `count_hits`: Kalshi strike words ("Airball / Air Ball") ->
  regexes that also match plurals and possessives.
- `snapshots`: one row per market: decision time, last Yes print before it,
  the first taker-Yes and taker-No prints after it, and the result.
- `history_features`: causal per-word / per-series base rates from markets
  that settled before t_s.
- `trades`, `summary`, `bankroll`: the scoring used for every model.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TAKER_FEE = 0.07
MAKER_FEE = 0.0175


def fee(p, rate: float = TAKER_FEE):
    """Kalshi fee per contract at price p (dollars), before rounding up."""
    p = np.asarray(p, dtype=float)
    return rate * p * (1 - p)


def variants(word: str) -> list[str]:
    """Strike text -> lower-case alternatives ("Airball / Air Ball" -> two)."""
    out = []
    for v in re.split(r"\s*/\s*|\s+or\s+", str(word)):
        v = re.sub(r"[\"“”()]", "", v).strip().lower()
        if v:
            out.append(v)
    return out


def word_regex(word: str) -> re.Pattern:
    """Whole-word match of any variant, plus plural / possessive endings."""
    alts = [r"\s+".join(re.escape(t) for t in v.split()) for v in variants(word)]
    return re.compile(r"\b(?:%s)(?:s|es|'s|’s)?\b" % "|".join(alts or ["$^"]), re.I)


def count_hits(word: str, text: str) -> int:
    return len(word_regex(word).findall(text or ""))


def norm_word(word: str) -> str:
    v = variants(word)
    return v[0] if v else ""


def speaker_of(series: str) -> str:
    """Series ticker -> speaker family (KXTRUMPMENTIONB -> TRUMP)."""
    s = re.sub(r"^KX", "", str(series))
    s = re.sub(r"(MENTION|SAY|TWEETS).*$", "", s)
    return s or str(series)


def snapshots(markets: pd.DataFrame, prints: pd.DataFrame, frac: float = 0.5,
              window_h: float = 6.0, hours: float | None = None) -> pd.DataFrame:
    """One row per settled market with the decision time and entry prices.

    The decision time is ``open + frac * (scheduled close - open)`` where the
    scheduled close is the event's median close_time (early Yes closes do not
    move it). Entry prices are the first taker-Yes / taker-No prints within
    ``window_h`` hours after t_s and before the scheduled close.
    """
    m = markets[markets["result"].isin(["yes", "no"])].copy()
    ev = m.groupby("event_ticker").agg(ev_open=("open_time", "min"), ev_close=("close_time", "median"))
    m = m.merge(ev, left_on="event_ticker", right_index=True)
    if hours is None:
        m["t_s"] = m["ev_open"] + frac * (m["ev_close"] - m["ev_open"])
    else:
        m["t_s"] = m["ev_open"] + pd.Timedelta(hours=hours)
    m = m[m["close_time"] > m["t_s"]]  # market still open at t_s
    p = prints[prints["ticker"].isin(m["ticker"])].merge(m[["ticker", "t_s", "ev_close"]], on="ticker")
    p = p.sort_values("created_time")
    before = p[p["created_time"] < p["t_s"]]
    last = before.groupby("ticker").agg(p0=("yes_price", "last"), n_before=("count", "size"),
                                        vol_before=("count", "sum"), t_last=("created_time", "last"))
    after = p[(p["created_time"] >= p["t_s"]) & (p["created_time"] < np.minimum(
        p["t_s"] + pd.Timedelta(hours=window_h), p["ev_close"]))]
    first = after.groupby(["ticker", "taker_side"]).agg(px=("yes_price", "first"), size=("count", "first"),
                                                        t=("created_time", "first")).unstack("taker_side")
    first.columns = [f"{a}_{b}" for a, b in first.columns]
    m = m.merge(last, left_on="ticker", right_index=True, how="inner").merge(
        first, left_on="ticker", right_index=True, how="left")
    m["p0"] = m["p0"] / 100
    m["yes_ask"] = m.get("px_yes") / 100  # a taker-Yes print is at the Yes ask
    m["no_ask"] = 1 - m.get("px_no") / 100  # a taker-No print is at the No ask
    m["y"] = (m["result"] == "yes").astype(int)
    m["word"] = m["yes_sub_title"].fillna("")
    m["nword"] = m["word"].map(norm_word)
    m["speaker"] = m["series"].map(speaker_of)
    m["hours_left"] = (m["ev_close"] - m["t_s"]).dt.total_seconds() / 3600
    keep = ["ticker", "event_ticker", "series", "speaker", "title", "word", "nword", "t_s", "ev_close",
            "hours_left", "p0", "n_before", "vol_before", "yes_ask", "no_ask", "size_yes", "size_no", "y"]
    return m[[c for c in keep if c in m.columns]].reset_index(drop=True)


def history_features(snap: pd.DataFrame, prior: float = 0.45, k: float = 2.0) -> pd.DataFrame:
    """Causal base rates: earlier settled markets (ev_close < t_s) of the same
    word in the same series, same word anywhere, and the series overall.
    Rates are shrunk toward ``prior`` with ``k`` pseudo-counts."""
    s = snap.sort_values("t_s").reset_index(drop=True)
    done = s[["ev_close", "series", "speaker", "nword", "y"]].sort_values("ev_close").reset_index(drop=True)
    out = {}
    for name, keys in [("sw", ["series", "nword"]), ("pw", ["speaker", "nword"]), ("w", ["nword"]),
                       ("ser", ["series"])]:
        n = np.zeros(len(s))
        hits = np.zeros(len(s))
        grp_done = dict(list(done.groupby(keys)))
        for key, sg in s.groupby(keys):
            idx = sg.index
            g = grp_done.get(key)
            if g is None:
                continue
            ct = g["ev_close"].values
            cy = np.concatenate([[0], np.cumsum(g["y"].values)])
            pos = np.searchsorted(ct, s.loc[idx, "t_s"].values, side="left")
            n[idx] = pos
            hits[idx] = cy[pos]
        out[f"n_{name}"] = n
        out[f"r_{name}"] = (hits + k * prior) / (n + k)
    return pd.concat([s, pd.DataFrame(out)], axis=1)


def trades(df: pd.DataFrame, q: pd.Series | np.ndarray, thr: float = 0.0, side: str = "both") -> pd.DataFrame:
    """Entries where q beats the taker price after fee by more than ``thr``.

    q is P(Yes). Yes is bought at ``yes_ask`` (first taker-Yes print after
    t_s), No at ``no_ask``. Net is per contract after the taker fee."""
    d = df.copy()
    d["q"] = np.asarray(q, dtype=float)
    ey = d["q"] - d["yes_ask"] - fee(d["yes_ask"])
    en = (1 - d["q"]) - d["no_ask"] - fee(d["no_ask"])
    rows = []
    if side in ("both", "yes"):
        t = d[ey > thr].assign(side="yes", px=lambda x: x["yes_ask"], edge=ey[ey > thr], size=lambda x: x["size_yes"])
        t["net"] = t["y"] - t["px"] - fee(t["px"])
        rows.append(t)
    if side in ("both", "no"):
        t = d[en > thr].assign(side="no", px=lambda x: x["no_ask"], edge=en[en > thr], size=lambda x: x["size_no"])
        t["net"] = (1 - t["y"]) - t["px"] - fee(t["px"])
        rows.append(t)
    out = pd.concat(rows) if rows else d.iloc[:0]
    if side == "both" and len(out):  # one side per market: the bigger edge
        out = out.sort_values("edge", ascending=False).drop_duplicates("ticker")
    return out.sort_values("t_s")


def event_t(t: pd.DataFrame) -> float:
    """t-stat of the mean net with trades clustered by event."""
    e = t.groupby("event_ticker")["net"].mean()
    return float(e.mean() / e.std() * np.sqrt(len(e))) if len(e) > 2 and e.std() > 0 else np.nan


def summary(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0, "events": 0, "net_c": np.nan, "event_t": np.nan, "hit": np.nan}
    return {"trades": len(t), "events": t["event_ticker"].nunique(), "net_c": round(t["net"].mean() * 100, 2),
            "event_t": round(event_t(t), 2), "hit": round(float((t["net"] > 0).mean()), 3)}


def log_loss(y, q) -> float:
    q = np.clip(np.asarray(q, dtype=float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def bankroll(t: pd.DataFrame, start: float = 5000.0, stake: float = 0.02, max_event: float = 0.10) -> pd.Series:
    """Equity path with ``stake`` of equity per trade, capped by the print's
    size and ``max_event`` of equity per event; each event settles at its
    close (trades of one event are sized off the same equity)."""
    eq = start
    path = {}
    for ev, g in t.sort_values("t_s").groupby("event_ticker", sort=False):
        budget = eq * max_event
        pnl = 0.0
        for _, r in g.iterrows():
            px = r["px"] + fee(r["px"])
            n = min(np.floor(min(eq * stake, budget) / px), r["size"] if pd.notna(r["size"]) else 0)
            if n <= 0:
                continue
            budget -= n * px
            pnl += n * r["net"]
        eq += pnl
        path[g["ev_close"].iloc[0]] = eq
    return pd.Series(path).sort_index()
