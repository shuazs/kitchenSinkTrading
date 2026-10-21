"""Kalshi crowd bias after big moves in the underlying.

Kalshi pushes app notifications when BTC, ETH or the indexes trend hard. With no
record of when those go out, a "hot" moment is proxied by a big 15-minute to
1-day move in the underlying (z vs the trailing 30-day spread of such moves) or
by a spike in Kalshi contracts traded. Three questions:

* calibration: after a trigger, is the continuation side of a ladder contract
  priced above or below its realized hit rate (vs normal times)?
* lean: do takers buy the continuation side more than usual, and does the
  underlying continue or reverse afterwards?
* trading: buy the continuation (follow) or reversal (fade) side, hold to
  settlement, with Kalshi fees, the ask/bid rebuilt from prints and fills capped
  at printed size.

Inputs are Kalshi prints with columns ticker, count, yes_price (cents),
taker_side, created_time, event_ticker, K, close_time, result, tau (minutes to
close), and a minute price series of the underlying indexed by the time each
price is known.
"""
import math

import numpy as np
import pandas as pd

WINDOWS = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
FEE_MULT = {"KXINXU": 0.035, "KXNASDAQ100U": 0.035, "KXBTCD": 0.07, "KXETHD": 0.07}
BANDS = {"longshot": (0.05, 0.25), "mid": (0.35, 0.65), "favorite": (0.75, 0.95)}
PRICE_BUCKETS = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001]


def kalshi_fee(mult, contracts, price):
    """Kalshi trading fee in dollars, rounded up to the cent."""
    return math.ceil(100 * mult * contracts * price * (1 - price) - 1e-9) / 100


def move_z(s, windows=WINDOWS, lookback="30D"):
    """z of the W-minute log move known at each minute, vs the trailing std of W-moves (no lookahead)."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    base = s.to_frame("S")
    out = pd.DataFrame(index=s.index)
    for name, w in windows.items():
        prev = pd.merge_asof(pd.DataFrame({"t": s.index - pd.Timedelta(minutes=w)}), base, left_on="t",
                             right_index=True, direction="backward", tolerance=pd.Timedelta("4D"))["S"].values
        r = pd.Series(np.log(s.values / prev), s.index)
        out[name] = r / r.rolling(lookback, min_periods=2000).std().shift(1)
        out["r" + name] = r
    return out


def move_triggers(z, window, threshold=2.5, cool="6h"):
    """First minute with |z| >= threshold after a quiet spell; value is the move's direction (+1/-1)."""
    hit = z.index[(z[window].abs() >= threshold).values]
    if not len(hit):
        return pd.Series(dtype=float)
    hit = hit[np.r_[True, np.diff(hit.values) > pd.Timedelta(cool).to_timedelta64()]]
    return np.sign(z.loc[hit, window])


def volume_ratio(prints, freq="15min", lookback="7D"):
    """Contracts traded per bucket over the trailing mean, indexed by bucket end (when it is known)."""
    v = prints.groupby(prints.created_time.dt.floor(freq))["count"].sum()
    v = v.reindex(pd.date_range(v.index.min(), v.index.max(), freq=freq)).fillna(0)
    r = v / v.rolling(lookback, min_periods=96).mean().shift(1)
    r.index = r.index + pd.Timedelta(freq)
    return r


def spike_triggers(prints, z, ratio=3.0, cool="6h"):
    """Kalshi volume spikes; direction is the sign of the underlying's last-hour move."""
    vr = volume_ratio(prints)
    hit = vr.index[(vr >= ratio).values]
    hit = hit[np.r_[True, np.diff(hit.values) > pd.Timedelta(cool).to_timedelta64()]] if len(hit) else hit
    r1 = pd.merge_asof(pd.DataFrame({"t": hit}), z[["r1h"]], left_on="t", right_index=True, direction="backward")["r1h"]
    d = pd.Series(np.sign(r1.values), hit)
    return d[d != 0].dropna()


def label_prints(prints, triggers, z, hold="60min"):
    """Add hot (print within `hold` after a trigger), d (trigger direction, else last-hour move sign),
    q (price of the continuation side), win (continuation side paid) and tk_cont (taker bought it)."""
    x = prints.sort_values("created_time").reset_index(drop=True)
    tr = triggers.rename("d").to_frame()
    tr["t_trig"] = tr.index
    y = pd.merge_asof(x[["created_time"]], tr, left_on="created_time", right_index=True, direction="backward")
    age = x.created_time - y.t_trig
    hot = (age >= pd.Timedelta(0)) & (age < pd.Timedelta(hold))
    base = pd.merge_asof(x[["created_time"]], z[["r1h"]], left_on="created_time", right_index=True, direction="backward")["r1h"]
    x["hot"] = hot.values
    x["d"] = np.where(hot, y.d, np.sign(base))
    x = x[x.d.notna() & (x.d != 0)].copy()
    up = x.d > 0
    p, yes = x.yes_price / 100, (x.result == "yes").astype(float)
    x["q"] = np.where(up, p, 1 - p)
    x["win"] = np.where(up, yes, 1 - yes)
    x["tk_cont"] = ((x.taker_side == "yes") == up).astype(float)
    return x


def clustered_mean(v, w, cluster):
    """Weighted mean, its cluster-robust standard error, and the number of clusters."""
    g = pd.DataFrame({"a": v * w, "w": w, "c": cluster}).groupby("c").sum()
    m = g.a.sum() / g.w.sum()
    return m, np.sqrt(((g.a - m * g.w) ** 2).sum()) / g.w.sum(), len(g)


def calibration(x):
    """Continuation-side excess (hit rate minus price, cents) and taker lean, hot vs normal, by price bucket."""
    rows = []
    for hot, g0 in x.groupby("hot"):
        parts = [(str(b), g) for b, g in g0.groupby(pd.cut(g0.q, PRICE_BUCKETS, right=False), observed=True)] + [("ALL", g0)]
        for b, g in parts:
            if len(g) < 50:
                continue
            w, dd = g["count"].values.astype(float), g.created_time.dt.floor("D").values
            ex, se, nd = clustered_mean((g.win - g.q).values, w, dd)
            lean, lse, _ = clustered_mean(g.tk_cont.values, w, dd)
            rows.append(dict(hot=hot, q=b, prints=len(g), days=nd, contracts=int(w.sum()), excess_c=100 * ex,
                             se_c=100 * se, taker_cont=lean, lean_se=lse))
    return pd.DataFrame(rows)


class Ladder:
    """Prints grouped by contract and event for fast lookups."""

    def __init__(self, prints):
        self.by_ticker = {k: (g.created_time.values, g.yes_price.values / 100, g.taker_side.values == "yes",
                              g["count"].values, g.result.iat[0]) for k, g in prints.groupby("ticker")}
        self.events = prints.groupby("event_ticker").agg(close=("close_time", "max"), opn=("created_time", "min")).sort_values("close")
        self.tickers = prints.groupby("event_ticker").ticker.unique()


def simulate(book, t0, d, side, band, execution, fee_mult, risk=250.0, lat="1min", window="30min",
             min_tau=15, max_tau=240, chase=0.05, maker_share=0.5):
    """One trade after a trigger at t0 with direction d. side: 'follow' buys the continuation side, 'fade'
    the reversal side. Picks the nearest-expiring event, the contract whose side price (last print) is
    closest to the band centre, and fills from later prints:

    * taker: prints where a taker bought our side (the ask) at <= last price + chase;
    * maker: a resting bid at the last price, filled by takers selling our side at <= it, `maker_share`
      of each print's size (queue unknown, so this is optimistic), maker fee = taker fee / 4.

    Fills stop at the printed size and at `risk` dollars. Held to settlement."""
    te = t0 + pd.Timedelta(lat)
    ev = book.events
    ev = ev[(ev.close >= te + pd.Timedelta(minutes=min_tau)) & (ev.close <= te + pd.Timedelta(minutes=max_tau)) & (ev.opn < te)]
    if ev.empty:
        return None
    e = ev.index[0]
    buy_yes = (d > 0) == (side == "follow")
    lo, hi = BANDS[band]
    centre = (lo + hi) / 2
    best = None
    for tk in book.tickers[e]:
        ct, py = book.by_ticker[tk][:2]
        i = np.searchsorted(ct, te.to_datetime64())
        if i == 0:
            continue
        ref = py[i - 1] if buy_yes else 1 - py[i - 1]
        if lo <= ref <= hi and (best is None or abs(ref - centre) < abs(best[1] - centre)):
            best = (tk, ref)
    if best is None:
        return None
    tk, ref = best
    ct, py, ty, cnt, res = book.by_ticker[tk]
    i0, i1 = np.searchsorted(ct, te.to_datetime64()), np.searchsorted(ct, (te + pd.Timedelta(window)).to_datetime64())
    px = py[i0:i1] if buy_yes else 1 - py[i0:i1]
    took_ours = ty[i0:i1] == buy_yes
    size = cnt[i0:i1].astype(float)
    if execution == "taker":
        ok = took_ours & (px <= ref + chase)
    else:
        ok = ~took_ours & (px <= ref)
        size, px = np.floor(size * maker_share), np.full(len(px), ref)
    want, got, cost = int(risk / max(ref, 0.01)), 0, 0.0
    for p, c in zip(px[ok], size[ok]):
        q = int(min(c, want - got))
        got, cost = got + q, cost + q * p
        if got >= want:
            break
    if got == 0:
        return dict(t=te, ticker=tk, ref=ref, qty=0)
    avg = cost / got
    win = float((res == "yes") == buy_yes)
    fee = kalshi_fee(fee_mult / 4 if execution == "maker" else fee_mult, got, avg)
    return dict(t=te, ticker=tk, close=book.events.close[e], ref=ref, qty=got, px=avg, win=win, fee=fee,
                pnl=got * (win - avg) - fee, cost=cost + fee)


def cap_capital(trades, capital=5000.0):
    """Drop trades that would take open cost (until settlement) above `capital`."""
    trades = trades.sort_values("t")
    keep, open_ = [], []
    for z in trades.itertuples():
        open_ = [(c, k) for c, k in open_ if c > z.t]
        ok = sum(k for _, k in open_) + z.cost <= capital
        keep.append(ok)
        if ok:
            open_.append((z.close, z.cost))
    return trades[keep]


def summarize(trades, capital=5000.0):
    if trades.empty:
        return dict(filled=0)
    mid = trades.t.iloc[len(trades) // 2]
    pnl = trades.pnl
    return dict(filled=len(trades), contracts=int(trades.qty.sum()), avg_px=trades.px.mean(),
                edge_c=100 * ((trades.win - trades.px) * trades.qty).sum() / trades.qty.sum(), fees=trades.fee.sum(),
                pnl=pnl.sum(), t=pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if len(pnl) > 2 and pnl.std() > 0 else np.nan,
                first_half=trades[trades.t < mid].pnl.sum(), second_half=trades[trades.t >= mid].pnl.sum(),
                return_pct=100 * pnl.sum() / capital)
