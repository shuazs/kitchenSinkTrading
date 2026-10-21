"""Short-horizon (hourly to end-of-day) signal combination for SPY/QQQ and
0DTE option trades.

Pipeline:

1. `hourly_panel` turns hourly bars (yfinance, ~730 days) plus optional
   event series (Kalshi implied closes, Polymarket theme indexes) into one
   row per decision time: the end of each regular-session hourly bar from
   10:30 to 15:30 New York time. Every feature uses only data known at that
   time. Targets are the next-hour return (`y_1h`) and the return to the
   16:00 close (`y_close`), the horizon a 0DTE option settles on.
2. `walk_forward` refits a model every month on all earlier rows, with a
   one-day embargo so no training target overlaps a test row.
3. `zero_dte` buys an at-the-money call or put at the decision time when
   the out-of-sample score clears a threshold set on training data, prices
   it with Black-Scholes at an implied vol tied to VIX9D, and holds it to
   the close.

The option model is the weak link: there is no free 0DTE option history,
so prices are modelled, not observed. `iv_mult` scales VIX9D to the 0DTE
implied vol. `fair_iv_mult` finds the scale at which buying straddles
breaks even, so a run at that scale isolates the signal, and runs at 10-20%
above it model the variance premium that option buyers normally pay.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

NY = "America/New_York"
DECISION_HOURS = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30")
MINUTES_PER_YEAR = 390 * 252


# --------------------------------------------------------------------------- data shaping

def bar_end_closes(bars: pd.DataFrame) -> pd.Series:
    """Hourly closes of yfinance RTH bars re-indexed by bar END time (New York).

    yfinance labels hourly stock bars by their start (09:30, ..., 15:30), and
    the 15:30 bar ends at the 16:00 close.
    """
    col = "Close" if "Close" in bars else "close"
    c = bars[col].dropna()
    idx = c.index.tz_convert(NY) if c.index.tz is not None else c.index.tz_localize(NY)
    end = idx + pd.Timedelta("1h")
    end = end.where(~((idx.hour == 15) & (idx.minute == 30)), idx.normalize() + pd.Timedelta("16h"))
    s = pd.Series(c.to_numpy(), index=end)
    return s[~s.index.duplicated(keep="last")].sort_index()


def asof(series: pd.Series, times: pd.DatetimeIndex, max_age: str | None = None) -> pd.Series:
    """Last value of `series` at or before each time (both tz-aware)."""
    s = series.dropna().sort_index()
    s.index = s.index.tz_convert(NY)
    pos = s.index.searchsorted(times, side="right") - 1
    vals = np.where((pos >= 0) & ~times.isna(), s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    if max_age is not None:
        age = times - s.index[np.clip(pos, 0, None)]
        vals = np.where((pos >= 0) & (age <= pd.Timedelta(max_age)), vals, np.nan)
    return pd.Series(vals, index=times)


def _session(c: pd.Series) -> pd.DataFrame:
    """Day x bar-end frame of RTH closes (columns 10:30 ... 16:00)."""
    df = pd.DataFrame({"c": c, "day": c.index.normalize(), "t": c.index.strftime("%H:%M")})
    df = df[df.t.isin(list(DECISION_HOURS) + ["16:00"])]
    return df.pivot_table(index="day", columns="t", values="c", aggfunc="last")


def hourly_panel(target: pd.DataFrame, others: dict[str, pd.DataFrame] | None = None,
                 vix: pd.DataFrame | None = None, vix3m: pd.DataFrame | None = None,
                 vix9d: pd.DataFrame | None = None, around_clock: dict[str, pd.DataFrame] | None = None,
                 events: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """One row per (day, decision time) for the `target` symbol's hourly bars.

    others: RTH symbols whose return since yesterday's close enters as a
        feature (raw for bonds/dollar/gold, relative to the target for equity
        sectors whose key ends in "_rel").
    around_clock: 24h markets (BTC, futures) whose last-hour and
        since-yesterday's-16:00 returns enter.
    events: series already indexed by the time each value became known
        (UTC or NY), e.g. a Kalshi implied-close gap or a Polymarket index
        change, taken as of each decision time.
    """
    c = bar_end_closes(target)
    o = target["Open" if "Open" in target else "open"]
    o.index = o.index.tz_convert(NY)
    first_open = o[(o.index.hour == 9) & (o.index.minute == 30)]
    first_open.index = first_open.index.normalize()

    S = _session(c)
    prev_close = S["16:00"].shift(1)
    rows = []
    for h in DECISION_HOURS:
        hh, mm = map(int, h.split(":"))
        t = S.index + pd.Timedelta(hours=hh, minutes=mm)
        p = S[h]
        prev_h = S[DECISION_HOURS[DECISION_HOURS.index(h) - 1]] if h != "10:30" else first_open.reindex(S.index)
        d = pd.DataFrame({
            "time": t,
            "hour": hh + mm / 60,
            "minutes_left": (16 * 60) - (hh * 60 + mm),
            "price": p.to_numpy(),
            "day_ret": np.log(p / prev_close).to_numpy(),
            "open_ret": np.log(p / first_open.reindex(S.index)).to_numpy(),
            "gap": np.log(first_open.reindex(S.index) / prev_close).to_numpy(),
            "last_hr": np.log(p / prev_h).to_numpy(),
            "y_close": np.log(S["16:00"] / p).to_numpy(),
            "close": S["16:00"].to_numpy(),
        })
        nxt = DECISION_HOURS[DECISION_HOURS.index(h) + 1] if h != "15:30" else "16:00"
        d["y_1h"] = np.log(S[nxt] / p).to_numpy()
        rows.append(d)
    P = pd.concat(rows).dropna(subset=["price"]).set_index("time").sort_index()
    t = P.index
    day = t.normalize()
    prev_day_close_time = pd.DatetimeIndex(pd.Series(day).map(dict(zip(S.index, S.index.to_series().shift(1)))).to_numpy())
    prev_close_time = pd.DatetimeIndex(prev_day_close_time).tz_localize(None).tz_localize(NY) + pd.Timedelta("16h") \
        if prev_day_close_time.tz is None else prev_day_close_time + pd.Timedelta("16h")

    def since_prev_close(series: pd.Series) -> pd.Series:
        now = asof(series, t)
        then = asof(series, prev_close_time)
        return pd.Series(np.log(now.to_numpy() / then.to_numpy()), index=t)

    # realised vol of the target's hourly returns over the prior 20 sessions, a scale for targets
    hr = np.log(c).diff()
    hr = hr[hr.index.strftime("%H:%M").isin(list(DECISION_HOURS[1:]) + ["16:00"])]
    rv = hr.pow(2).groupby(hr.index.normalize()).sum().pipe(np.sqrt).rolling(20, min_periods=10).mean().shift(1)
    P["rv20"] = rv.reindex(day).to_numpy()

    for name, bars in (others or {}).items():
        r = since_prev_close(bar_end_closes(bars))
        P[name] = (r - P["day_ret"]).to_numpy() if name.endswith("_rel") else r.to_numpy()

    for name, bars in (around_clock or {}).items():
        col = "Close" if "Close" in bars else "close"
        s = bars[col].dropna()
        s.index = s.index.tz_convert(NY) + pd.Timedelta("1h")  # label by bar end
        now = asof(s, t, "2h")
        P[f"{name}_hr"] = np.log(now.to_numpy() / asof(s, t - pd.Timedelta("1h"), "2h").to_numpy())
        P[f"{name}_day"] = np.log(now.to_numpy() / asof(s, prev_close_time, "2h").to_numpy())

    if vix is not None:
        v = bar_end_closes(vix)
        vn = asof(v, t)
        P["vix"] = vn.to_numpy()
        P["vix_day"] = since_prev_close(v).to_numpy()
        P["vix_hr"] = np.log(vn.to_numpy() / asof(v, t - pd.Timedelta("1h")).to_numpy())
        if vix3m is not None:
            P["vix_ts"] = np.log(vn / asof(bar_end_closes(vix3m), t)).to_numpy()
        if vix9d is not None:
            P["vix9_ts"] = np.log(asof(bar_end_closes(vix9d), t) / vn).to_numpy()
    if vix9d is not None:
        P["vix9d"] = asof(bar_end_closes(vix9d), t).to_numpy()  # sets the option's implied vol

    for name, s in (events or {}).items():
        P[name] = asof(s, t, "75min").to_numpy()

    P["dow"] = t.dayofweek
    return P


# --------------------------------------------------------------------------- models

def zscore_combo(X: pd.DataFrame, y: pd.Series):
    """Equal-weight combination: each feature z-scored on training data and
    signed by its training correlation with the target, then averaged."""
    mu, sd = X.mean(), X.std().replace(0, np.nan)
    Z = (X - mu) / sd
    sign = np.sign(Z.corrwith(y)).fillna(0)

    def predict(Xn: pd.DataFrame) -> np.ndarray:
        return (((Xn - mu) / sd).fillna(0) * sign).mean(axis=1).to_numpy()
    return predict


def ridge(alpha: float = 50.0):
    def fit(X: pd.DataFrame, y: pd.Series):
        mu, sd = X.mean(), X.std().replace(0, 1.0)
        Z = ((X - mu) / sd).fillna(0).to_numpy()
        w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y.to_numpy() - y.mean()))
        return lambda Xn: ((Xn - mu) / sd).fillna(0).to_numpy() @ w + y.mean()
    return fit


def lightgbm(**params):
    def fit(X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb
        p = dict(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=100,
                 subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        p.update(params)
        m = lgb.LGBMRegressor(**p).fit(X, y)
        return lambda Xn: m.predict(Xn)
    return fit


def walk_forward(P: pd.DataFrame, features: list[str], target: str, fit,
                 start: str, end: str | None = None, embargo_days: int = 1,
                 min_train: int = 500) -> pd.DataFrame:
    """Expanding-window monthly refit over test months in [start, end).

    Returns the test rows with `pred` plus `train_q50/q80/q90`: quantiles of
    |pred| on the training rows of the same decision hour, used as
    confidence thresholds that never look at test data.
    """
    P = P.dropna(subset=[target])
    days = P.index.normalize().tz_localize(None)
    stop = pd.Timestamp(end) if end is not None else days.max() + pd.Timedelta("1D")
    months = pd.date_range(pd.Timestamp(start), stop, freq="MS")
    edges = list(months) + ([stop] if months.empty or months[-1] < stop else [])
    out = []
    for m0, m1 in zip(edges[:-1], edges[1:]):
        test = P[(days >= m0) & (days < m1)]
        train = P[days < m0 - pd.tseries.offsets.BDay(embargo_days)]
        if len(test) == 0 or len(train) < min_train:
            continue
        predict = fit(train[features], train[target])
        tr = pd.DataFrame({"a": np.abs(np.asarray(predict(train[features]))),
                           "h": train["hour"].to_numpy() if "hour" in train else 0})
        t = test.copy()
        t["pred"] = predict(test[features])
        h = t["hour"].to_numpy() if "hour" in t else np.zeros(len(t))
        for q in (50, 80, 90):
            qs = tr.groupby("h").a.quantile(q / 100)
            t[f"train_q{q}"] = pd.Series(h).map(qs).to_numpy()
        out.append(t)
    return pd.concat(out) if out else pd.DataFrame()


def score(oos: pd.DataFrame, target: str) -> dict:
    """Hit rate overall and on the most confident rows, plus rank IC."""
    y, p = oos[target], oos["pred"]
    ok = y != 0
    res = {"n": int(ok.sum()), "hit": float((np.sign(p[ok]) == np.sign(y[ok])).mean()),
           "ic": float(p.corr(y, method="spearman"))}
    for q in ("q80", "q90"):
        sel = ok & (p.abs() >= oos[f"train_{q}"])
        res[f"n_{q}"] = int(sel.sum())
        res[f"hit_{q}"] = float((np.sign(p[sel]) == np.sign(y[sel])).mean()) if sel.any() else np.nan
    res["ret_bps"] = float((np.sign(p) * y).mean() * 1e4)
    return res


# --------------------------------------------------------------------------- 0DTE options

def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_price(S: float, K: float, minutes: float, vol: float, call: bool) -> float:
    """Black-Scholes price, zero rates, time in trading minutes."""
    if minutes <= 0 or vol <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    sd = vol * sqrt(minutes / MINUTES_PER_YEAR)
    d1 = log(S / K) / sd + sd / 2
    d2 = d1 - sd
    return S * _ncdf(d1) - K * _ncdf(d2) if call else K * _ncdf(-d2) - S * _ncdf(-d1)


def fair_iv_mult(P: pd.DataFrame, lo: float = 0.2, hi: float = 1.5) -> float:
    """Scale on VIX9D at which modelled at-the-money straddles bought at the
    rows' decision times break even against their realised payoff at the
    close. VIX9D includes overnight and weekend variance, so the fair scale
    for the rest of one session is well under 1 (about 0.5-0.7 for
    SPY/QQQ in 2024-26). Price trades at this times (1 + premium) to model a
    variance risk premium.
    """
    P = P.dropna(subset=["vix9d", "close", "price"])
    K = P.price.round().to_numpy()
    pay = np.abs(P.close.to_numpy() - K).mean()
    args = list(zip(P.price, K, P.minutes_left, P.vix9d))

    def cost(m: float) -> float:
        return np.mean([bs_price(S, k, mn, m * v / 100, True) + bs_price(S, k, mn, m * v / 100, False)
                        for S, k, mn, v in args])
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if cost(mid) > pay else (mid, hi)
    return (lo + hi) / 2


@dataclass
class OptionCosts:
    half_spread: float = 0.01   # dollars per share per side (SPY/QQQ 0DTE ATM quotes are usually a penny or two wide)
    commission: float = 0.65    # per contract per side; worthless expiries pay no closing fee


def zero_dte(oos: pd.DataFrame, iv_mult: float, threshold: str | None = "train_q80", hours=("10:30",),
             risk_frac: float = 0.10, capital: float = 5_000.0, costs: OptionCosts | None = None, mode: str = "signal", strike_step: float = 1.0) -> pd.DataFrame:
    """At most one 0DTE trade a day, held to the 16:00 close.

    mode "signal": buy a call if pred > 0 else a put, when |pred| clears the
        `threshold` column (None trades every day).
    mode "against": the same trades on the opposite side. Signal minus
        against isolates directional skill from the volatility pricing.
    mode "straddle" / "call" / "put": no-signal baselines.
    iv_mult scales VIX9D to the option's implied vol; see `fair_iv_mult`.
    Premium per trade is `risk_frac` of current equity, whole contracts only.
    The SPY/QQQ close stands in for the option's settlement (index-level
    0DTE on SPX/NDX settles the same way, at 10x the notional).
    """
    costs = costs or OptionCosts()
    want = {(int(h.split(":")[0]), int(h.split(":")[1])) for h in hours}
    rows = oos[[(t.hour, t.minute) in want for t in oos.index]]
    eq = capital
    out = []
    for day, g in rows.groupby(rows.index.normalize()):
        r = None
        for t, row in g.iterrows():
            if mode not in ("signal", "against") or threshold is None or abs(row.pred) >= row[threshold]:
                r = row; time = t
                break
        if r is None or not np.isfinite(r.get("vix9d", np.nan)) or not np.isfinite(r.close):
            continue
        S, close, mins = r.price, r.close, r.minutes_left
        K = round(S / strike_step) * strike_step
        vol = iv_mult * r.vix9d / 100
        legs = {"signal": [r.pred > 0], "against": [r.pred <= 0], "call": [True], "put": [False], "straddle": [True, False]}[mode]
        prem = sum(bs_price(S, K, mins, vol, c) for c in legs) + costs.half_spread * len(legs)
        payoff = sum(max(close - K, 0) if c else max(K - close, 0) for c in legs)
        payoff = max(payoff - costs.half_spread * len(legs) * (payoff > 0), 0)
        n = int(risk_frac * eq // (prem * 100))
        if n < 1:
            continue
        fees = costs.commission * n * len(legs) * (1 + (payoff > 0))
        pnl = n * 100 * (payoff - prem) - fees
        eq += pnl
        out.append({"time": time, "side": "call" if legs == [True] else "put" if legs == [False] else "straddle",
                    "contracts": n, "premium": prem, "payoff": payoff, "pnl": pnl, "equity": eq,
                    "ret_on_premium": pnl / (n * 100 * prem)})
    return pd.DataFrame(out).set_index("time") if out else pd.DataFrame(
        columns=["side", "contracts", "premium", "payoff", "pnl", "equity", "ret_on_premium"])


def option_summary(trades: pd.DataFrame, capital: float = 5_000.0) -> dict:
    if trades.empty:
        return {"trades": 0}
    eq = pd.concat([pd.Series([capital]), trades.equity.reset_index(drop=True)])
    return {"trades": len(trades), "win_rate": float((trades.pnl > 0).mean()),
            "avg_ret_on_premium": float(trades.ret_on_premium.mean()),
            "final_equity": float(trades.equity.iloc[-1]),
            "max_dd": float((eq / eq.cummax() - 1).min())}
