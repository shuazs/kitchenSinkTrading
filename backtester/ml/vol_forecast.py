"""Volatility forecasts for position sizing: HAR versus gradient boosting.

Volatility is where machine learning has the best out-of-sample record in
finance, because volatility is persistent and forecastable while returns are
mostly not. Corsi (2009) HAR regresses future realised variance on daily,
weekly and monthly averages of past variance; Christensen, Siggaard and
Veliyev (2023, "A Machine Learning Approach to Volatility Forecasting", J.
Financial Econometrics) find tree ensembles and neural nets beat HAR on Dow
stocks, mainly at longer horizons, while others find a well-fitted HAR hard
to beat. Free daily data has no intraday realised variance, so this uses the
Garman-Klass range estimator from daily OHLC, plus VIX.

The repo's VolTarget sizes UPRO from SPY's trailing 20-day volatility. A
better forecast of the next month's volatility should hold risk closer to
target; whether that adds return is the question the backtest answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import MarketData
from ..strategies.vol_target import VolTarget


def garman_klass(bars: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass variance from open, high, low, close (plus the
    overnight gap), in squared log-return units."""
    o, h, l, c = (np.log(bars[k]) for k in ("open", "high", "low", "close"))
    gap = (o - c.shift()) ** 2
    rs = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return (gap + rs).clip(lower=1e-10)


def vol_features(bars: pd.DataFrame, vix: pd.Series | None = None, vix3m: pd.Series | None = None) -> pd.DataFrame:
    """Log-variance features known at each close."""
    v = garman_klass(bars)
    r = np.log(bars["close"]).diff()
    X = pd.DataFrame(index=bars.index)
    X["lv_1"] = np.log(v)
    X["lv_5"] = np.log(v.rolling(5).mean())
    X["lv_22"] = np.log(v.rolling(22).mean())
    X["lv_66"] = np.log(v.rolling(66).mean())
    X["lr2_20"] = np.log((r**2).rolling(20).mean() + 1e-10)
    X["ret_5"] = r.rolling(5).sum()
    X["ret_22"] = r.rolling(22).sum()
    X["neg_5"] = (r.clip(upper=0) ** 2).rolling(5).mean().pipe(lambda s: np.log(s + 1e-10))
    if vix is not None:
        vx = vix.reindex(bars.index).ffill()
        X["lvix"] = np.log((vx / 100) ** 2 / 252)
        if vix3m is not None:
            X["vix_ratio"] = vx / vix3m.reindex(bars.index).ffill() - 1
    return X


HAR_FEATURES = ["lv_1", "lv_5", "lv_22"]


def future_log_var(bars: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Target: log of the mean squared close-to-close return over the next
    `horizon` days (what a vol-targeted book actually experiences)."""
    r2 = np.log(bars["close"]).diff() ** 2
    return np.log(r2[::-1].rolling(horizon).mean()[::-1].shift(-1) + 1e-10)


def walk_forward_vol(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    first_test: str,
    test_end: str,
    horizon: int = 21,
    features=None,
    seed: int = 0,
) -> pd.Series:
    """Out-of-sample forecasts of annualised volatility, refit each year on
    rows whose target window ended before the year starts.

    model: "trailing" (realised 20-day vol, what VolTarget uses), "har"
    (OLS on HAR_FEATURES), "har_vix" (HAR plus log VIX), "ridge" (all
    features) or "lightgbm" (all features).
    """
    from sklearn.linear_model import LinearRegression, Ridge

    df = X.join(y.rename("y")).replace([np.inf, -np.inf], np.nan)
    cols = features or {"har": HAR_FEATURES, "har_vix": HAR_FEATURES + ["lvix"]}.get(model, list(X.columns))
    df = df.dropna(subset=cols)
    out = []
    for year in range(pd.Timestamp(first_test).year, pd.Timestamp(test_end).year + 1):
        lo = pd.Timestamp(f"{year}-01-01")
        hi = min(pd.Timestamp(test_end), pd.Timestamp(f"{year}-12-31"))
        test = df[(df.index >= lo) & (df.index <= hi)]
        train = df[df.index < lo].iloc[: -(horizon + 1)].dropna(subset=["y"])
        if test.empty:
            continue
        if model == "trailing":
            lv = test["lr2_20"]
        else:
            if model in ("har", "har_vix"):
                m = LinearRegression()
            elif model == "ridge":
                m = Ridge(alpha=1.0)
            elif model == "lightgbm":
                from lightgbm import LGBMRegressor

                m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
            else:
                raise ValueError(model)
            m.fit(train[cols], train["y"])
            fit_resid = train["y"] - m.predict(train[cols])
            # exp(E[log var]) underestimates E[var]; correct with the
            # training residuals (a smearing estimate).
            lv = pd.Series(m.predict(test[cols]), index=test.index) + np.log(np.mean(np.exp(fit_resid)))
        out.append(np.sqrt(np.exp(lv) * 252))
    return pd.concat(out).rename(model)


def qlike(forecast_vol: pd.Series, bars: pd.DataFrame, horizon: int = 21) -> float:
    """Mean QLIKE loss of annualised vol forecasts against the realised
    variance over the next `horizon` days (lower is better; robust to noise
    in the realised proxy, Patton 2011)."""
    real = np.exp(future_log_var(bars, horizon)) * 252
    f = (forecast_vol**2).reindex(real.index)
    ok = f.notna() & real.notna() & (real > 0)
    ratio = real[ok] / f[ok]
    return float((ratio - np.log(ratio) - 1).mean())


class ForecastVolTarget(VolTarget):
    """VolTarget sized from a supplied volatility forecast of `signal`
    (annualised, one value per day known at that close) instead of trailing
    realised volatility."""

    name = "forecast_vol_target"
    params = {**VolTarget.params, "forecast": None}

    def target_weight(self, data: MarketData) -> pd.Series:
        vol = self.p["forecast"].reindex(data.index).ffill()
        raw = self.p["target_vol"] / (vol * self.p["leverage"])
        return raw.clip(upper=self.p["max_weight"])
