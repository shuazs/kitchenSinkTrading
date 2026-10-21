"""Combine short-horizon signals for SPY/QQQ and test them as 0DTE option trades.

    python examples/run_shortterm_combo.py

Decision points are the end of each regular-session hourly bar (10:30 to
15:30 New York). Signals: intraday momentum (since yesterday's close, since
the open, last hour, overnight gap), cross-asset moves since yesterday's
close (TLT, HYG, UUP, GLD, USO, IWM/SMH/QQQ relative), BTC and 10y futures
over the last hour and since 16:00, VIX level, change and term structure,
and the target's recent realised vol. Optional event inputs (Kalshi implied
close, Polymarket theme indexes) are read from the project's shared folder
when it exists; see `EVENTS` below.

Equal weight, ridge (pooled and per hour) and LightGBM are refit monthly on
an expanding window with a one-day embargo. 2026-05-01 onward was held out
and scored once. yfinance serves only ~730 days of hourly bars, so the
window moves with the run date and numbers will drift from the write-up.
"""
import os
import warnings

import numpy as np
import pandas as pd

from backtester import shortterm as st

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

HOLDOUT = "2026-05-01"
SHARED = "/mnt/project-files"


def hourly(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def events() -> dict[str, pd.Series]:
    """Polymarket Iran-escalation and Fed indexes (hourly changes, labelled by
    bucket start) from the "Prediction markets" thread, if present."""
    path = f"{SHARED}/prediction-markets/poly/themes.parquet"
    if not os.path.exists(path):
        return {}
    th = pd.read_parquet(path)
    th.index = th.index + pd.Timedelta("1h")  # known at the end of the hour
    return {"iran_hr": th.iran, "fed_hr": th.fed}


def main():
    bars = {s: hourly(s) for s in ["SPY", "QQQ", "TLT", "HYG", "UUP", "GLD", "USO", "IWM", "SMH",
                                   "^VIX", "^VIX3M", "^VIX9D", "BTC-USD", "ZN=F"]}
    ev = events()
    for tgt, rel in [("SPY", "QQQ"), ("QQQ", "SPY")]:
        P = st.hourly_panel(
            bars[tgt],
            others={"tlt": bars["TLT"], "hyg": bars["HYG"], "uup": bars["UUP"], "gld": bars["GLD"],
                    "uso": bars["USO"], "iwm_rel": bars["IWM"], "smh_rel": bars["SMH"],
                    f"{rel.lower()}_rel": bars[rel]},
            vix=bars["^VIX"], vix3m=bars["^VIX3M"], vix9d=bars["^VIX9D"],
            around_clock={"btc": bars["BTC-USD"], "zn": bars["ZN=F"]}, events=ev,
        ).dropna(subset=["vix9d"])
        # Train on the close-to-close move scaled by the vol VIX9D implies for the time left
        P["z_close"] = P.y_close / (P.vix9d / 100 * np.sqrt(P.minutes_left / st.MINUTES_PER_YEAR))
        feats = [c for c in P.columns if c not in
                 ("price", "close", "y_close", "y_1h", "z_close", "minutes_left", "dow", "vix9d")]
        dev = P[P.index < pd.Timestamp(HOLDOUT, tz=st.NY)]
        start = (dev.index.min() + pd.DateOffset(months=7)).strftime("%Y-%m-01")

        models = {"equal weight": st.zscore_combo, "ridge": st.ridge(50.0), "lightgbm": st.lightgbm()}
        rows = {}
        fair = {h: st.fair_iv_mult(dev[dev.hour == int(h[:2]) + 0.5]) for h in st.DECISION_HOURS}
        for name, fit in models.items():
            for label, (s, e) in {"walk-forward": (start, HOLDOUT), "holdout": (HOLDOUT, None)}.items():
                oos = st.walk_forward(P, feats, "z_close", fit, s, e, min_train=300)
                r = st.score(oos, "y_close")
                # 0DTE: buy the signalled side when |score| is in the top 20% for that hour,
                # vs the same trades on the opposite side, priced at the break-even vol
                sig, ag = [], []
                for h in st.DECISION_HOURS:
                    sig.append(st.zero_dte(oos, fair[h], hours=(h,)))
                    ag.append(st.zero_dte(oos, fair[h], hours=(h,), mode="against"))
                sig, ag = pd.concat(sig), pd.concat(ag)
                r["0dte_trades"] = len(sig)
                r["0dte_ret_signal"] = sig.ret_on_premium.mean() if len(sig) else np.nan
                r["0dte_ret_against"] = ag.ret_on_premium.mean() if len(ag) else np.nan
                rows[(name, label)] = r
        print(f"== {tgt}: {P.index.min().date()} to {P.index.max().date()}, rest-of-day direction")
        print("break-even VIX9D multiple by hour:", {h: round(v, 2) for h, v in fair.items()})
        print(pd.DataFrame(rows).T, "\n")


if __name__ == "__main__":
    main()
