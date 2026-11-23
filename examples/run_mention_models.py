"""Compare models that price Kalshi mention markets, out of sample after fees.

Data: Kalshi mention markets (HF TrevorJS/kalshi-trades via
/mnt/project-files/event-markets/), Feb 2025 - Jan 2026. Each market is
priced at t_s = 6 h after its event was listed (snapshot "h6"; a market must
have printed before t_s and still be open); entries are the first taker
print of the side bought within 6 h after t_s, held to settlement, Kalshi
taker fee 0.07 p (1-p). Walk-forward: models are refit on events that closed
before each test month, Aug 2025 - Jan 2026; Nov-Jan is the holdout. (A first
pass put t_s at 25% of the listing-to-close window; close_time is the actual
close, so that leaned on the future and inflated the buy-No rule.)

Models (all output P(Yes)):
- price: the last print before t_s (the market itself)
- buy-No 20-80c: the old near-miss rule (no model)
- history: logistic on price + causal per-word / per-speaker hit rates
- transformer: MiniLM fine-tuned on "speaker | title | word" + the history
  features (examples/run_mention_transformer.py), and the same net without text
- Trump only: word-rate (Truth Social counts before t_s, interview vocabulary)
  and the LoRA LLM speech simulation (examples/mention_llm_sim.py) vs the plain LLM

Trade rule for every model, fixed before the holdout: take the side whose
model edge after fee is > 3c. Bankroll: $5,000, 2% of equity per trade,
<= 10% per event, capped at the printed size.

Result: nothing prices these markets better than the market after fees
(event-clustered t < 2.2 for every model on the holdout); the transformer's
ranking of buy-No trades is the one lead (see `rank_no_trades`).

    python examples/run_mention_models.py [--snap h6]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtester import mention_models as mm

D = "/mnt/project-files/mention-transformer/"
MONTHS = ["2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"]
HOLD = "2025-11-01"
THR = 0.03


def lg(x):
    x = np.clip(np.asarray(x, dtype=float), 0.01, 0.99)
    return np.log(x / (1 - x))


def hist_X(d):
    f = pd.DataFrame({"lp": lg(d["p0"])}, index=d.index)
    for k in ["sw", "pw", "w", "ser"]:
        f["lr_" + k] = lg(d["r_" + k])
        f["ln_" + k] = np.log1p(d["n_" + k])
    return f


def trump_X(d):
    f = hist_X(d)
    for c in ["ts7", "ts30", "ts30_posts"]:
        f[c] = np.log1p(d[c])
    f["iv"] = np.log1p(d["iv_rate"])
    return f


def sim_X(d):
    f = hist_X(d)
    f["sim"] = lg((d["sim_hits"] + 0.5) / (d["sim_n"] + 1))
    return f


def walk_forward(s, featfn, C=0.5):
    q = pd.Series(np.nan, index=s.index)
    for lo, hi in zip(MONTHS[:-1], MONTHS[1:]):
        tr = s[s["ev_close"] < lo]
        te = s[(s["ev_close"] >= lo) & (s["ev_close"] < hi)]
        if len(te) == 0:
            continue
        clf = LogisticRegression(C=C, max_iter=3000).fit(featfn(tr), tr["y"])
        q[te.index] = clf.predict_proba(featfn(te))[:, 1]
    return q


def report(name, s, q, rows):
    ho = s[s["ev_close"] >= HOLD]
    qh = q.loc[ho.index] if isinstance(q, pd.Series) else q
    ok = pd.notna(qh)
    t = mm.trades(ho[ok], qh[ok], THR)
    tn = mm.trades(ho[ok], qh[ok], THR, side="no")
    eq = mm.bankroll(t)
    rows.append({"model": name, "markets": int(ok.sum()), "logloss": round(mm.log_loss(ho["y"][ok], qh[ok]), 4),
                 "logloss_price": round(mm.log_loss(ho["y"][ok], ho["p0"][ok]), 4),
                 **{f"{k}": v for k, v in mm.summary(t).items()},
                 "no_only_net_c": mm.summary(tn)["net_c"], "no_only_t": mm.summary(tn)["event_t"],
                 "end_$": round(eq.iloc[-1]) if len(eq) else 5000})
    return t


def rank_no_trades(s, q):
    """Buy-No 20-80c trades split by the model's No edge: top vs bottom half
    per period, and the top-minus-bottom gap inside the same event."""
    r = s.assign(q=q.values)[s["no_ask"].between(0.2, 0.8) & q.notna().values].copy()
    r["net"] = (1 - r["y"]) - r["no_ask"] - mm.fee(r["no_ask"])
    r["e"] = (1 - r["q"]) - r["no_ask"]
    r["period"] = np.where(r["ev_close"] < HOLD, "Aug-Oct", "Nov-Jan")
    r["top"] = r.groupby("period")["e"].transform(lambda x: x > x.median())
    out = r.groupby(["period", "top"]).apply(lambda g: pd.Series(mm.summary(g)))

    def gap(g):
        m = g["e"].median()
        return g.loc[g["e"] > m, "net"].mean() - g.loc[g["e"] <= m, "net"].mean() if len(g) > 1 else np.nan

    for per, g in r.groupby("period"):
        d = g.groupby("event_ticker").apply(gap).dropna()
        print(f"  {per}: within-event top-minus-bottom {d.mean() * 100:+.2f}c, t {d.mean() / d.std() * np.sqrt(len(d)):.2f}, {len(d)} events")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="h6")
    a = ap.parse_args()
    s = pd.read_parquet(D + f"snap_{a.snap}.parquet")
    rows = []
    report("price (market)", s, s["p0"], rows)
    ho = s[s["ev_close"] >= HOLD]
    r = ho[ho["no_ask"].between(0.2, 0.8)].assign(px=lambda x: x["no_ask"])
    r["net"] = (1 - r["y"]) - r["px"] - mm.fee(r["px"])
    r["size"] = r["size_no"]
    rows.append({"model": "buy-No 20-80c rule", "markets": len(ho), **mm.summary(r),
                 "end_$": round(mm.bankroll(r).iloc[-1])})
    qh = walk_forward(s, hist_X)
    report("history + price", s, qh, rows)
    for tag, lab in [("tf", "transformer (text + history)"), ("notext", "same net, no text")]:
        f = D + f"q_{tag}_{a.snap}.parquet"
        if os.path.exists(f):
            qt = s[["ticker"]].merge(pd.read_parquet(f), on="ticker", how="left")["q"]
            qt.index = s.index
            report(lab, s, qt, rows)
    print(f"All mention markets, holdout Nov 2025 - Jan 2026 (edge > {THR * 100:.0f}c after fee)")
    print(pd.DataFrame(rows).to_string(index=False))
    f = D + f"q_tf_{a.snap}.parquet"
    if os.path.exists(f):
        print("\nBuy-No 20-80c trades ranked by the transformer (walk-forward, both periods)")
        qt = s[["ticker"]].merge(pd.read_parquet(f), on="ticker", how="left")["q"]
        print(rank_no_trades(s, qt).to_string())

    tf = D + f"trump_feats_{a.snap}.parquet"
    if os.path.exists(tf):
        st = s.merge(pd.read_parquet(tf), on="ticker")
        st.index = range(len(st))
        rows = []
        report("price (market)", st, st["p0"], rows)
        report("history + price", st, walk_forward(st, hist_X), rows)
        report("word-rate (posts + interviews)", st, walk_forward(st, trump_X), rows)
        for kind in ["lora", "base"]:
            f = D + f"llm_sims_{kind}_{a.snap}.parquet"
            if os.path.exists(f):
                from mention_llm_sim import hit_rates
                ss = st.merge(hit_rates(pd.read_parquet(f), st), on="ticker")
                ss.index = range(len(ss))
                raw = (ss["sim_hits"] + 0.5) / (ss["sim_n"] + 1)
                print(f"  {kind} LLM: raw sim rate vs outcome corr {np.corrcoef(raw, ss['y'])[0, 1]:.3f}, "
                      f"vs price {np.corrcoef(raw, ss['p0'])[0, 1]:.3f}, {len(ss)} markets")
                report(f"LLM sim ({kind}) + history", ss, walk_forward(ss, sim_X), rows)
        print("\nTrump markets only, same holdout")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
