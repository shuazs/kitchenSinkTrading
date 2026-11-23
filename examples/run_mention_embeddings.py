"""Frozen-transformer variant for Kalshi mention markets (cheap alternative to fine-tuning).

MiniLM sentence embeddings of "speaker | title | word" (no fine-tuning), cut
to 32 PCA components fit on the training months, plus the price and history
features of run_mention_transformer.py, in a logistic regression refit
walk-forward each month (Aug 2025 - Jan 2026, snapshot h6). Minutes on a CPU.

Result: the best-calibrated model (holdout log loss 0.489 vs 0.519 for the
price), but it does not trade: both sides at > 3c edge +3.5c holdout (event
t 0.3), No only +5.4c (t 1.9), Aug-Oct +1.9c; its ranking of buy-No trades is
weaker than the fine-tuned net's.

    python examples/run_mention_embeddings.py
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from backtester import mention_models as mm
from run_mention_models import rank_no_trades
from run_mention_transformer import MONTHS, numeric, text

D = "/mnt/project-files/mention-transformer/"


def main():
    s = pd.read_parquet(D + "snap_h6.parquet")
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(text(s), batch_size=64)
    num = numeric(s)
    q = pd.Series(np.nan, index=s.index)
    for lo, hi in zip(MONTHS[:-1], MONTHS[1:]):
        tr = (s["ev_close"] < lo).values
        te = ((s["ev_close"] >= lo) & (s["ev_close"] < hi)).values
        pca = PCA(32, random_state=0).fit(emb[tr])
        mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-6

        def feats(i):
            return np.hstack([(num[i] - mu) / sd, pca.transform(emb[i])])

        clf = LogisticRegression(C=0.1, max_iter=3000).fit(feats(tr), s["y"][tr])
        q[te] = clf.predict_proba(feats(te))[:, 1]
    ho = s["ev_close"] >= "2025-11-01"
    print(f"holdout log loss: price {mm.log_loss(s['y'][ho], s['p0'][ho]):.4f}, model {mm.log_loss(s['y'][ho], q[ho]):.4f}")
    for per, g in [("Aug-Oct", s[~ho]), ("Nov-Jan", s[ho])]:
        print(per, "both >3c", mm.summary(mm.trades(g, q[g.index], 0.03)), "| No >3c",
              mm.summary(mm.trades(g, q[g.index], 0.03, side="no")))
    print(rank_no_trades(s, q).to_string())
    pd.DataFrame({"ticker": s["ticker"], "q": q}).dropna().to_parquet(D + "q_emb_h6.parquet")


if __name__ == "__main__":
    main()
