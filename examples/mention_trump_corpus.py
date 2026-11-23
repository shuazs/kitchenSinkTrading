"""Trump word-rate features for Kalshi Trump mention markets.

For each Trump market (series KXTRUMP*), counts the strike word in
- his Truth Social posts in the 7 and 30 days before the decision time t_s
  (HF chrissoria/trump-truth-social), and
- a fixed corpus of his interview answers (HF pookie3000/donald_trump_interviews,
  ~114k words; undated, so it is treated as background vocabulary).
Writes /mnt/project-files/mention-transformer/trump_feats_<snap>.parquet,
used by run_mention_models.py ("word-rate" model).
"""
import sys

import numpy as np
import pandas as pd

from backtester import mention_models as mm

D = "/mnt/project-files/mention-transformer/"


def main(snap: str = "h6"):
    s = pd.read_parquet(D + f"snap_{snap}.parquet")
    s = s[s["speaker"].str.startswith("TRUMP")].copy()
    posts = pd.read_parquet(D + "truth_posts.parquet").sort_values("datetime")
    iv = " ".join(pd.read_parquet(D + "trump_interviews.parquet")["a"])
    iv_words = len(iv.split())
    pt = posts["datetime"].values
    rows = []
    for word, g in s.groupby("word"):
        rx = mm.word_regex(word)
        hit = posts["text"].map(lambda x: len(rx.findall(x))).values
        cum = np.concatenate([[0], np.cumsum(hit)])
        cnt_posts = np.concatenate([[0], np.cumsum(hit > 0)])
        iv_rate = len(rx.findall(iv)) / iv_words * 1e4
        for i, t in zip(g.index, g["t_s"].values):
            hi = np.searchsorted(pt, t)
            lo7 = np.searchsorted(pt, t - np.timedelta64(7, "D"))
            lo30 = np.searchsorted(pt, t - np.timedelta64(30, "D"))
            rows.append({"ticker": s.at[i, "ticker"], "ts7": cum[hi] - cum[lo7], "ts30": cum[hi] - cum[lo30],
                         "ts30_posts": cnt_posts[hi] - cnt_posts[lo30], "n30": hi - lo30, "iv_rate": iv_rate})
    out = pd.DataFrame(rows)
    out.to_parquet(D + f"trump_feats_{snap}.parquet")
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h6")
