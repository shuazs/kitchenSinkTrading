"""Fine-tuned text transformer for Kalshi mention markets (needs torch + transformers).

Input per market: "speaker | event title | word" through MiniLM, its [CLS]
vector concatenated with numeric features (logit of the last price before
t_s and the causal history rates from `mention_models.history_features`),
then a linear head. The head starts as a copy of the price, so the net
learns a correction to the market rather than the price itself.

Walk-forward: train on events that closed before each test month, test
on Nov 2025, Dec 2025 and Jan 2026 (the holdout). Writes out-of-sample q to
/mnt/project-files/mention-transformer/q_transformer.parquet; scoring is in
run_mention_models.py.

    python examples/run_mention_transformer.py [--model sentence-transformers/all-MiniLM-L6-v2] [--text-only]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from backtester import mention_models as mm

D = "/mnt/project-files/mention-transformer/"
MONTHS = ["2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"]


def logit(x):
    x = np.clip(np.asarray(x, dtype=float), 0.01, 0.99)
    return np.log(x / (1 - x))


def numeric(d: pd.DataFrame) -> np.ndarray:
    cols = [logit(d["p0"])]
    for k in ["sw", "pw", "w", "ser"]:
        cols += [logit(d[f"r_{k}"]), np.log1p(d[f"n_{k}"])]
    cols += [np.log1p(d["vol_before"])]  # no hours_left: it uses the actual close
    return np.stack(cols, 1).astype(np.float32)


def text(d: pd.DataFrame) -> list[str]:
    return (d["speaker"].str.title() + " | " + d["title"].fillna("") + " | " + d["word"]).tolist()


class Net(nn.Module):
    def __init__(self, name: str, n_num: int, use_text: bool = True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.use_text = use_text
        h = self.enc.config.hidden_size
        self.txt = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))
        self.num = nn.Linear(n_num, 1)
        with torch.no_grad():  # start at the market price
            self.num.weight.zero_()
            self.num.weight[0, 0] = 1.0
            self.num.bias.zero_()
            self.txt[1].weight.zero_()
            self.txt[1].bias.zero_()

    def forward(self, ids, mask, x):
        out = self.num(x)
        if self.use_text:
            e = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            e = (e * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            out = out + self.txt(e)
        return out.squeeze(-1)


def fit_predict(tr, te, name, use_text, epochs=3, bs=32, lr=3e-5, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    mu, sd = numeric(tr).mean(0), numeric(tr).std(0) + 1e-6
    mu[0], sd[0] = 0.0, 1.0  # keep the price logit as is

    def batches(d, shuffle):
        enc = tok(text(d), padding=True, truncation=True, max_length=64, return_tensors="pt")
        x = torch.tensor((numeric(d) - mu) / sd)
        y = torch.tensor(d["y"].values, dtype=torch.float32)
        idx = np.random.permutation(len(d)) if shuffle else np.arange(len(d))
        for i in range(0, len(d), bs):
            j = idx[i:i + bs]
            yield enc["input_ids"][j], enc["attention_mask"][j], x[j], y[j]

    net = Net(name, numeric(tr).shape[1], use_text)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": lr},
                             {"params": list(net.txt.parameters()) + list(net.num.parameters()), "lr": 1e-3}],
                            weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for ids, mask, x, y in batches(tr, True):
            opt.zero_grad()
            loss = lossf(net(ids, mask, x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        print(f"    epoch {ep + 1} train loss {tot / len(tr):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        q = np.concatenate([torch.sigmoid(net(ids, mask, x)).numpy() for ids, mask, x, _ in batches(te, False)])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--snap", default="h6", help="snapshot file tag: h6 = 6 h after listing (causal)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--no-text", action="store_true", help="control: same net without the text encoder")
    ap.add_argument("--tag", default="transformer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--months", default=",".join(MONTHS), help="test-month boundaries, e.g. pre-holdout 2025-08-01,...,2025-11-01")
    a = ap.parse_args()
    torch.set_num_threads(4)
    s = pd.read_parquet(D + f"snap_{a.snap}.parquet")
    outs = []
    months = a.months.split(",")
    for lo, hi in zip(months[:-1], months[1:]):
        tr = s[s["ev_close"] < lo]
        te = s[(s["ev_close"] >= lo) & (s["ev_close"] < hi)]
        print(f"test {lo[:7]}: train {len(tr)} test {len(te)}", flush=True)
        q = fit_predict(tr, te, a.model, not a.no_text, a.epochs, seed=a.seed)
        outs.append(te[["ticker"]].assign(q=q))
        print(f"    log loss price {mm.log_loss(te['y'], te['p0']):.4f} model {mm.log_loss(te['y'], q):.4f}", flush=True)
    pd.concat(outs).to_parquet(D + f"q_{a.tag}_{a.snap}.parquet")


if __name__ == "__main__":
    main()
