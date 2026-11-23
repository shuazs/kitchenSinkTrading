"""LoRA speaker LLM + Monte Carlo speech simulation for Trump mention markets.

Josh's idea: fine-tune a small LLM on one speaker's words, sample many
continuations of "the speech", and read P(word is said) off how often each
strike word shows up across the samples.

1. `train`: LoRA (r=8, q/v projections) on Qwen2.5-0.5B over Trump's interview
   answers (HF pookie3000/donald_trump_interviews) plus his Truth Social posts
   before the first test month (2025-11-01), so nothing after the training
   window leaks into the adapter. CPU only, so the run is short.
2. `sim`: for each Trump mention event, prompt with the event title (never the
   strike word: "Will Trump say X" titles get a generic prompt) and his last
   few posts before t_s, sample N continuations, and store the texts.
   `--base` samples from the plain model as a control.
3. Scoring (`hit_rates`) counts, per market, the share of samples that contain
   the word; run_mention_models.py turns that into a calibrated q.

This is the pre-speech version only: there are no timestamped live transcripts
in reach, so re-simulating as the speech comes in can't be backtested here.

Result (166 Trump events, 8 samples x 128 tokens, 100 LoRA steps on CPU): the
sampled hit rate barely tracks outcomes (corr 0.06, same for the plain model),
and adding it to price + word history leaves holdout log loss unchanged
(0.517 LoRA, 0.516 plain, 0.517 history alone). Short samples rarely contain
a given word (97% of markets get 0 hits); a real test needs GPU-length
speeches and live transcripts.

    python examples/mention_llm_sim.py train
    python examples/mention_llm_sim.py sim [--base] [--n 12] [--tokens 200]
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd

D = "/mnt/project-files/mention-transformer/"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = D + "lora_trump"
CUTOFF = "2025-11-01"


def corpus() -> list[str]:
    iv = pd.read_parquet(D + "trump_interviews.parquet")["a"].tolist()
    posts = pd.read_parquet(D + "truth_posts.parquet")
    posts = posts[(posts["datetime"] < CUTOFF) & (posts["datetime"] >= "2024-01-01")]
    posts = posts[posts["text"].str.len() > 80]["text"].tolist()
    return [f"TRUMP: {a}" for a in iv] + [f"TRUMP (post): {p}" for p in posts]


def train(steps: int, seq: int = 256, bs: int = 4):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = tok("\n\n".join(np.random.default_rng(0).permutation(corpus())))["input_ids"]
    blocks = [ids[i:i + seq] for i in range(0, len(ids) - seq, seq)]
    print(f"{len(ids)} tokens, {len(blocks)} blocks; training {steps} steps of {bs}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    order = np.random.default_rng(1).permutation(len(blocks))
    model.train()
    t0 = time.time()
    for st in range(steps):
        b = torch.tensor([blocks[j] for j in order[(st * bs) % len(blocks):][:bs]])
        if len(b) < bs:
            continue
        loss = model(input_ids=b, labels=b).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if st % 10 == 0:
            print(f"step {st} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(ADAPTER)


def prompt(title: str, recent: list[str]) -> str:
    if re.search(r"Will Trump say", title):
        setting = "a public appearance (speech, press gaggle or interview) this week"
    else:
        setting = re.sub(r"^What will (Donald )?Trump say (during|at|in) ", "", title).rstrip("?")
    ctx = "\n".join(f"- {r[:200]}" for r in recent)
    return (f"Transcript of Donald Trump at {setting}.\nWhat he has been posting about lately:\n{ctx}\n\n"
            f"TRUMP: ")


def sim(base: bool, n: int, tokens: int, snap: str = "h6"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    if not base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.eval()
    s = pd.read_parquet(D + f"snap_{snap}.parquet")
    s = s[s["speaker"].str.startswith("TRUMP")]
    ev = s.sort_values("t_s").groupby("event_ticker").agg(title=("title", "first"), t_s=("t_s", "first"))
    posts = pd.read_parquet(D + "truth_posts.parquet").sort_values("datetime")
    posts = posts[posts["text"].str.len() > 60]
    out_path = D + f"llm_sims_{'base' if base else 'lora'}_{snap}.parquet"
    done = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=["event_ticker", "text"])
    rows = done.to_dict("records")
    t0 = time.time()
    for k, (e, r) in enumerate(ev.iterrows()):
        if e in set(done["event_ticker"]):
            continue
        recent = posts[posts["datetime"] < r["t_s"]]["text"].tail(5).tolist()
        enc = tok([prompt(r["title"], recent)] * n, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=tokens,
                               pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"event_ticker": e, "text": t} for t in texts]
        if k % 5 == 0:
            pd.DataFrame(rows).to_parquet(out_path)
            print(f"{k + 1}/{len(ev)} events {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(out_path)


def hit_rates(sims: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Per market: share of sampled continuations that contain the word."""
    from backtester import mention_models as mm

    by_ev = {e: g["text"].tolist() for e, g in sims.groupby("event_ticker")}
    rows = []
    for _, r in snap.iterrows():
        texts = by_ev.get(r["event_ticker"])
        if not texts:
            continue
        rx = mm.word_regex(r["word"])
        k = sum(bool(rx.search(t)) for t in texts)
        rows.append({"ticker": r["ticker"], "sim_hits": k, "sim_n": len(texts)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sim"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.steps)
    else:
        sim(a.base, a.n, a.tokens)


if __name__ == "__main__":
    main()
