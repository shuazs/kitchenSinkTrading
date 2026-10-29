"""Earnings-call text signals: lexical tone/hedging/evasion features from S&P 500 call transcripts.

Stage 1 (`features`): stream the HF parquet shards (glopardo/sp500-earnings-transcripts), compute word-count
features per call (prepared remarks vs Q&A), write features.parquet.
Stage 2 (`backtest`) lives in run_earnings_call_backtest.py.
Usage: python examples/run_earnings_call_text.py features OUTDIR SHARD...
"""
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

POS = set("""achieve achieved achievement attractive benefit benefited beneficial best better strong stronger strongest
strength strengthen strengthened improve improved improvement improving gain gains gained growth grew growing record
outperform outperformed exceed exceeded exceeding excellent favorable favorably opportunity opportunities
positive positively profitable profitability progress success successful succeeded momentum robust solid
confident confidence pleased delighted accelerate accelerated accelerating""".split())
NEG = set("""loss losses decline declined declining decrease decreased decreasing weak weaker weakness weakened
adverse adversely difficult difficulties challenge challenges challenging deteriorate deteriorated deterioration
impair impaired impairment negative negatively poor poorly pressure pressures slowdown slowing slow slowed
shortfall shortfalls unfavorable unfavorably disappoint disappointing disappointed concern concerns concerned
headwind headwinds risk risks risky volatile volatility restructuring litigation lawsuit writedown charge charges
delay delayed delays underperform underperformed""".split())
UNC = set("""may might could possibly perhaps uncertain uncertainty uncertainties unclear unpredictable depend depends
depending approximately roughly somewhat appear appears believe believes anticipate anticipates assume assumes
suggest suggests probably likely unlikely fluctuate contingent""".split())
HEDGE = set("""maybe sort kind somewhat fairly relatively pretty bit little hopefully presumably generally typically
essentially basically arguably""".split())
RAISE = re.compile(r"\b(rais(?:e|ed|ing) (?:our )?(?:guidance|outlook|forecast|full[- ]year)|above (?:our )?(?:guidance|expectations)|"
                   r"better than (?:expected|anticipated)|increas(?:e|ed|ing) (?:our )?(?:guidance|outlook))\b")
CUT = re.compile(r"\b(lower(?:ed|ing)? (?:our )?(?:guidance|outlook|forecast)|reduc(?:e|ed|ing) (?:our )?(?:guidance|outlook|forecast)|"
                 r"below (?:our )?(?:guidance|expectations)|worse than (?:expected|anticipated)|cut (?:our )?(?:guidance|outlook))\b")
EVADE = re.compile(r"\b(not going to (?:comment|get into|speculate|give)|(?:can't|cannot|won't|don't) (?:comment|speculate|give|provide|break out|disclose)|"
                   r"get back to you|take that offline|not (?:going to )?(?:provide|break out)|too early to (?:say|tell)|"
                   r"we'll see|hard to say|difficult to (?:say|predict|quantify)|not ready to)\b")
WORD = re.compile(r"[a-z']+")
QA_MARK = re.compile(r"(question[- ]and[- ]answer|\[operator instructions\]|first question|open (?:it |the (?:line|call) )?up for questions|take (?:your|our) questions)", re.I)


def split_call(text: str):
    m = QA_MARK.search(text, int(len(text) * 0.15))
    return (text[: m.start()], text[m.start():]) if m else (text, "")


def feats(text: str, pre: str) -> dict:
    t = text.lower()
    w = WORD.findall(t)
    n = max(len(w), 1)
    out = {pre + "n": len(w)}
    for name, S in (("pos", POS), ("neg", NEG), ("unc", UNC), ("hedge", HEDGE)):
        out[pre + name] = sum(x in S for x in w) / n
    out[pre + "raise"] = len(RAISE.findall(t)) / n * 1000
    out[pre + "cut"] = len(CUT.findall(t)) / n * 1000
    out[pre + "evade"] = len(EVADE.findall(t)) / n * 1000
    out[pre + "digits"] = sum(c.isdigit() for c in t) / max(len(t), 1)
    return out


def run_features(outdir: str, shards: list[str]) -> None:
    rows = []
    for sh in shards:
        f = pq.ParquetFile(sh)
        meta_cols = [c for c in f.schema.names if c != "transcript"]
        for g in range(f.num_row_groups):
            df = f.read_row_group(g).to_pandas()
            for _, r in df.iterrows():
                tr = r["transcript"] or ""
                prep, qa = split_call(tr)
                d = {c: r[c] for c in ["ticker", "sector", "earnings_date", "datacqtr", "eps12mtrailing_eoq", "eps12mfwd_eoq", "eps12mfwd_qavg", "peforw_eoq"]}
                d.update(feats(prep, "p_"))
                d.update(feats(qa, "q_") if qa else {})
                rows.append(d)
            print(sh, g, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(f"{outdir}/features.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "features":
        run_features(sys.argv[2], sys.argv[3:])
