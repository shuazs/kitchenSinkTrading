"""On-chain batch 1 follow-up: top features vs trend filters and a 500-draw block-shuffle placebo (holdout)."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
import numpy as np, pandas as pd
from run_onchain_batch1 import *
stable=stablecoin_supply()
rng=np.random.default_rng(0)
def blockshuf(sig,n=500,blk=30):
    v=sig.values.astype(float); L=len(v); out=[]
    for _ in range(n):
        nb=L//blk+1; starts=rng.integers(0,L-blk,nb)
        out.append(np.concatenate([v[s:s+blk] for s in starts])[:L])
    return out
for coin in ("btc","eth"):
    d=load_coin(coin); px=d.PriceUSD.dropna(); ret=px.pct_change().loc[START[coin]:]
    F=features(coin,stable)
    trend={f"trend{w}":(px>px.rolling(w).mean()).loc[ret.index] for w in (50,100,200)}
    cands={"stable90>0":zscore(F["stable90"])>0,"stable90>1":zscore(F["stable90"])>1,"mvrv>1":zscore(F["mvrv"])>1,"mvrv_mom90>0":zscore(F["mvrv_mom90"])>0,"adr30>.5":zscore(F["adr30"])>.5}
    sigs={**trend,**{k:v.loc[ret.index] for k,v in cands.items()}}
    for t in ("trend100",):
        for k,v in cands.items(): sigs[f"{t}&{k}"]=trend[t]&v.loc[ret.index]
    print("=====",coin)
    for k,s in sigs.items():
        r=run(s,ret); a,b=r.loc[:SPLIT].iloc[:-1],r.loc[SPLIT:]
        # placebo on holdout: shuffled signal same exposure
        sh=[]
        for v in blockshuf(s.loc[SPLIT:]):
            ss=pd.Series(v,index=s.loc[SPLIT:].index); rr=run(ss,ret.loc[SPLIT:]); sh.append(stats(rr)["sharpe"])
        p=(np.array(sh)>=stats(b)["sharpe"]).mean()
        print(f"{k:22s} des {stats(a)['sharpe']:.2f} hold {stats(b)['sharpe']:.2f} cagr_h {stats(b)['cagr']:.2f} dd_h {stats(b)['dd']:.2f} exp_h {s.loc[SPLIT:].mean():.2f} placebo_p {p:.2f}")
