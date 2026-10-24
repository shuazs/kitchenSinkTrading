"""Single-stock leveraged ETFs and their end-of-day rebalancing pressure.

A daily-reset ETF with leverage L must trade L(L-1) x its assets x the
underlying's return near the close, in the direction of the move. After a
big down day that selling can overshoot; the batch 10 test
(examples/run_alpha_batch10.py) found the next day's open-to-close rebound
only where this pressure is large. Used there and by
paper_loggers/letf_reversal.py.
"""
import pandas as pd

# underlying -> {LETF: leverage}
LETF = {"TSLA": {"TSLL": 2, "TSLQ": -1, "TSLZ": -2, "TSLT": 2}, "NVDA": {"NVDL": 2, "NVDU": 2, "NVDX": 2, "NVD": -2, "NVDQ": -2},
        "MSTR": {"MSTU": 2, "MSTX": 2, "SMST": -2, "MSTZ": -2}, "COIN": {"CONL": 2, "COIW": 2, "CONI": -2},
        "PLTR": {"PLTU": 2, "PTIR": 2, "PLTD": -1}, "SMCI": {"SMCX": 2, "SMCL": 2}, "HOOD": {"HOOX": 2, "ROBN": 2},
        "AMD": {"AMDL": 2, "AMUU": 2}, "AAPL": {"AAPU": 2, "AAPD": -1}, "AMZN": {"AMZU": 2, "AMZD": -1},
        "META": {"METU": 2, "FBL": 2}, "GOOGL": {"GGLL": 2, "GGLS": -1}, "MSFT": {"MSFU": 2, "MSFD": -1}}
# newer 2x names checked out of sample in batch 12 (examples/run_alpha_batch12.py)
NEW_LETF = {"AVGO": "AVGU AVGX AVL", "NFLX": "NFXL NFLU", "RDDT": "RDTL RDTX", "IONQ": "IONX IONL", "RGTI": "RGTX RGTU",
            "OKLO": "OKLL OKLX", "SMR": "SMU SMUP", "CRWV": "CRWL CWVX CORW", "MU": "MUU", "ARM": "ARMG", "BABA": "BABX",
            "LLY": "LLYX", "UNH": "UNHG", "SOFI": "SOFX", "ASTS": "ASTX", "QBTS": "QBTX", "APP": "APPX", "CRCL": "CRCA",
            "HIMS": "HIMZ", "NBIS": "NBIL", "RIVN": "RVNL", "UBER": "UBRL", "BA": "BOEU", "AFRM": "AFRU", "SHOP": "SHPU",
            "NET": "NETX", "CRWD": "CRWU", "PANW": "PALU", "ORCL": "ORCX", "SNOW": "SNOU", "INTC": "LINT", "DJT": "DJTU",
            "ACHR": "ACHX", "RKLB": "RKLX", "SOUN": "SOUX", "U": "UXX", "LCID": "LCDL", "CVNA": "CVNX", "BMNR": "BMNU", "SBET": "SBTU"}
ALL_LETF = {**LETF, **{u: {e: 2 for e in s.split()} for u, s in NEW_LETF.items()}}


def down_z(close: pd.Series, window: int = 60) -> pd.Series:
    """Daily return in units of the trailing `window`-day volatility (known at the close)."""
    r = close.pct_change(fill_method=None)
    return r / r.rolling(window).std().shift(1)


def pressure(close: pd.DataFrame, volume: pd.DataFrame, underlying: str, legs: dict[str, float] | None = None,
             min_obs: int = 51) -> pd.Series:
    """sum |L(L-1)| x LETF 20-day $ volume / underlying 20-day $ volume, each lagged a day.
    LETFs with fewer than `min_obs` closes are left out."""
    legs = {s: lv for s, lv in (legs or ALL_LETF[underlying]).items() if s in close and close[s].notna().sum() >= min_obs}
    dv = sum(abs(lv * (lv - 1)) * (close[s] * volume[s]).rolling(20, min_periods=5).mean().shift(1).fillna(0)
             for s, lv in legs.items())
    return dv / (close[underlying] * volume[underlying]).rolling(20).mean().shift(1)
