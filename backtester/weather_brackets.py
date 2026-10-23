"""Kalshi daily temperature brackets vs weather-model consensus (alpha hunt #3).

Uses the frozen replay rows of Hugging Face `openthomas/weather-forecasts`:
one row per Kalshi bracket at a decision snapshot, with the model
probability (7-model NWP consensus; `p_model_raw` before calibration), the
Kalshi bid/ask and the outcome. A rule buys YES when the model beats the ask
by `edge`, or NO when the bid beats the model by `edge`, paying the taker fee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROWS = "https://huggingface.co/datasets/openthomas/weather-forecasts/resolve/main/replay/rows-2026-07-09-to-2026-10-21.jsonl"


def load_rows(path: str = ROWS) -> pd.DataFrame:
    r = pd.read_json(path, lines=True)
    r["y"] = r["outcome_yes"].astype(float)
    r["mid"] = (r["yes_bid"] + r["yes_ask"]) / 2
    return r


def taker_fee(p) -> np.ndarray:
    """Kalshi taker fee per contract in dollars, rounded up to the cent."""
    p = np.asarray(p, float)
    return np.ceil(7 * p * (1 - p) - 1e-9) / 100


def brier(rows: pd.DataFrame, cols=("mid", "p_model", "p_model_raw")) -> dict:
    return {c: float(((rows[c] - rows["y"]) ** 2).mean()) for c in cols}


def edge_trades(rows: pd.DataFrame, prob: str = "p_model", edge: float = 0.10) -> pd.DataFrame:
    """One contract per signal: P&L in dollars after the taker fee."""
    buy = rows[(rows[prob] - rows["yes_ask"] >= edge) & (rows["yes_ask"] > 0)]
    sell = rows[(rows["yes_bid"] - rows[prob] >= edge) & (rows["yes_bid"] < 1)]
    no_ask = 1 - sell["yes_bid"]
    return pd.concat([
        pd.DataFrame({"day": buy["day"], "side": "yes", "price": buy["yes_ask"], "pnl": buy["y"] - buy["yes_ask"] - taker_fee(buy["yes_ask"])}),
        pd.DataFrame({"day": sell["day"], "side": "no", "price": no_ask, "pnl": (1 - sell["y"]) - no_ask - taker_fee(no_ask)}),
    ], ignore_index=True)
