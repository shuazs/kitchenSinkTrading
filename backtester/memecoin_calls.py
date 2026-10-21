"""Following public meme coin "calls" on pump.fun, with realistic fills.

A pump.fun token trades on a constant-product bonding curve with virtual
reserves of 30 SOL and 1.073B tokens at launch, so the market cap in SOL is
v**2 / 32.19 where v = 30 + the curve's real SOL reserve. That lets a call's
dollar market cap be turned into a curve state, and our own buy and sell be
pushed through the curve to get price impact. The curve graduates at 85 SOL
of real reserve (about 410 SOL of market cap).

What the data cannot show is the price path between the call and the end,
so outcomes are resolved generously for the trader:

- a call whose peak, measured from our entry, reaches `take_profit` sells
  exactly there;
- otherwise a graduated token is sold flat at our entry price;
- otherwise the token is sold at its last observed curve reserve, or at
  `stop_loss` if one is set (assumed to fill exactly, which it would not
  in a rug).

`delay_premium` is how far the price has already run between the call and
our fill: bots read the same channels and buy first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

V0_SOL = 30.0
V0_TOKENS = 1.073e9
K = V0_SOL * V0_TOKENS
SUPPLY = 1e9


def mc_to_v(mc_sol):
    """Virtual SOL reserve for a market cap in SOL."""
    return np.sqrt(np.asarray(mc_sol, float) * K / SUPPLY)


def buy(v_sol: float, sol_in: float) -> tuple[float, float]:
    """Tokens out and the new virtual SOL reserve."""
    return K / v_sol - K / (v_sol + sol_in), v_sol + sol_in


def sell(v_sol: float, tokens: float) -> float:
    """SOL out for selling tokens at virtual SOL reserve v_sol."""
    return v_sol - K / (K / v_sol + tokens)


def simulate_calls(
    calls: pd.DataFrame,
    sol_usd: pd.Series,
    size_usd: float = 100.0,
    take_profit: float = 2.0,
    stop_loss: float | None = None,
    delay_premium: float = 0.10,
    fee: float = 0.01,
    tx_cost_usd: float = 0.5,
) -> pd.DataFrame:
    """Per-call trade results.

    `calls` needs mc (dollars), peak, time, graduated and terminal_rs (from
    load_launches). `sol_usd` is a daily SOL price used to convert dollars.
    Returns the calls with `ret` (return on size_usd after costs) and `exit`.
    """
    sol = sol_usd.reindex(calls["time"].dt.normalize()).to_numpy(float)
    rows = []
    for c, px in zip(calls.itertuples(index=False), sol):
        if not (px > 0 and c.mc > 0):
            rows.append((np.nan, "no_price"))
            continue
        v_call = float(mc_to_v(c.mc / px))
        v_in = v_call * np.sqrt(1 + delay_premium)
        tokens, _ = buy(v_in, size_usd / px * (1 - fee))
        peak_from_entry = c.peak / (1 + delay_premium)
        if peak_from_entry >= take_profit:
            v_out, how = v_in * np.sqrt(take_profit), "take_profit"
        elif c.graduated == 1:
            v_out, how = v_in, "flat"
        else:
            v_out, how = V0_SOL + c.terminal_rs, "terminal"
            if stop_loss is not None and v_out < v_in * np.sqrt(stop_loss):
                v_out, how = v_in * np.sqrt(stop_loss), "stop"
        proceeds = sell(v_out, tokens) * (1 - fee) * px - 2 * tx_cost_usd
        rows.append((proceeds / size_usd - 1, how))
    out = calls.copy()
    out["ret"], out["exit"] = zip(*rows)
    return out


def bankroll(results: pd.DataFrame, capital: float = 5_000.0, bet_frac: float = 0.02) -> pd.Series:
    """Equity from betting `bet_frac` of current equity on each call in time
    order (one at a time, so it ignores overlap between open trades)."""
    eq = capital * np.cumprod(1 + bet_frac * results["ret"].dropna().to_numpy())
    return pd.Series(eq, index=results["time"][results["ret"].notna()].to_numpy(), name="equity")
