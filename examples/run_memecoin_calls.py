"""Would following public pump.fun meme coin calls make money?

    python examples/run_memecoin_calls.py

Joins a public Telegram call feed (with the losing calls kept in) to
on-chain launch outcomes, then pushes a $100 buy and sell per call through
the pump.fun bonding curve with fees. Outcomes are resolved in the trader's
favour wherever the data has no price path; see backtester/memecoin_calls.py.
"""
import pandas as pd

from backtester.data.memecoins import load_calls, load_launches
from backtester.data.yahoo import load_bars
from backtester.memecoin_calls import bankroll, simulate_calls

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)

# The launch file's complete-decoder subset starts 2026-09-27.
days = pd.date_range("2026-09-27", "2026-10-14")
launches = load_launches(days)
launches = launches[(launches["quote_asset"] == "sol") & (launches["decoder_complete"] == 1)]
calls = load_calls()
calls = calls[(calls["time"] >= days[0]) & (calls["time"] < days[-1] + pd.Timedelta(days=1))]
joined = calls.merge(launches, on="mint", how="inner").dropna(subset=["graduated"])
joined = joined[(joined["graduated"] == 1) | joined["terminal_rs"].notna()]
sol = load_bars("SOL-USD", start="2026-09-01", refresh=True)["close"]

print(f"{len(calls)} calls in the window, {len(joined)} matched to resolved launches")
print(f"graduated {joined['graduated'].mean():.1%}, rug rule fired {joined['rugged'].mean():.1%}, "
      f"median peak {joined['peak'].median():.2f}x, doubled {(joined['peak'] >= 2).mean():.1%}")

rows = []
for delay in [0.0, 0.10, 0.25]:
    for tp in [1.3, 1.5, 2.0, 3.0, 5.0]:
        for stop in [None, 0.5]:
            r = simulate_calls(joined, sol, take_profit=tp, stop_loss=stop, delay_premium=delay)
            eq = bankroll(r)
            rows.append({
                "fill_delay_premium": delay,
                "take_profit": tp,
                "stop": "none" if stop is None else f"{stop:.0%} (perfect fill)",
                "win_rate": (r["exit"] == "take_profit").mean(),
                "mean_ret": r["ret"].mean(),
                "median_ret": r["ret"].median(),
                "bankroll_2pct": eq.iloc[-1],
            })
print("\n== $100 per call, 1% fee each way, $0.50 per transaction, curve price impact")
print(pd.DataFrame(rows).to_string(index=False))
