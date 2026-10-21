"""Crypto cash-and-carry (long spot, short the CME front month) on $5,000.

    python examples/run_basis_carry.py
"""
import pandas as pd

from backtester.data.fred import tbill_rate
from backtester.strategies.basis import BasisCarry, load_basis_inputs

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

tbill = tbill_rate()
# Yahoo's futures and spot closes are hours apart; before 2021 the gap swamps the basis.
start = "2021-03-01"
for coin in ["BTC", "ETH"]:
    inputs = load_basis_inputs(coin)
    runs = {
        "every month": BasisCarry(),
        "when basis > T-bill": BasisCarry(hurdle=0.0),
        "when basis > T-bill + 3%": BasisCarry(hurdle=0.03),
        "> T-bill, 50% margin": BasisCarry(hurdle=0.0, margin=0.5),
    }
    out = {}
    for name, s in runs.items():
        st = s.run(inputs, cash_rate=tbill, start=start).stats(rf=tbill)
        months = s.monthly(inputs, tbill)
        months = months[months["start"] >= start]
        st["months_held"] = months["hold"].mean()
        st["basis_when_held"] = months.loc[months["hold"], "ann_basis"].mean()
        out[name] = st
    print(f"\n{coin}")
    print(pd.DataFrame(out).loc[["start", "end", "cagr", "max_drawdown", "months_held", "basis_when_held", "total_costs"]])
    print(f"average T-bill rate over the period: {tbill.loc[start:].mean():.3f}")
    m = BasisCarry().monthly(inputs, tbill)
    print("median annualised basis by year:")
    print(m.groupby(m["start"].dt.year)[["ann_basis", "tbill"]].median().T)
