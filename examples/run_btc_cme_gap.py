"""Does BTC "fill the CME gap"? (pre-registered, kalshi-macro/HYPOTHESES.md part 3)

CME bitcoin futures stop Friday 17:00 ET and reopen Sunday 18:00 ET while
spot trades all weekend. Retail lore says price returns to the Friday level.
Rule: gap = BTC at Sun 18:00 / BTC at Fri 17:00 - 1; bet on the fill (short
a gap up, long a gap down) from the 18:01 close until price touches the
Friday level or 72 h pass; 10 bp per side. H8 |gap| >= 1%, H9 >= 2%;
placebo uses a Saturday 17:00 reference. Holdout from Jul 2024.
Data: Binance 1-min BTCUSDT (/mnt/project-files/minute-data/BTC.parquet).

    python examples/run_btc_cme_gap.py
"""
import numpy as np
import pandas as pd

BTC = "/mnt/project-files/minute-data/BTC.parquet"
HOLD = pd.Timestamp("2024-07-01", tz="America/New_York")
COST = 0.001


def gap_trades(close: pd.Series, ref_day: int, ref_hour: int, thr: float, hold_h: int = 72) -> pd.DataFrame:
    """close: 1-min closes indexed in ET. ref_day: weekday of the reference
    close (4 = Fri, 5 = Sat); entry is Sunday 18:01."""
    rows = []
    for sun in pd.date_range(close.index.min().normalize(), close.index.max().normalize(), freq="W-SUN"):
        ref_t = sun - pd.Timedelta(days=6 - ref_day) + pd.Timedelta(hours=ref_hour)
        ent_t = sun + pd.Timedelta(hours=18, minutes=1)
        if ref_t not in close.index or ent_t not in close.index:
            continue
        ref, ent = close[ref_t], close[ent_t]
        gap = close[sun + pd.Timedelta(hours=18)] / ref - 1
        if abs(gap) < thr:
            continue
        path = close[ent_t:ent_t + pd.Timedelta(hours=hold_h)]
        side = -np.sign(gap)  # bet on the fill
        hit = path[(path <= ref) if gap > 0 else (path >= ref)]
        exit_px, filled = (ref, True) if len(hit) else (path.iloc[-1], False)
        rows.append({"sunday": sun, "gap": gap, "ret": side * (exit_px / ent - 1) - 2 * COST, "filled": filled,
                     "hours": ((hit.index[0] if len(hit) else path.index[-1]) - ent_t).total_seconds() / 3600})
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> dict:
    if len(t) < 3:
        return {"n": len(t)}
    return {"n": len(t), "fill_rate": round(t["filled"].mean(), 2), "mean_bp": round(t["ret"].mean() * 1e4, 1),
            "t": round(t["ret"].mean() / t["ret"].std() * np.sqrt(len(t)), 2), "med_hours": round(t["hours"].median(), 1)}


def main():
    c = pd.read_parquet(BTC, columns=["close"])["close"]
    c.index = c.index.tz_convert("America/New_York")
    for lab, day, hr in [("Fri 17:00 (CME close)", 4, 17), ("placebo Sat 17:00", 5, 17)]:
        for thr in (0.01, 0.02):
            t = gap_trades(c, day, hr, thr)
            ins, ho = t[t["sunday"] < HOLD], t[t["sunday"] >= HOLD]
            print(f"{lab:<22} |gap|>={thr:.0%}  IS {summary(ins)}  HO {summary(ho)}")


if __name__ == "__main__":
    main()
