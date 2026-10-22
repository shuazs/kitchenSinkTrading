"""Non-financial prediction markets (sports, politics, weather, culture) on Kalshi and Polymarket.

Uses the ready-made files in /mnt/project-files/event-markets/ (cloud threads)
or rebuilds the Kalshi side from the Hugging Face mirror (slow: ~5GB scan).

    python examples/run_event_markets.py              # summary of every test
    python examples/run_event_markets.py --rebuild    # re-run calibration and the Kalshi-vs-Polymarket tests
    python examples/run_event_markets.py --families   # per-family tables: games, soccer, UFC, esports, futures, elections, awards, Fed

Tests: taker calibration by category and price, Kalshi-vs-Polymarket
two-venue arbitrage and the Kalshi -> Polymarket lead on matched sports
games. Bracket arbitrage and jump tests were run by the scratch scripts in
event-markets/scripts/; their results files are summarised here.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtester.event_markets import arb_episodes, cross_arb, lead_follow, taker_prints  # noqa: E402

DIR = os.environ.get("KS_EVENT_DIR", "/mnt/project-files/event-markets")


def game_t(d: pd.DataFrame, col: str = "ret") -> float:
    g = d.groupby("game")[col].mean()
    return float(g.mean() / (g.std() / np.sqrt(len(g))))


def calibration_table(rebuild: bool) -> pd.DataFrame:
    path = f"{DIR}/ev_agg.parquet"
    if rebuild or not os.path.exists(path):
        m = pd.read_parquet(f"{DIR}/nf_markets.parquet", columns=["ticker", "event_ticker", "series", "cat", "result", "close_time"])
        parts = []
        for f in sorted(glob.glob(f"{DIR}/raw/nf_trades_*.parquet")):
            p = taker_prints(pd.read_parquet(f), m)
            p["year"] = p["created_time"].dt.year
            p["bucket"] = pd.cut(p["price"], [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1.0001], right=False).astype(str)
            w = p["count"]
            p = p.assign(cw=w, pw=p["price"] * w, ww=p["won"] * w, nw=p["net"] * w)
            parts.append(p.groupby(["cat", "series", "event_ticker", "year", "bucket", "taker_side"])[["cw", "pw", "ww", "nw"]].sum().reset_index())
        pd.concat(parts).to_parquet(path)
    a = pd.read_parquet(path)
    a["period"] = np.where(a["year"] <= 2024, "2021-24", "2025+")
    e = a.groupby(["cat", "period", "event_ticker"])[["cw", "nw", "pw", "ww"]].sum()
    e["en"] = e["nw"] / e["cw"]
    g = e.groupby(level=[0, 1])
    out = pd.DataFrame({"events": g.size(), "Mcontracts": g["cw"].sum() / 1e6,
                        "taker_net_c": 100 * g["nw"].sum() / g["cw"].sum(),
                        "maker_gross_c": 100 * (g["pw"].sum() - g["ww"].sum()) / g["cw"].sum(),
                        "t_events": g["en"].mean() / (g["en"].std() / np.sqrt(g.size()))})
    return out.round(2)


def cross_venue(rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    arb_path, lf_path = f"{DIR}/xv_arb.parquet", f"{DIR}/leadfollow_sweep2.parquet"
    if rebuild or not os.path.exists(arb_path):
        mt = pd.read_parquet(f"{DIR}/xvenue_matches.parquet")
        k = pd.read_parquet(f"{DIR}/xvenue_kalshi_prints.parquet").rename(columns={"created_time": "time"})
        p = pd.read_parquet(f"{DIR}/xvenue_poly_fills.parquet")
        p["time"] = pd.to_datetime(p["timestamp"], unit="s", utc=True)
        kg, pg = dict(tuple(k.groupby("ticker"))), dict(tuple(p.groupby("market_id")))
        arbs, lfs = [], []
        for r in mt.itertuples():
            ka, kb, pf = kg.get(r.k_a), kg.get(r.k_b), pg.get(str(r.pm_id))
            if ka is None or kb is None or pf is None:
                continue
            leg = lambda d, side, px: d[d["taker_side"] == side].assign(price=px, size=d["count"])  # noqa: E731
            kA = leg(ka, "yes", ka["yes_price"] / 100)
            kB = leg(kb, "yes", kb["yes_price"] / 100)
            pA = pf[pf["side"] == "BUY"].assign(size=pf["usd_amount"] / pf["price"])
            pB = pf[pf["side"] == "SELL"].assign(price=1 - pf["price"], size=pf["usd_amount"] / pf["price"])
            for name, l1, l2 in [("Kalshi A + Poly B", kA, pB), ("Kalshi B + Poly A", kB, pA)]:
                e = arb_episodes(cross_arb(l1, l2, "10s", 0.07, 0.0), legs=1)
                if len(e):
                    arbs.append(e.assign(game=r.slug, lg=r.lg, legs=name))
            lead = ka.assign(yes=ka["yes_price"] / 100)
            fol = pf.assign(yes=pf["price"], taker_side=np.where(pf["side"] == "BUY", "yes", "no"),
                            size=pf["usd_amount"] / pf["price"])
            for lat in ["60s", "300s"]:
                for hold in ["15min", "60min"]:
                    x = lead_follow(lead, fol, 0.05, lat, hold, "60s", 0.0)
                    if len(x):
                        lfs.append(x.assign(game=r.slug, lg=r.lg, lat=lat, hold=hold))
        pd.concat(arbs).to_parquet(arb_path)
        pd.concat(lfs).to_parquet(lf_path)
    return pd.read_parquet(arb_path), pd.read_parquet(lf_path)


def bankroll_lead_follow(r: pd.DataFrame, start: float = 5000, stake: float = 0.05) -> tuple[float, float]:
    """Compound `stake` x equity per signal, capped at the entry print's size; returns final equity and max drawdown."""
    r = r.sort_values("entry")
    cost = np.where(r["gap"] > 0, r["p_in"], 1 - r["p_in"]).clip(0.01)
    eq, path = start, []
    for c, n, ret in zip(cost, r["size"], r["ret"]):
        eq += min(n, stake * eq / c) * ret
        path.append(eq)
    s = pd.Series(path)
    return eq, float((s / s.cummax() - 1).min())


def families() -> None:
    """Lead-follow and two-venue arbitrage per market family (built by event-markets/scripts/match2.py and xv2.py)."""
    lf, arb = pd.read_parquet(f"{DIR}/xv2_lf.parquet"), pd.read_parquet(f"{DIR}/xv2_arb.parquet")
    settle = pd.read_parquet(f"{DIR}/xv2_lf_settle.parquet")

    def stats(d, col="ret"):
        g = d.groupby("id")[col].mean()
        return pd.Series({"trades": len(d), "markets": len(g), "ret_c": 100 * d[col].mean(),
                          "t_mkt": g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 2 else np.nan})

    print("\n== By market family: follow the leader on the other venue, 5c+ gap")
    for th, lat, hold in [(0.05, "2s", "15min"), (0.05, "60s", "60min")]:
        x = lf[(lf["th"] == th) & (lf["lat"] == lat) & (lf["hold"] == hold)]
        print(f"-- {lat} late, {hold} hold, exit at the next opposite print")
        print(x.groupby(["family", "dir"]).apply(stats).round(2).unstack("dir").to_string())
    print("-- 60s late, held to settlement")
    print(settle.groupby(["family", "dir"]).apply(stats, col="ret_s").round(2).unstack("dir").to_string())
    print("\n== By market family: two-venue arbitrage (same contract, YES on one venue + NO on the other, 10s)")
    print(arb.groupby("family").agg(episodes=("profit", "size"), markets=("id", "nunique"),
                                    edge_c=("edge", lambda e: 100 * e.median()), dollars=("profit", "sum")).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--families", action="store_true", help="only the per-family Kalshi/Polymarket tables")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.families:
        families()
        return

    print("== Kalshi takers by category (cents per contract; makers earn the mirror before maker fees)")
    print(calibration_table(args.rebuild).to_string())

    if os.path.exists(f"{DIR}/arb_episodes.parquet"):
        a = pd.read_parquet(f"{DIR}/arb_episodes.parquet")
        a = a[a["fresh"] == "10s"].assign(net=lambda d: d["profit"] - 0.005 * d["n"])
        print("\n== Kalshi sum-to-$1 arbitrage (legs printed within 10s; fee rounding charged)")
        print(a.groupby(["series"])["net"].agg(["size", "sum"]).sort_values("sum").tail(8).round(0).to_string())
        print(f"total {len(a)} episodes, ${a['net'].sum():,.0f}, median edge {100 * a['edge'].median():.2f}c/set")

    arb, lf = cross_venue(args.rebuild)
    print(f"\n== Kalshi vs Polymarket two-venue arbitrage, {arb['game'].nunique()} games")
    print(arb.groupby("lg").agg(episodes=("profit", "size"), edge_c=("edge", "median"), dollars=("profit", "sum")).round(3).to_string())

    print("\n== Follow Kalshi on Polymarket when last prices differ by 5c+ (ret per share, Polymarket fee 0)")
    s = lf.groupby(["lat", "hold"]).apply(lambda d: pd.Series({"trades": len(d), "games": d["game"].nunique(),
                                                                "ret_c": 100 * d["ret"].mean(), "t_game": game_t(d)}))
    print(s.round(2).to_string())
    best = lf[(lf["lat"] == "60s") & (lf["hold"] == "60min")]
    eq, dd = bankroll_lead_follow(best)
    months = best["entry"].dt.strftime("%Y-%m")
    print(f"60s late, 60min hold: $5,000 -> ${eq:,.0f}, max DD {dd:.0%}, "
          f"{(best.groupby(months)['ret'].mean() > 0).mean():.0%} of months positive")
    if os.path.exists(f"{DIR}/xv2_lf.parquet"):
        families()


if __name__ == "__main__":
    main()
