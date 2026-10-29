"""Run every once-a-day paper logger, one after another; logs only, never trades.

    python paper_loggers/run_daily.py            # after 4pm ET on trading days (cron / Task Scheduler)
    python paper_loggers/run_daily.py --report   # every logger's scorecard

A logger that fails (network, missing key) is reported and skipped; the rest
still run. The always-on loggers (polymarket_iran_oil, kalshi_near_misses,
kalshi_sp_catchup) poll on their own; start them separately (README.md).
"""
import argparse
import importlib
import traceback

DAILY = ["treasury_auction_cycle", "etf_breakout_meta", "commodity_january", "odd_lot_tenders", "cef_rights_short",
         "letf_reversal", "pdufa_decision_short"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", nargs="+", choices=DAILY)
    args = ap.parse_args()
    failed = []
    for name in args.only or DAILY:
        print(f"\n== {name}")
        try:
            mod = importlib.import_module(name)
            mod.report() if args.report else mod.run()
        except SystemExit as e:  # e.g. a missing key
            print(f"  skipped: {e}")
            failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    if failed:
        print(f"\nfailed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
