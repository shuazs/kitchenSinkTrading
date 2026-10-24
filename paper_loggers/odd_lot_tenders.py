"""Paper-trade odd-lot fixed-price tender offers; logs only, never places orders.

    python paper_loggers/odd_lot_tenders.py            # daily: new SC TO-I offers + settle finished ones
    python paper_loggers/odd_lot_tenders.py --report   # net return per event so far

Rule (examples/run_alpha_batch8.py `odd_lot_tenders`, batch 8): when a
company (not a fund) files an SC TO-I issuer tender at a fixed price above
the market that gives odd lots (< 100 shares) priority over proration, buy
99 shares at the next close, tender them all, and receive the offer price
(or the final price in the results amendment). Skip Dutch auctions, funds,
exchange offers and Canadian-dollar offers. Use a broker that charges no
tender fee.

Each run searches EDGAR full-text search for SC TO-I filings mentioning odd
lots in the last --days days, parses them with backtester/data/tender_offers.py
and logs every filing (kept or skipped, with the reason) to
~/.cache/kitchensink/paper/odd_lot_tenders/events.csv. After expiry it reads
the SC TO-I/A amendments for the final price or a termination and fills in
the paper result: max(offer, close at expiry) per share, or the expiry close
if the offer was terminated, less a $1 commission. Needs SEC_USER_AGENT in
keys.env. Graduates after 6 events at +3% or better per event.
"""
import argparse

import pandas as pd

from _common import closes, log_dir, read_log, sec_get, today, tstat, write_log
from backtester.data.tender_offers import clean_text, final_price, parse_offer

EFTS = "https://efts.sec.gov/LATEST/search-index"
LOG = log_dir("odd_lot_tenders") / "events.csv"
FIELDS = ["file_date", "adsh", "cik", "company", "ticker", "status", "fixed", "expiry", "entry_date", "entry",
          "premium", "final", "terminated", "exit_close", "net"]
SHARES, COMMISSION, BAR = 99, 1.0, 0.03


def search(start: pd.Timestamp, end: pd.Timestamp, q: str, forms: str) -> list[dict]:
    """EDGAR full-text search hits (one per matching document), all pages."""
    out, frm = [], 0
    while True:
        h = sec_get(EFTS, {"q": q, "forms": forms, "dateRange": "custom", "startdt": f"{start:%Y-%m-%d}",
                           "enddt": f"{end:%Y-%m-%d}", "from": frm}).json()["hits"]["hits"]
        out += h
        frm += len(h)
        if len(h) < 100:
            return out


def doc_url(hit: dict) -> str:
    adsh, name = hit["_id"].split(":", 1)
    return f"https://www.sec.gov/Archives/edgar/data/{int(hit['_source']['ciks'][0])}/{adsh.replace('-', '')}/{name}"


def ticker_map() -> dict[int, str]:
    j = sec_get("https://www.sec.gov/files/company_tickers_exchange.json").json()
    out: dict[int, str] = {}
    for cik, _, tk, _ in j["data"]:  # prefer the common share over preferreds/warrants (RIV over RIV-PA)
        if int(cik) not in out or ("-" in out[int(cik)] and "-" not in tk):
            out[int(cik)] = tk
    return out


def amendments(cik: int, after: pd.Timestamp) -> list[str]:
    """Texts of SC TO-I/A filings by `cik` after `after` (the results usually come in the last one)."""
    rec = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    out = []
    for form, adsh, d, doc in zip(rec["form"], rec["accessionNumber"], rec["filingDate"], rec["primaryDocument"]):
        if form == "SC TO-I/A" and pd.Timestamp(d) > after:
            out.append(clean_text(sec_get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{doc}").text))
    return out


def scan(log: pd.DataFrame, days: int) -> pd.DataFrame:
    day = today()
    hits = search(day - pd.Timedelta(days=days), day, '"odd lot" OR "odd lots" OR "odd-lot"', "SC TO-I")
    by_filing: dict[str, list[dict]] = {}
    for h in hits:
        by_filing.setdefault(h["_id"].split(":")[0], []).append(h)
    new = sorted((a for a in by_filing if a not in set(log.adsh)), key=lambda a: by_filing[a][0]["_source"]["file_date"])
    if not new:
        return log
    tk = ticker_map()
    active = set(log.ticker[log.status.isin(["open", "entered"])])
    rows = []
    for adsh in new:
        hs = by_filing[adsh]
        src = hs[0]["_source"]
        cik = int(src["ciks"][0])
        t = " ".join(clean_text(sec_get(doc_url(h)).text) for h in hs[:3])
        d = parse_offer(t)
        tick = tk.get(cik, "")
        why = ("no ticker" if not tick or "-" in tick else "fund (NAV)" if "net asset value" in t else
               "Dutch auction" if d["dutch"] else "no fixed price" if d["fixed"] is None else
               "no odd-lot priority" if not d["odd"] else "exchange/other security" if d["other"] else
               "Canadian dollars" if d["cad"] else "")
        if not why and tick in active:
            why = "repeat filing for an offer already logged"
        if not why:
            active.add(tick)
        rows.append({"file_date": src["file_date"], "adsh": adsh, "cik": cik, "company": src["display_names"][0][:60],
                     "ticker": tick, "status": f"skip: {why}" if why else "open", "fixed": d["fixed"],
                     "expiry": pd.to_datetime(d["exp"], format="%B %d %Y", errors="coerce")})
        print(f"{src['file_date']} {tick or '?':<6} {rows[-1]['status']:<28} {src['display_names'][0][:50]}")
    return pd.concat([log, pd.DataFrame(rows)], ignore_index=True)


def update(log: pd.DataFrame) -> pd.DataFrame:
    day = today()
    live = log.index[log.status.isin(["open", "entered"])]
    if not len(live):
        return log
    px = closes(sorted(set(log.loc[live, "ticker"])), pd.to_datetime(log.loc[live, "file_date"]).min() - pd.Timedelta(days=5))
    for i in live:
        tk, filed = log.at[i, "ticker"], pd.Timestamp(log.at[i, "file_date"])
        p = px[tk].dropna() if tk in px else pd.Series(dtype=float)
        if log.at[i, "status"] == "open":
            j = p.index.searchsorted(filed, side="right")
            if j >= len(p):
                if day > filed + pd.Timedelta(days=7):
                    log.at[i, "status"] = "skip: no Yahoo price"
                continue
            entry, fixed = float(p.iloc[j]), float(log.at[i, "fixed"])
            prem = fixed / entry - 1
            log.loc[i, ["entry_date", "entry", "premium"]] = [p.index[j], round(entry, 4), round(prem, 4)]
            ok = 0 < prem <= 0.3
            log.at[i, "status"] = "entered" if ok else f"skip: premium {prem:+.1%}"
            if ok:
                print(f"  PAPER BUY {SHARES} {tk} at {entry:.2f} (offer {fixed:.2f}, {prem:+.1%}); tender all shares")
            continue
        exp = pd.Timestamp(log.at[i, "expiry"]) if pd.notna(log.at[i, "expiry"]) and log.at[i, "expiry"] != "" else \
            pd.Timestamp(log.at[i, "entry_date"]) + pd.offsets.BDay(22)
        if day < exp + pd.Timedelta(days=7) or p.empty:
            continue  # results amendment usually lands within a week of expiry
        clear, term = final_price(amendments(int(log.at[i, "cik"]), filed))
        k = min(p.index.searchsorted(exp, side="right") - 1, len(p) - 1)
        close_exp, entry, fixed = float(p.iloc[k]), float(log.at[i, "entry"]), float(log.at[i, "fixed"])
        pay = clear if clear and fixed <= clear <= 1.25 * fixed else fixed
        out = close_exp if term else max(pay, close_exp)
        net = out / entry - 1 - COMMISSION / (SHARES * entry)
        log.loc[i, ["final", "terminated", "exit_close", "net", "status"]] = [pay, term, round(close_exp, 4), round(net, 4), "done"]
        print(f"  DONE {tk}: paid {out:.2f} vs entry {entry:.2f} -> {net:+.2%} net{' (terminated)' if term else ''}")
    return log


def run(days: int = 14) -> None:
    write_log(update(scan(read_log(LOG, FIELDS), days)), LOG)


def report() -> None:
    log = read_log(LOG, FIELDS)
    net = pd.to_numeric(log.net, errors="coerce").dropna()
    print(f"{len(log)} filings seen, {log.status.eq('entered').sum()} open trades, {len(net)} done: "
          f"mean {net.mean():+.2%} per event, t {tstat(net):.2f}; graduates at >= {BAR:+.0%} over 6+ events")
    print(log[~log.status.str.startswith("skip")].tail(15).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="look back this many days for new filings")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else run(args.days)


if __name__ == "__main__":
    main()
