"""Live order books on Kalshi and Polymarket US, keyless and read-only.

- Kalshi: api.elections.kalshi.com/trade-api/v2. `/markets?series_ticker=..`
  lists games; `/markets/{ticker}/orderbook` has YES and NO bids (a YES ask is
  $1 minus the best NO bid).
- Polymarket US (the CFTC-regulated venue open to US residents, a separate
  book from the international site): gateway.polymarket.us. `/v1/markets`
  lists markets, `/v1/markets/{slug}/bbo` gives the best bid and ask of the
  market's long side. Paths are from the official `polymarket-us` SDK (2.4.1).

`pair_games` matches live Kalshi and Polymarket US markets on the same game
and side; `gap_followthrough` replays the Kalshi -> Polymarket rule from the
backtest on logged quotes, after the Polymarket US taker fee.

`NEAR_MISS_RULES`, `paper_entry` and `paper_pnl` hold the two Kalshi rules
from the 2026-10 alpha loop that passed in backtest but need a forward test
(paper_loggers/kalshi_near_misses.py).

Both hosts are blocked from the project's cloud sandbox; run this at home.
The parsers and replay are pure functions so tests cover them offline.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_US = "https://gateway.polymarket.us"
GAME_SERIES = ["KXNBAGAME", "KXNFLGAME", "KXNHLGAME", "KXMLBGAME", "KXNCAAFGAME", "KXNCAAMBGAME", "KXWNBAGAME",
               "KXEPLGAME", "KXUCLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXMLSGAME", "KXUFCFIGHT"]
POLY_US_SPORTS = 0.05  # Polymarket US taker fee on sports since Jul 2026, x p(1-p); makers pay nothing
_STOP = {"the", "fc", "cf", "sc", "ac", "afc", "club", "de"}
_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def _get(url: str, params: dict | None = None, timeout: float = 10.0) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _px(x) -> float | None:
    """Polymarket US price: {"value": "0.55", "currency": "USD"}, a bare number, or missing."""
    if x is None:
        return None
    v = x.get("value") if isinstance(x, dict) else x
    return None if v in (None, "") else float(v)


def name_tokens(text: str) -> frozenset:
    """Lower-case, accent-free word set of a team or player name, minus filler words."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return frozenset(w for w in re.findall(r"[a-z][a-z0-9]+", t) if w not in _STOP)


def name_score(a: frozenset, b: frozenset) -> float:
    """Overlap of two token sets as a share of the smaller one."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def kalshi_top(ob: dict) -> tuple[float | None, float | None]:
    """Best YES bid and ask in dollars from a Kalshi orderbook response (cents or the newer *_dollars fields)."""
    ob = ob.get("orderbook", ob)

    def best(side):
        lv = ob.get(f"{side}_dollars")
        if lv:
            return max(float(p) for p, _ in lv)
        lv = ob.get(side)
        return max(p for p, _ in lv) / 100 if lv else None
    yes, no = best("yes"), best("no")
    return yes, (None if no is None else round(1 - no, 4))


def poly_us_top(bbo: dict) -> tuple[float | None, float | None]:
    """Best long-side bid and ask from a Polymarket US /bbo response."""
    d = bbo.get("marketData", bbo)
    return _px(d.get("bestBid")), _px(d.get("bestAsk"))


def kalshi_game_key(event_ticker: str) -> tuple[str, str] | None:
    """KXNBAGAME-26JAN30DETGSW -> ('2026-01-30', 'DETGSW')."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:\d{4})?([A-Z]+)$", event_ticker)
    if not m or m.group(2) not in _MONTHS:
        return None
    return f"20{m.group(1)}-{_MONTHS[m.group(2)]:02d}-{m.group(3)}", m.group(4)


def poly_game_key(slug: str) -> tuple[str, tuple[str, ...]] | None:
    """nba-det-gsw-2026-01-30 (or with a trailing side) -> ('2026-01-30', ('DET', 'GSW'))."""
    m = re.match(r"[a-z0-9]+-([a-z0-9-]+?)-(\d{4}-\d{2}-\d{2})", slug or "")
    return (m.group(2), tuple(m.group(1).upper().split("-"))) if m else None


def kalshi_games(series: list[str] = GAME_SERIES) -> list[dict]:
    out = []
    for s in series:
        try:
            out += _get(f"{KALSHI}/markets", {"series_ticker": s, "status": "open", "limit": 1000}).get("markets", [])
        except requests.RequestException as e:
            print(f"  Kalshi {s}: {e}")
    return out


def poly_us_markets(max_pages: int = 20, page: int = 500) -> list[dict]:
    out = []
    for i in range(max_pages):
        ms = _get(f"{POLY_US}/v1/markets", {"active": "true", "closed": "false", "limit": page, "offset": i * page}).get("markets", [])
        out += ms
        if len(ms) < page:
            break
    return out


def kalshi_series_markets(series: str, status: str = "open") -> list[dict]:
    """Every market of one Kalshi series, following the cursor."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = _get(f"{KALSHI}/markets", params)
        out += r.get("markets", [])
        cursor = r.get("cursor")
        if not cursor or not r.get("markets"):
            return out


def kalshi_market(ticker: str) -> dict:
    return _get(f"{KALSHI}/markets/{ticker}").get("market", {})


def kalshi_mention_series() -> list[str]:
    """Series tickers Kalshi lists under its Mentions category (empty if the endpoint fails)."""
    try:
        ss = _get(f"{KALSHI}/series", {"category": "Mentions"}).get("series", [])
    except requests.RequestException:
        return []
    return [x["ticker"] for x in ss if x.get("ticker")]


def market_top(m: dict) -> tuple[float | None, float | None]:
    """Best YES bid and ask in dollars from a Kalshi market object (cents or the newer *_dollars fields)."""
    def px(k):
        v = m.get(f"{k}_dollars")
        if v not in (None, ""):
            return float(v)
        v = m.get(k)
        return None if v in (None, "") else v / 100
    bid, ask = px("yes_bid"), px("yes_ask")
    return (bid if bid else None), (ask if ask and ask < 1 else None)


# Alpha-loop near misses (batch 4-5, /mnt/project-files/alpha-hunt/): taker entries, price = what the taker pays.
# Mentions: buy NO at 20-80c (+3.4c/contract holdout, t 5.7). ATP men's matches: buy YES at 80-95c (+2.6c, t 6.4).
NEAR_MISS_RULES = {
    "mentions_no": {"side": "no", "lo": 0.20, "hi": 0.80},
    "atp_yes": {"side": "yes", "lo": 0.80, "hi": 0.95, "series": ["KXATPMATCH", "KXATPCHALLENGERMATCH"]},
}


def paper_entry(rule: str, yes_bid: float | None, yes_ask: float | None) -> tuple[str, float] | None:
    """(side, price) a taker would pay under `rule` at this quote, or None if the quote is outside its band."""
    r = NEAR_MISS_RULES[rule]
    if r["side"] == "yes":
        price = yes_ask
    else:
        price = None if yes_bid is None else round(1 - yes_bid, 4)
    if price is None or not r["lo"] <= price < r["hi"]:
        return None
    return r["side"], price


def paper_pnl(side: str, price: float, result: str, contracts: int = 10, rate: float = 0.07) -> float:
    """Net P&L per contract of a settled taker buy, with Kalshi's fee rounded up to the cent per order."""
    fee = np.ceil(round(rate * contracts * price * (1 - price) * 100, 6)) / 100 / contracts
    return float(result == side) - price - fee


def kalshi_book(ticker: str) -> tuple[float | None, float | None]:
    return kalshi_top(_get(f"{KALSHI}/markets/{ticker}/orderbook"))


def poly_us_book(slug: str) -> tuple[float | None, float | None]:
    return poly_us_top(_get(f"{POLY_US}/v1/markets/{slug}/bbo"))


def pair_games(kalshi: list[dict], poly: list[dict]) -> list[dict]:
    """Pair live Kalshi game markets with Polymarket US markets on the same game and side.

    Games match on date and team codes (Kalshi KXNBAGAME-26JAN30DETGSW vs
    Polymarket slug nba-det-gsw-2026-01-30, either team order). The side
    matches on the Polymarket team abbreviation or name against the Kalshi
    ticker suffix and yes_sub_title. If the Polymarket market's long side is
    the opponent, `flip` is True and its prices become 1 - price.
    Returns kalshi_ticker, poly_slug, flip.
    """
    games: dict = {}
    for m in poly:
        key = poly_game_key(m.get("slug", "")) or poly_game_key(m.get("eventSlug", ""))
        if key:
            games.setdefault((key[0], "".join(key[1])), []).append(m)
            games.setdefault((key[0], "".join(reversed(key[1]))), []).append(m)

    def side_tokens(m):
        t = m.get("team") or {}
        abbr = {str(t.get(k, "")).upper() for k in ("abbreviation", "displayAbbreviation") if t.get(k)}
        name = " ".join(str(x) for x in (t.get("name"), t.get("alias"), m.get("outcome")) if x)
        tok = frozenset({"tie"}) if re.search(r"\b(draw|tie)\b", name, re.I) else name_tokens(name)
        return abbr, tok

    def same(kt, ksub, m):
        abbr, tok = side_tokens(m)
        code = kt.rsplit("-", 1)[-1].upper()
        ktok = frozenset({"tie"}) if code == "TIE" or str(ksub).lower() in ("tie", "draw") else name_tokens(ksub)
        return code in abbr or name_score(ktok, tok) >= 0.67

    by_event: dict = {}
    for k in kalshi:
        by_event.setdefault(k.get("event_ticker", ""), []).append(k)
    out = []
    for ev, ks in by_event.items():
        key = kalshi_game_key(ev)
        cands = games.get(key, []) if key else []
        for k in ks:
            hit = [m for m in cands if same(k["ticker"], k.get("yes_sub_title", ""), m)]
            if len(hit) == 1:
                out.append({"kalshi_ticker": k["ticker"], "poly_slug": hit[0]["slug"], "flip": False})
                continue
            # A single Polymarket market per game whose long side is the other team.
            opp = [o for o in ks if o is not k]
            if len(cands) == 1 and len(ks) == 2 and same(opp[0]["ticker"], opp[0].get("yes_sub_title", ""), cands[0]):
                out.append({"kalshi_ticker": k["ticker"], "poly_slug": cands[0]["slug"], "flip": True})
    return out


def gap_followthrough(quotes: pd.DataFrame, threshold: float = 0.05, latency: str = "60s", hold: str = "60min",
                      rate: float = POLY_US_SPORTS) -> pd.DataFrame:
    """Replay the Kalshi -> Polymarket rule on logged top-of-book quotes.

    quotes: time, kalshi_ticker, k_bid, k_ask, p_bid, p_ask (Polymarket
    already oriented to the Kalshi side). A signal is a poll where the Kalshi
    mid moved and the mid gap first reaches `threshold`. Enter on Polymarket at
    the first poll `latency` later, crossing the spread (buy the long side at
    the ask if Kalshi is higher, else the short side at 1 - bid); exit by
    crossing back at the first poll `hold` after entry (or the last poll).
    ret is per share after the taker fee on both legs; `toward` is how far the
    Polymarket mid moved toward Kalshi from signal to exit. One trade at a
    time per market.
    """
    lat, hd = pd.Timedelta(latency), pd.Timedelta(hold)
    rows = []
    for tk, q in quotes.sort_values("time").groupby("kalshi_ticker"):
        q = q.reset_index(drop=True)
        km, pm = (q["k_bid"] + q["k_ask"]) / 2, (q["p_bid"] + q["p_ask"]) / 2
        gap = km - pm
        sig = (gap.abs() >= threshold) & (gap.shift().abs() < threshold) & (km != km.shift())
        t = q["time"].dt.as_unit("ns").astype("int64").to_numpy()
        free_at = t[0]
        for i in np.flatnonzero(sig.to_numpy()):
            if t[i] < free_at:
                continue
            e = int(np.searchsorted(t, t[i] + lat.value))
            if e >= len(q):
                break
            x = min(int(np.searchsorted(t, t[e] + hd.value)), len(q) - 1)
            up = gap.iat[i] > 0
            p_in = q["p_ask"].iat[e] if up else 1 - q["p_bid"].iat[e]
            p_out = q["p_bid"].iat[x] if up else 1 - q["p_ask"].iat[x]
            if not (np.isfinite(p_in) and np.isfinite(p_out)) or x <= e:
                continue
            fee = rate * (p_in * (1 - p_in) + p_out * (1 - p_out))
            rows.append({"kalshi_ticker": tk, "signal": q["time"].iat[i], "entry": q["time"].iat[e], "exit": q["time"].iat[x],
                         "gap": gap.iat[i], "p_in": p_in, "p_out": p_out, "ret": p_out - p_in - fee,
                         "toward": np.sign(gap.iat[i]) * (pm.iat[x] - pm.iat[i])})
            free_at = t[x]
    return pd.DataFrame(rows, columns=["kalshi_ticker", "signal", "entry", "exit", "gap", "p_in", "p_out", "ret", "toward"])
