"""Live order books on Kalshi and Polymarket US, keyless and read-only.

- Kalshi: api.elections.kalshi.com/trade-api/v2. `/markets?series_ticker=..`
  lists games; `/markets/{ticker}/orderbook` has YES and NO bids (a YES ask is
  $1 minus the best NO bid).
- Polymarket US (the CFTC-regulated venue open to US residents, a separate
  book from the international site): gateway.polymarket.us. `/v1/markets`
  lists markets, `/v1/markets/{slug}/bbo` gives the best bid and ask of the
  market's long side. Paths are from the official `polymarket-us` SDK (2.4.1).

Both hosts are blocked from the project's cloud sandbox; run this at home.
The parsers are pure functions so tests cover them offline.
"""
from __future__ import annotations

import re

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_US = "https://gateway.polymarket.us"
GAME_SERIES = ["KXNBAGAME", "KXNFLGAME", "KXNHLGAME", "KXMLBGAME", "KXNCAAFGAME", "KXNCAAMBGAME", "KXWNBAGAME",
               "KXEPLGAME", "KXUCLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXMLSGAME", "KXUFCFIGHT"]
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


def kalshi_book(ticker: str) -> tuple[float | None, float | None]:
    return kalshi_top(_get(f"{KALSHI}/markets/{ticker}/orderbook"))


def poly_us_book(slug: str) -> tuple[float | None, float | None]:
    return poly_us_top(_get(f"{POLY_US}/v1/markets/{slug}/bbo"))
