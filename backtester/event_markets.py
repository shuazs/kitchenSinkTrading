"""Trading non-financial prediction markets (sports, politics, weather, culture).

Kalshi and Polymarket contracts on events with no tradeable underlying. Tests:

* calibration: per price bucket and category, what a taker earns buying a
  side at the printed price and holding to settlement, after the Kalshi fee
  (makers earn the mirror image, before any maker fee);
* fade longshots: buy NO on cheap YES contracts at real NO-taker prints,
  fills capped at the printed size, a $5,000 bankroll and fixed stake;
* bracket arbitrage: mutually exclusive brackets (weather highs, winners)
  whose YES asks, rebuilt from YES-taker prints, sum to less than $1;
* cross-venue: the same game on Kalshi and Polymarket, buy YES on one and
  NO on the other when the two asks sum to less than $1 after fees;
* jumps: after a big hourly move, does the price keep going or revert.

Inputs are Kalshi prints (ticker, count, yes_price in cents, taker_side,
created_time) joined to markets (event_ticker, series, result, close_time),
and Polymarket fills (time, market_id, YES price, usd_amount, side).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

KALSHI_TAKER = 0.07
POLY_US_SPORTS = 0.05  # Polymarket US taker fee on sports since Jul 2026, x p(1-p); makers pay nothing
BUCKETS = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0001]

_FIN = (r"^(KX)?(INX|NASDAQ|BTC|ETH|SOL|XRP|DOGE|FED|CPI|GDP|PAYROLL|U3|JOBLESS|PCE|PPI|RATECUT|TNOTE|AAAGAS|WTI|OIL|"
        r"GOLD|SILVER|EURUSD|USDJPY|GBPUSD|MORTGAGE|RECSS|TARIFF|LARGETARIFF|EGGS|TRUFEGGS|RETAIL|HOUSE(PRICE|ST)|DOW|"
        r"SPX|SHIB|COIN|STOCK|TSLA|NVDA|AAPL|EARN|IPO|ECON|INFL|UNEMP|TREAS|10Y|2Y|YIELD|VIX|CRYPTO|LTC|ADA|BNB|COPPER|"
        r"NATGAS|CORN|WHEAT)")
_SPORT = (r"(NFL|NBA|MLB|NHL|NCAA|WNBA|MLS|EPL|UCL|UEL|ATP|WTA|UFC|PGA|LPGA|LIV|F1|NASCAR|INDY|BOXING|MASTERS|USOPEN|"
          r"THEOPEN|WIMBLE|WMENS|WWOMEN|USOM|USOW|FOM|FOW|AOM|AOW|SB$|MARMAD|HEISMAN|LALIGA|SERIEA|BUNDES|LIGUE|EREDIV|"
          r"PREMIER|FIFA|WC|AFCON|COPA|EURO|EFL|ALEAGUE|NBL|CBA|KBO|NPB|CSGO|LOL|DOTA|VALORANT|ESPORT|CRICKET|IPL|"
          r"RUGBY|SAUDIPL|CHALLENGER|UNITEDCUP|DAVIS|RYDER|TOUR|GOLF|TENNIS|SOCCER|DERBY|STANLEY|COACH|DRAFT|MVP|"
          r"OLYMPIC|MATCH|GAME|SPREAD|TOTAL|SERIES|FIGHT|RACE|BOWL|CUP|LEAGUE|CHAMP|ANYTD|FIRSTTD|2TD|YDS|GOAL|TEAM)")
_WEATHER = r"(HIGH|LOW|TEMP|RAIN|SNOW|HURR|STORM|TORNADO|WEATHER|HEAT|COLD|CLIMATE|PRECIP)"
_POLITICS = (r"(PRES|SENATE|HOUSE|GOV|MAYOR|ELECT|POPVOTE|ECMOV|CLOSESTSTATE|POWER|NOM|PARTY|CONGRESS|SCOTUS|CABINET|"
             r"IMPEACH|APPROV|PRIMARY|VOTE|POLL|SPEAKER|TRUMP|BIDEN|HARRIS|VANCE|MUSK|PARLIAMENT|PM|MINISTER|LEADER|"
             r"SHUT|PARDON|COURT|INDICT|MAMDANI|CUOMO|NEWSOM|NATO|WAR|CEASE|IRAN|ISRAEL|UKRAINE|RUSSIA|CHINA|PUTIN|POPE)")
_CULTURE = (r"(SPOTIFY|BILLBOARD|OSCAR|GRAMMY|EMMY|GOLDENGLOBE|NETFLIX|ROTTEN|RT|BOXOFFICE|MOVIE|ALBUM|SONG|ARTIST|"
            r"TAYLOR|TIKTOK|APP|TIME|SNL|SHOW|AWARD|EUROVISION|LLM|AI|SPACEX|TSA|MEASLES)")
_PM_FIN = re.compile(
    r"bitcoin|btc|ethereum|eth |solana|xrp|crypto|fed |fed\b|interest rate|s&p|nasdaq|stock|price of|price will|token|"
    r"fdv|market cap|airdrop|microstrategy|dogecoin|doge |up or down|opens? (up|down)|close (above|below)|tesla|"
    r"nvidia|apple|largest company|gold|oil|crude|inflation|cpi|gdp|recession|tariff|treasury|yield|usdt|tether|"
    r"coinbase|binance|etf|memecoin|pump|ipo", re.I)


def kalshi_category(series: str) -> str:
    """Rough category of a Kalshi series ticker: sports, politics, weather, culture, mentions, finance, parlay, other."""
    s = series.upper()
    if s.startswith("KXMVE"):
        return "parlay"
    if re.search(_FIN, s):
        return "finance"
    if "MENTION" in s or re.search(r"(SAY|TWEET|POST)", s):
        return "mentions"
    if re.search(_WEATHER, s) and not re.search(r"HIGHEST|HIGHLIGHT", s):
        return "weather"
    if re.search(_SPORT, s):
        return "sports"
    if re.search(_POLITICS, s):
        return "politics"
    if re.search(_CULTURE, s):
        return "culture"
    return "other"


def is_financial_text(text: str) -> bool:
    """True if a Polymarket question/event title is about prices, rates, crypto or macro."""
    return bool(_PM_FIN.search(text))


def kalshi_fee(p, rate: float = KALSHI_TAKER):
    """Kalshi taker fee per contract at price p (0-1), before rounding to the cent."""
    p = np.asarray(p, dtype=float)
    return rate * p * (1 - p)


def taker_prints(trades: pd.DataFrame, markets: pd.DataFrame, rate: float = KALSHI_TAKER) -> pd.DataFrame:
    """One row per Kalshi print with the taker's side, price paid, outcome and net P&L per contract.

    `markets` needs ticker, event_ticker, result ('yes'/'no'), close_time and,
    optionally, cat. Unsettled or scalar markets are dropped.
    """
    cols = [c for c in ["ticker", "event_ticker", "series", "cat", "result", "close_time"] if c in markets]
    t = trades.merge(markets[cols], on="ticker", how="inner")
    t = t[t["result"].isin(["yes", "no"])].copy()
    yes = t["taker_side"].eq("yes").to_numpy()
    t["price"] = np.where(yes, t["yes_price"] / 100, 1 - t["yes_price"] / 100)
    t["won"] = (t["result"] == t["taker_side"]).astype(float)
    t["fee"] = kalshi_fee(t["price"], rate)
    t["net"] = t["won"] - t["price"] - t["fee"]
    t["tau_h"] = (t["close_time"] - t["created_time"]).dt.total_seconds() / 3600
    return t


def calibration(prints: pd.DataFrame, by=("cat",), buckets=BUCKETS) -> pd.DataFrame:
    """Contract-weighted hit rate vs price per bucket of the taker's price.

    taker_net is per contract after the taker fee; maker_gross is what the
    other side made per contract before any maker fee. t is across events
    (each event's average taker net), so correlated prints don't inflate it.
    """
    p = prints.assign(bucket=pd.cut(prints["price"], buckets, right=False))
    w = p["count"]
    p = p.assign(cw=w, pw=p["price"] * w, ww=p["won"] * w, nw=p["net"] * w)
    keys = list(by) + ["bucket"]
    ev = p.groupby(keys + ["event_ticker"], observed=True)[["cw", "pw", "ww", "nw"]].sum()
    ev["evnet"] = ev["nw"] / ev["cw"]
    g = ev.groupby(level=keys, observed=True)
    out = g[["cw", "pw", "ww", "nw"]].sum()
    out = pd.DataFrame({
        "contracts": out["cw"], "events": g.size(),
        "price": out["pw"] / out["cw"], "hit": out["ww"] / out["cw"],
        "taker_net": out["nw"] / out["cw"],
    })
    out["maker_gross"] = out["price"] - out["hit"]
    sd = g["evnet"].std()
    out["t"] = g["evnet"].mean() / (sd / np.sqrt(out["events"]))
    return out.reset_index()


def exclusive_events(markets: pd.DataFrame, min_share: float = 0.98) -> pd.Series:
    """Event tickers whose markets are mutually exclusive and exhaustive (exactly one settles YES).

    A series counts when at least `min_share` of its settled events have exactly
    one YES market; within it, only events with exactly one YES are kept.
    Returns event_ticker -> number of markets.
    """
    m = markets[markets["result"].isin(["yes", "no"])]
    ev = m.groupby("event_ticker").agg(series=("series", "first"), n=("ticker", "size"),
                                       yes=("result", lambda r: int((r == "yes").sum())))
    ev = ev[ev["n"] >= 2]
    share = ev.groupby("series")["yes"].apply(lambda y: float((y == 1).mean()))
    ok = ev[ev["series"].map(share).ge(min_share) & ev["yes"].eq(1)]
    return ok["n"]


def bracket_arb(prints: pd.DataFrame, n_markets: int, side: str = "yes", fresh: str = "5min",
                rate: float = KALSHI_TAKER) -> pd.DataFrame:
    """Moments when buying `side` on every market of one exclusive event costs less than it must pay.

    Asks are rebuilt from taker prints on that side (the last price a taker
    paid per market), and only count while every leg has printed within
    `fresh`. Buying YES on all n pays $1; buying NO on all pays $(n-1).
    Rows: time, cost, fee, edge (per set, after fees), size (smallest last
    print among the legs, in sets).
    """
    p = prints[prints["taker_side"] == side].sort_values("created_time")
    if p["ticker"].nunique() < n_markets:
        return pd.DataFrame(columns=["time", "cost", "fee", "edge", "size"])
    price = p["yes_price"] / 100 if side == "yes" else 1 - p["yes_price"] / 100
    p = p.assign(price=price.to_numpy(), seen=p["created_time"])
    last = p.pivot_table(index="created_time", columns="ticker", values=["price", "seen", "count"], aggfunc="last").ffill()
    last = last.dropna()
    if last.empty:
        return pd.DataFrame(columns=["time", "cost", "fee", "edge", "size"])
    px = last["price"].to_numpy(float)
    cost = px.sum(axis=1)
    fee = kalshi_fee(px, rate).sum(axis=1)
    payout = 1.0 if side == "yes" else n_markets - 1.0
    oldest = last["seen"].min(axis=1)
    ok = (last.index - oldest) <= pd.Timedelta(fresh)
    out = pd.DataFrame({"time": last.index, "cost": cost, "fee": fee, "edge": payout - cost - fee,
                        "size": last["count"].min(axis=1).to_numpy()})
    return out[ok.to_numpy()].reset_index(drop=True)


def arb_episodes(arb: pd.DataFrame, min_edge: float = 0.0, gap: str = "10min", legs: int = 0) -> pd.DataFrame:
    """Collapse consecutive positive-edge rows into episodes; one fill of the first row's size per episode.

    Kalshi rounds each order's fee up to the cent, about half a cent per leg
    on average; `legs` charges that ($0.005 per leg) against each episode.
    """
    a = arb[arb["edge"] > min_edge].sort_values("time")
    if a.empty:
        return a.assign(profit=pd.Series(dtype=float))
    new = a["time"].diff().gt(pd.Timedelta(gap)) | a["time"].diff().isna()
    first = a[new.to_numpy()].copy()
    first["profit"] = first["edge"] * first["size"] - 0.005 * legs
    return first.reset_index(drop=True)


def first_entries(prints: pd.DataFrame, lo: float, hi: float, min_tau_h: float = 1.0) -> pd.DataFrame:
    """First taker print per (market, side) whose price is in [lo, hi) at least `min_tau_h` hours before close.

    That print is our fill: same side, same price, at most its printed size.
    `prints` comes from `taker_prints`.
    """
    p = prints[(prints["price"] >= lo) & (prints["price"] < hi) & (prints["tau_h"] >= min_tau_h)]
    p = p.sort_values("created_time")
    return p.drop_duplicates(["ticker", "taker_side"], keep="first").reset_index(drop=True)


def bankroll(entries: pd.DataFrame, start: float = 5000.0, stake: float = 0.02, rate: float = KALSHI_TAKER) -> pd.DataFrame:
    """Hold-to-settlement ledger: each entry stakes `stake` x current equity, capped by the print's size and by cash.

    Cash is tied up from entry to close_time; P&L lands at close. One row per
    filled entry with contracts, cost, fee, pnl and equity after settlement.
    """
    e = entries.sort_values("created_time").reset_index(drop=True)
    events = [(t, 0, i) for i, t in enumerate(e["created_time"])] + [(t, 1, i) for i, t in enumerate(e["close_time"])]
    events.sort(key=lambda x: (x[0], -x[1]))  # settle before opening at the same timestamp
    cash, equity, open_cost = start, start, {}
    rows = []
    for t, kind, i in events:
        r = e.iloc[i]
        if kind == 0:
            budget = min(stake * equity, cash)
            unit = r["price"] + float(kalshi_fee(r["price"], rate))
            n = int(min(budget // unit, r["count"]))
            if n <= 0:
                continue
            cost, fee = n * r["price"], n * float(kalshi_fee(r["price"], rate))
            cash -= cost + fee
            open_cost[i] = (n, cost, fee)
        elif i in open_cost:
            n, cost, fee = open_cost.pop(i)
            pnl = n * r["won"] - cost - fee
            cash += n * r["won"]
            equity += pnl
            rows.append({"ticker": r["ticker"], "side": r["taker_side"], "entry": r["created_time"], "close": t,
                         "price": r["price"], "contracts": n, "cost": cost, "fee": fee, "pnl": pnl, "equity": equity})
    return pd.DataFrame(rows)


def cross_arb(leg1: pd.DataFrame, leg2: pd.DataFrame, fresh: str = "10s",
              rate1: float = KALSHI_TAKER, rate2: float = 0.0) -> pd.DataFrame:
    """Two-venue arbitrage: buy outcome A on one venue and not-A on the other for less than $1.

    Each leg is a frame of time, price (what a taker paid for that leg's
    outcome) and size (contracts or shares). At every print on either leg the
    other leg's last print must be no older than `fresh`. Fee rates are
    Kalshi-style rate * p * (1 - p) per contract (Polymarket: 0 by default).
    Rows: time, p1, p2, edge per $1 payout after fees, size.
    """
    a = leg1[["time", "price", "size"]].astype({"time": "datetime64[ns, UTC]"}).sort_values("time")
    b = leg2[["time", "price", "size"]].astype({"time": "datetime64[ns, UTC]"}).sort_values("time")
    tol = pd.Timedelta(fresh)
    x = pd.merge_asof(a, b, on="time", direction="backward", tolerance=tol, suffixes=("1", "2"))
    y = pd.merge_asof(b, a, on="time", direction="backward", tolerance=tol, suffixes=("2", "1"))
    out = pd.concat([x, y], ignore_index=True).dropna().sort_values("time")
    out["edge"] = 1 - out["price1"] - out["price2"] - kalshi_fee(out["price1"], rate1) - kalshi_fee(out["price2"], rate2)
    out["size"] = np.minimum(out["size1"], out["size2"])
    return out.rename(columns={"price1": "p1", "price2": "p2"})[["time", "p1", "p2", "edge", "size"]].reset_index(drop=True)


def snapshot_quotes(prints: pd.DataFrame, at: pd.Series, fresh: str = "30min") -> pd.DataFrame:
    """Last YES ask and bid per market just before a time, rebuilt from taker prints.

    `prints`: market, time, yes (YES price 0-1), taker_side ('yes' = taker
    bought YES, so the price was an ask; 'no' = a bid). `at`: market -> time.
    Only prints in (at - fresh, at] count. Rows: market, ask, bid, mid.
    """
    p = prints.assign(at=prints["market"].map(at)).dropna(subset=["at"])
    p = p[(p["time"] <= p["at"]) & (p["time"] > p["at"] - pd.Timedelta(fresh))].sort_values("time")
    last = p.groupby(["market", "taker_side"])["yes"].last().unstack()
    out = pd.DataFrame({"ask": last.get("yes"), "bid": last.get("no")}).dropna()
    out["mid"] = (out["ask"] + out["bid"]) / 2
    return out.rename_axis("market").reset_index()


def favourite_bets(quotes: pd.DataFrame, yes_won: pd.Series, lo: float = 0.5, hi: float = 1.0,
                   rate: float = 0.0) -> pd.DataFrame:
    """Buy whichever side is the favourite (mid in [lo, hi) for that side) at its ask; hold to settlement.

    YES favourite pays the YES ask; NO favourite pays 1 - YES bid. `rate` is a
    Kalshi-style fee rate (0 for Polymarket). ret is per $ staked.
    """
    q = quotes.copy()
    yes_fav = q["mid"] >= 0.5
    q["side"] = np.where(yes_fav, "yes", "no")
    q["fav_mid"] = np.where(yes_fav, q["mid"], 1 - q["mid"])
    q["price"] = np.where(yes_fav, q["ask"], 1 - q["bid"])
    q = q[(q["fav_mid"] >= lo) & (q["fav_mid"] < hi) & (q["price"] < 1)]
    won = q["market"].map(yes_won).astype(float)
    q["won"] = np.where(q["side"] == "yes", won, 1 - won)
    q["fee"] = kalshi_fee(q["price"], rate)
    q["ret"] = (q["won"] - q["price"] - q["fee"]) / (q["price"] + q["fee"])
    return q.reset_index(drop=True)


def lead_follow(lead: pd.DataFrame, follow: pd.DataFrame, threshold: float = 0.03, latency: str = "2s",
                hold: str = "5min", fresh: str = "60s", rate: float = 0.0) -> pd.DataFrame:
    """Trade the follower venue toward the leader when their prices for the same outcome disagree.

    lead: time, yes (last-trade YES price on the leading venue). follow: time,
    yes, taker_side ('yes' = print at the ask, 'no' = at the bid), size.
    At each lead print, the follower mid is the average of its last ask and
    bid prints, both within `fresh`. If lead - mid >= threshold, buy YES on the
    follower at its first ask print after `latency`, sell at the first bid
    print after `hold`; mirror image when the gap is <= -threshold. One trade
    at a time. ret is per contract after a Kalshi-style fee `rate` on both legs.
    """
    tol = pd.Timedelta(fresh)
    f = follow.astype({"time": "datetime64[ns, UTC]"}).sort_values("time")
    asks = f[f["taker_side"] == "yes"][["time", "yes", "size"]]
    bids = f[f["taker_side"] == "no"][["time", "yes", "size"]]
    sig = lead[["time", "yes"]].astype({"time": "datetime64[ns, UTC]"}).sort_values("time").rename(columns={"yes": "lead"})
    sig = pd.merge_asof(sig, asks.rename(columns={"yes": "ask"})[["time", "ask"]], on="time", tolerance=tol)
    sig = pd.merge_asof(sig, bids.rename(columns={"yes": "bid"})[["time", "bid"]], on="time", tolerance=tol)
    sig["gap"] = sig["lead"] - (sig["ask"] + sig["bid"]) / 2
    sig = sig[sig["gap"].abs() >= threshold]
    rows, busy_until = [], None
    lat, hld = pd.Timedelta(latency), pd.Timedelta(hold)
    for t, gap in zip(sig["time"], sig["gap"]):
        if busy_until is not None and t < busy_until:
            continue
        side_in, side_out = (asks, bids) if gap > 0 else (bids, asks)
        i = side_in["time"].searchsorted(t + lat)
        if i >= len(side_in):
            break
        t_in, p_in, n = side_in["time"].iloc[i], side_in["yes"].iloc[i], side_in["size"].iloc[i]
        j = side_out["time"].searchsorted(t_in + hld)
        if j >= len(side_out):
            break
        t_out, p_out = side_out["time"].iloc[j], side_out["yes"].iloc[j]
        sign = 1.0 if gap > 0 else -1.0
        ret = sign * (p_out - p_in) - float(kalshi_fee(p_in, rate)) - float(kalshi_fee(p_out, rate))
        rows.append({"signal": t, "entry": t_in, "exit": t_out, "gap": gap, "p_in": p_in, "p_out": p_out,
                     "size": n, "ret": ret})
        busy_until = t_out
    return pd.DataFrame(rows, columns=["signal", "entry", "exit", "gap", "p_in", "p_out", "size", "ret"])


_STOP = {"the", "will", "win", "wins", "winner", "fc", "cf", "sc", "ac", "afc", "club", "de", "on", "be", "to", "of",
         "and", "in", "a", "vs", "at", "who", "for", "by", "team", "game", "match"}


def name_tokens(text: str) -> frozenset:
    """Lower-case, accent-free word set of a team/person name, minus filler words and dates."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z][a-z0-9'.]*", t)
    return frozenset(w.strip(".'") for w in words if w.strip(".'") not in _STOP and len(w.strip(".'")) >= 2)


def poly_subject(question: str, answer1: str) -> str:
    """What a Polymarket market's YES token pays on: answer1 for two-way markets, else the 'Will X ...' name."""
    if answer1 and answer1.lower() not in ("yes", "no"):
        return answer1
    q = re.sub(r"\s+", " ", question)
    if re.search(r"\bdraw\b|\btie\b", q, re.I):
        return "tie"
    m = re.match(r"Will (?:the )?(.+?) (?:win|be|finish|become|make|advance|reach|qualify)\b", q, re.I)
    return m.group(1) if m else q


def name_score(a: frozenset, b: frozenset) -> float:
    """Overlap of two name token sets as a share of the smaller one (1.0 = one name contains the other)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def match_by_name(kalshi: pd.DataFrame, poly: pd.DataFrame, tol: str = "2D", min_score: float = 0.67) -> pd.DataFrame:
    """Pair each Polymarket market with the Kalshi market on the same subject and date.

    kalshi: ticker, event_ticker, subject, date. poly: id, subject, date.
    A pair needs dates within `tol`, a name score of at least `min_score`, and
    a clear winner (the runner-up scores lower). Draws match Kalshi 'Tie'.
    Returns id, ticker, event_ticker, score.
    """
    tie = frozenset({"tie"})
    tok = [tie if str(s).lower() in ("tie", "draw") else name_tokens(s) for s in kalshi["subject"]]
    k = kalshi.assign(tok=tok).sort_values("date").reset_index(drop=True)
    kd = k["date"].to_numpy()
    tol = pd.Timedelta(tol).to_timedelta64()
    rows = []
    for pid, subj, d in zip(poly["id"], poly["subject"], poly["date"]):
        tok = tie if str(subj).lower() in ("tie", "draw") else name_tokens(subj)
        d = np.datetime64(d.tz_convert(None) if getattr(d, "tzinfo", None) else d, "ns")
        lo, hi = np.searchsorted(kd, d - tol), np.searchsorted(kd, d + tol, side="right")
        if hi <= lo:
            continue
        cand = k.iloc[lo:hi]
        s = np.array([name_score(tok, t) for t in cand["tok"]])
        best = int(s.argmax())
        if s[best] < min_score or (s >= s[best]).sum() > 1:
            continue
        r = cand.iloc[best]
        rows.append({"id": pid, "ticker": r["ticker"], "event_ticker": r["event_ticker"], "score": float(s[best])})
    return pd.DataFrame(rows, columns=["id", "ticker", "event_ticker", "score"])


def pair_games(kalshi: list[dict], poly: list[dict]) -> list[dict]:
    """Pair live Kalshi game markets with Polymarket US markets on the same game and side.

    Games match on date and team codes (Kalshi KXNBAGAME-26JAN30DETGSW vs
    Polymarket slug nba-det-gsw-2026-01-30, either team order). The side
    matches on the Polymarket team abbreviation or name against the Kalshi
    ticker suffix and yes_sub_title. If the Polymarket market's long side is
    the opponent, `flip` is True and its prices become 1 - price.
    Returns kalshi_ticker, poly_slug, flip.
    """
    from .data.live_books import kalshi_game_key, poly_game_key

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
