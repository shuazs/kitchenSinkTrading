"""Parse SEC tender-offer texts (SC TO-I and its amendments): offer type,
price, odd-lot priority, expiry and final results. Used by the odd-lot
tender backtest (examples/run_alpha_batch8.py) and its paper logger
(paper_loggers/odd_lot_tenders.py). Texts are lower-cased plain text, see
`clean_text`.
"""
import html
import re

ODD_YES = (r"(?:first|priority)[^.]{0,250}(?:fewer than 100|less than 100|not more than 99|odd lot)|odd lots?[^.]{0,300}"
           r"(?:will not be subject to proration|not subject to proration|without proration|without regard to proration|in full)")
ODD_NO = (r"no (?:special consideration or )?priority will be given[^.]{0,60}odd|odd lots?[^.]{0,80}"
          r"(?:subject to the same proration|will be subject to proration)|same proration terms|only tenders that are .?round lots")
MON = "(january|february|march|april|may|june|july|august|september|october|november|december)"


def num(s: str) -> float:
    return float(s.replace(",", "").rstrip("."))


def clean_text(raw: str, limit: int = 3_000_000) -> str:
    """Filing HTML/text -> one lower-cased line without tags."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw[:limit]))).lower()


def parse_offer(t: str) -> dict:
    rng = re.search(r"(?:not|no) (?:greater|more) than \$\s?([\d,]+\.?\d*)\s?(?:nor|or|and not) (?:less|lower) than \$\s?([\d,]+\.?\d*)", t)
    rng2 = re.search(r"(?:price|prices) (?:of|between|ranging from) (?:not less than )?\$\s?([\d,]+\.?\d*) (?:and|to) \$\s?([\d,]+\.?\d*)", t)
    lo = hi = None
    if rng:
        hi, lo = num(rng.group(1)), num(rng.group(2))
    elif rng2:
        lo, hi = sorted([num(rng2.group(1)), num(rng2.group(2))])
    fx = re.search(r"(?:at|for) a (?:purchase )?price of \$\s?([\d,]+\.?\d*) per (?:share|common share)", t)
    ex = re.search(r"expire[sd]? at[^.]{0,80}?" + MON + r" (\d{1,2}), (\d{4})", t)
    return dict(dutch=bool(re.search(r"dutch auction", t)) or lo is not None, lo=lo, hi=hi,
                fixed=num(fx.group(1)) if fx else None, odd=bool(re.search(ODD_YES, t)) and not re.search(ODD_NO, t),
                other=bool(re.search(r"offer to (?:exchange|purchase)[^.]{0,80}(?:warrants|notes|debentures|preferred)", t[:20000])),
                cad=bool(re.search(r"canadian dollars|cdn\s?\$", t[:200000])),
                exp=" ".join(ex.groups()) if ex else None)


def final_price(texts) -> tuple:
    """(price in the final/preliminary results, terminated?) from SC TO-I/A texts."""
    clear, term = None, False
    for u in texts:
        if re.search(r"(?:has|have) (?:terminated|withdrawn) the (?:tender )?offer|offer (?:has been|was) (?:terminated|withdrawn)", u):
            term = True
        if re.search(r"final results|preliminary (?:count|results)|accepted for (?:purchase|payment)", u):
            m = re.findall(r"purchase price of \$\s?([\d,]+\.?\d*)", u) or re.findall(r"at a price of \$\s?([\d,]+\.?\d*) per share", u)
            if m:
                clear = num(m[-1])
    return clear, term
