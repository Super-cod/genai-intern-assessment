"""Deterministic parsing of raw strings quoted by the LLM.

The LLM only *locates* values; turning "03/04/2024" or "1.234,56 EUR" into typed values is done
here, in code, where ambiguity can be detected instead of silently guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


@dataclass
class Parsed:
    value: object | None = None
    ok: bool = True
    ambiguous: bool = False
    candidates: list = field(default_factory=list)
    issue: str | None = None


# ----------------------------------------------------------------------------- dates
_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_TEXT_DATE_FORMATS = ["%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d %B, %Y", "%B %d, %Y",
                      "%b %d, %Y", "%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d %b %y"]


def parse_date(raw: str) -> Parsed:
    raw_s = raw.strip()
    # DD.MM.YYYY (dot-separated, all digits) is day-first by convention (DE/FR/EU) -> not ambiguous.
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", raw_s)
    if m:
        try:
            return Parsed(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            return Parsed(None, ok=False, issue=f"invalid date '{raw}'")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw_s, flags=re.I).replace(".", " ").strip()
    s = re.sub(r"\s+", " ", s)
    for fmt in _TEXT_DATE_FORMATS:
        try:
            return Parsed(datetime.strptime(s, fmt).date())
        except ValueError:
            continue
    m = re.fullmatch(r"(\d{1,2})[/\-. ](\d{1,2})[/\-. ](\d{2,4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        cands = []
        for d, mth in ((a, b), (b, a)):  # day-first, month-first
            try:
                cands.append(date(y, mth, d))
            except ValueError:
                pass
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1:
            return Parsed(cands[0])
        if len(cands) == 2:
            return Parsed(None, ok=True, ambiguous=True, candidates=[c.isoformat() for c in cands],
                          issue=f"'{raw}' is ambiguous: day-first {cands[0]} or month-first {cands[1]}")
    m = re.fullmatch(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})", s)
    if m:
        try:
            return Parsed(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    return Parsed(None, ok=False, issue=f"could not parse date '{raw}'")


# ----------------------------------------------------------------------------- amounts
_CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY", "Rs": "INR", "Rs.": "INR"}
_AMBIGUOUS_SYMBOLS = {"$": ["USD", "CAD", "AUD", "SGD", "NZD"]}
ISO_CURRENCIES = {"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SGD", "NZD", "SEK",
                  "NOK", "DKK", "AED", "ZAR", "BRL", "MXN", "HKD", "PLN"}


def parse_amount(raw: str) -> Parsed:
    """Parse '1,234.56', '1.234,56', '1 234,56', '1,25,000.00', '(120.00)', '$ 99' etc."""
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = re.sub(r"[^\d.,]", "", s).strip(".,")  # "Rs. 1,29,800.00" -> "1,29,800.00"
    if not s or not re.search(r"\d", s):
        return Parsed(None, ok=False, issue=f"no number in '{raw}'")
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot != -1 and last_comma != -1:            # both present: the later one is decimal sep
        dec = "." if last_dot > last_comma else ","
        thousands = "," if dec == "." else "."
        s = s.replace(thousands, "").replace(dec, ".")
    elif last_comma != -1:                             # only commas
        tail = s[last_comma + 1:]
        if len(tail) == 2 and s.count(",") == 1:      # 1234,56 -> decimal comma
            s = s.replace(",", ".")
        else:                                          # 1,234 / 1,25,000 -> thousands
            s = s.replace(",", "")
    elif s.count(".") > 1:                             # 1.234.567 -> thousands dots
        s = s.replace(".", "")
    try:
        val = Decimal(s).quantize(Decimal("0.01"))
    except InvalidOperation:
        return Parsed(None, ok=False, issue=f"could not parse amount '{raw}'")
    return Parsed(-val if negative else val)


def parse_currency(raw: str, document_text: str = "") -> Parsed:
    s = raw.strip().upper()
    code = re.search(r"\b[A-Z]{3}\b", s)
    if code and code.group(0) in ISO_CURRENCIES:
        return Parsed(code.group(0))
    for sym, iso in _CURRENCY_SYMBOLS.items():
        if sym.upper() in s:
            return Parsed(iso)
    if "$" in s:
        # A bare "$" is ambiguous unless an ISO code appears somewhere in the document.
        for iso in _AMBIGUOUS_SYMBOLS["$"]:
            if re.search(rf"\b{iso}\b", document_text):
                return Parsed(iso)
        return Parsed(None, ambiguous=True, candidates=_AMBIGUOUS_SYMBOLS["$"],
                      issue="'$' alone does not identify the currency (USD, CAD, AUD, ...)")
    return Parsed(None, ok=False, issue=f"unrecognised currency '{raw}'")


def parse_number(raw: str) -> Parsed:
    p = parse_amount(raw)
    if p.ok and p.value is not None and p.value == p.value.to_integral():
        p.value = p.value.quantize(Decimal("1"))
    return p


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
