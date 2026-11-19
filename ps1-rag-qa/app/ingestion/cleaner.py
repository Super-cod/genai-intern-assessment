"""Text preprocessing: normalisation, header/footer removal, de-hyphenation, date metadata."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime

from .loaders import PageText

_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_DATE_LINE = re.compile(
    r"(effective(?:\s+date)?|last\s+updated|updated|revised|version\s+date)\s*[:\-]?\s*"
    r"([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "retriev-\nal" -> "retrieval"
    text = "\n".join(_MULTI_SPACE.sub(" ", ln).strip() for ln in text.split("\n"))
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def remove_repeated_lines(pages: list[PageText], min_ratio: float = 0.6) -> list[PageText]:
    """Drop running headers/footers: short lines that repeat on most pages, plus bare page numbers."""
    if len(pages) < 3:
        return [PageText(p.page, "\n".join(l for l in p.text.split("\n") if not _PAGE_NUM.match(l)))
                for p in pages]
    counts = Counter()
    for p in pages:
        counts.update({ln.strip() for ln in p.text.split("\n") if 0 < len(ln.strip()) < 100})
    repeated = {ln for ln, c in counts.items() if c / len(pages) >= min_ratio}
    cleaned = []
    for p in pages:
        lines = [ln for ln in p.text.split("\n") if ln.strip() not in repeated and not _PAGE_NUM.match(ln)]
        cleaned.append(PageText(p.page, "\n".join(lines)))
    return cleaned


def clean_pages(pages: list[PageText]) -> list[PageText]:
    pages = [PageText(p.page, normalize(p.text)) for p in pages]
    pages = remove_repeated_lines(pages)
    return [PageText(p.page, normalize(p.text)) for p in pages if p.text.strip()]


def extract_effective_date(text: str) -> str | None:
    """Find an 'Effective: <date>' style marker; used to resolve conflicts (newer wins)."""
    m = _DATE_LINE.search(text[:3000])
    if not m:
        return None
    raw = m.group(2).replace(",", "")
    for fmt in ("%B %d %Y", "%b %d %Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None
