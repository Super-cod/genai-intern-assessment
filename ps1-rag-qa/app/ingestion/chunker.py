"""Structure-aware recursive chunking.

Strategy (see docs/DECISIONS.md, D1):
1. Split text into paragraphs and detect headings; a heading starts a new chunk, so a
   chunk never mixes two sections (e.g. "Annual Leave" and "Sick Leave").
2. Greedily pack whole paragraphs up to `chunk_size` characters.
3. Paragraphs longer than `chunk_size` are split on sentence boundaries.
4. Consecutive chunks in the same section overlap by up to `overlap` characters
   (whole sentences) so facts that straddle a boundary are still retrievable.
5. Each chunk's *embedded* text is prefixed with "document title > section" so short
   chunks keep their context ("contextual chunk headers").
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..schemas import Chunk
from .cleaner import _DATE_LINE
from .loaders import PageText

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.)\s+[A-Z][^.!?]{1,80}$")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")


@dataclass
class _Para:
    text: str
    page: int | None
    section: str | None
    is_heading: bool = False


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if _MD_HEADING.match(line):
        return True
    if line.endswith((".", ",", ";", ":", "?", "!")) or re.search(r":\s*\d", line) or _DATE_LINE.search(line):
        return False  # sentences and "Effective: 2025-01-01"-style metadata lines
    if _NUMBERED_HEADING.match(line):
        return True
    words = line.split()
    if len(words) <= 10 and line.isupper() and any(c.isalpha() for c in line):
        return True
    # Title Case line with no sentence punctuation, e.g. "Annual Leave Policy"
    caps = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
    return 1 <= len(words) <= 8 and caps == len(words) and not line.endswith("?")


def is_question_heading(line: str) -> bool:
    """FAQ style: a short question on its own line followed by its answer."""
    line = line.strip()
    return line.endswith("?") and 2 <= len(line.split()) <= 16 and line[:1].isupper()


def _paragraphs(pages: list[PageText]) -> list[_Para]:
    paras: list[_Para] = []
    section: str | None = None
    for p in pages:
        for block in re.split(r"\n\s*\n", p.text):
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if not lines:
                continue
            # A heading can sit on the first line of a block without a blank line after it.
            while lines and (
                    (is_heading(lines[0]) and (len(lines) > 1 or len(lines[0].split()) <= 10))
                    or (len(lines) > 1 and is_question_heading(lines[0]))):
                section = _MD_HEADING.sub(r"\1", lines[0]).strip()
                paras.append(_Para(section, p.page, section, is_heading=True))
                lines = lines[1:]
            if not lines:
                continue
            # Keep list items on separate lines; join wrapped prose lines with spaces.
            joined, buf = [], []
            for ln in lines:
                if re.match(r"^([-*•]|\d+[.)])\s+", ln) and buf:
                    joined.append(" ".join(buf))
                    buf = [ln]
                else:
                    buf.append(ln)
            joined.append(" ".join(buf))
            paras.append(_Para("\n".join(joined), p.page, section))
    return paras


def _split_long(text: str, size: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    out, cur = [], ""
    for s in sentences:
        while len(s) > size:  # pathological sentence with no punctuation
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:size])
            s = s[size:]
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _tail_overlap(text: str, overlap: int) -> str:
    """Last whole sentences of `text` that fit in `overlap` characters."""
    if overlap <= 0:
        return ""
    tail = ""
    for s in reversed(_SENTENCE_SPLIT.split(text)):
        if len(tail) + len(s) + 1 > overlap:
            break
        tail = f"{s} {tail}".strip()
    return tail


def chunk_document(pages: list[PageText], doc_id: str, doc_name: str, title: str,
                   chunk_size: int = 900, overlap: int = 150,
                   effective_date: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur_text, cur_page, cur_section = "", None, None

    def flush():
        nonlocal cur_text
        body = cur_text.strip()
        if len(body) < 40:  # drop fragments (stray headings, list bullets, leftover headers)
            cur_text = ""
            return
        idx = len(chunks)
        header = f"{title} > {cur_section}" if cur_section else title
        chunks.append(Chunk(
            chunk_id=f"{doc_id}:{idx:04d}",
            doc_id=doc_id, doc_name=doc_name, text=body,
            embed_text=f"{header}\n{body}",
            page=cur_page, section=cur_section, effective_date=effective_date,
        ))
        cur_text = ""

    for para in _paragraphs(pages):
        if para.is_heading:
            flush()
            cur_section, cur_page = para.section, para.page
            continue
        pieces = [para.text] if len(para.text) <= chunk_size else _split_long(para.text, chunk_size)
        for piece in pieces:
            if not cur_text:
                cur_page, cur_section = para.page, para.section
                cur_text = piece
            elif len(cur_text) + len(piece) + 2 <= chunk_size:
                cur_text = f"{cur_text}\n\n{piece}"
            else:
                carry = _tail_overlap(cur_text, overlap)
                flush()
                cur_page, cur_section = para.page, para.section
                cur_text = f"{carry}\n\n{piece}".strip() if carry else piece
    flush()
    return chunks


def content_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()
