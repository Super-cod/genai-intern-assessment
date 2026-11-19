"""Deterministic conflict detection between retrieved chunks.

Small local LLMs frequently ignore an instruction like "prefer the newest source" and silently
pick one passage. So conflict handling is done in application logic, not left to the prompt:

  A pair (older, newer) is flagged as a likely conflict when
    1. the chunks come from different documents with different effective dates,
    2. they are about the same topic (embedding cosine >= threshold), and
    3. they contain different numeric values (most policy conflicts are numbers: 20 vs 24 days).

The answerer then (a) orders the newer chunk first, (b) annotates the older one as possibly
superseded, (c) guarantees a conflict note in the response, and (d) regenerates without the
superseded chunk if the model cited only the outdated source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..schemas import RetrievedChunk

_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


@dataclass
class Conflict:
    older: str        # chunk_id
    newer: str        # chunk_id
    older_doc: str
    newer_doc: str
    older_date: str
    newer_date: str


def _numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


def detect_conflicts(retrieved: list[RetrievedChunk], similarity: Callable[[str, str], float],
                     threshold: float = 0.70) -> list[Conflict]:
    out: list[Conflict] = []
    for i, a in enumerate(retrieved):
        for b in retrieved[i + 1:]:
            ca, cb = a.chunk, b.chunk
            if ca.doc_id == cb.doc_id or not ca.effective_date or not cb.effective_date:
                continue
            if ca.effective_date == cb.effective_date:
                continue
            if similarity(ca.chunk_id, cb.chunk_id) < threshold:
                continue
            na, nb = _numbers(ca.text), _numbers(cb.text)
            if not na or not nb or na == nb:
                continue
            older, newer = (ca, cb) if ca.effective_date < cb.effective_date else (cb, ca)
            out.append(Conflict(older.chunk_id, newer.chunk_id, older.doc_name, newer.doc_name,
                                older.effective_date, newer.effective_date))
    return out


def order_newest_first(retrieved: list[RetrievedChunk], conflicts: list[Conflict]) -> list[RetrievedChunk]:
    """Move superseded chunks behind everything else, keeping relative order otherwise."""
    superseded = {c.older for c in conflicts}
    return ([rc for rc in retrieved if rc.chunk.chunk_id not in superseded] +
            [rc for rc in retrieved if rc.chunk.chunk_id in superseded])
