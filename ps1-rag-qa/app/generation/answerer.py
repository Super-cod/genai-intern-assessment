"""Grounded answer generation + post-hoc citation/grounding verification."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..llm import LLMClient
from ..retrieval.index import tokenize
from ..retrieval.retriever import _IDENTIFIER
from ..schemas import AnswerDraft, AnswerStatus, Citation, RetrievedChunk
from . import prompts
from .conflicts import Conflict, detect_conflicts, order_newest_first

_LABEL = re.compile(r"\[(S\d+)\]")
_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


@dataclass
class Answer:
    status: AnswerStatus
    text: str
    citations: list[Citation]
    conflict_note: str | None
    grounding_score: float | None


def format_source(label: str, rc: RetrievedChunk, superseded_by: str | None = None) -> str:
    c = rc.chunk
    meta = [f"document: {c.doc_name}"]
    if c.page:
        meta.append(f"page {c.page}")
    if c.section:
        meta.append(f"section: {c.section}")
    if c.effective_date:
        meta.append(f"effective date: {c.effective_date}")
    if c.duplicate_of:
        meta.append("also appears in: " + ", ".join(c.duplicate_of))
    header = f"[{label}] ({'; '.join(meta)})"
    if superseded_by:
        header += f"\nWARNING: OUTDATED - this older text may be superseded by the newer source {superseded_by}."
    return f"{header}\n{c.text}"


def _numbers(text: str) -> list[str]:
    """Numeric values in text, ignoring numbers that are part of identifiers (RX-200, E204)."""
    return [n.replace(",", "") for n in _NUMBER.findall(_IDENTIFIER.sub(" ", text))]


def grounding_score(answer: str, sources: list[str], question: str = "") -> tuple[float, list[str]]:
    """Share of answer content words present in the cited sources, plus unsupported numbers.

    A cheap, deterministic hallucination signal: numbers are the most damaging thing to
    invent, so any number in the answer that appears neither in a cited source nor in the
    question is flagged."""
    clean = _LABEL.sub(" ", answer)
    src = " ".join(sources).lower()
    src_tokens = set(tokenize(src + " " + question))
    words = [w for w in tokenize(clean) if len(w) > 2 and not w.isdigit()]
    score = (sum(w in src_tokens for w in words) / len(words)) if words else 1.0
    allowed = set(_numbers(src)) | set(_numbers(question))
    bad_numbers = [n for n in _numbers(clean) if n not in allowed]
    return round(score, 3), bad_numbers


class Answerer:
    def __init__(self, llm: LLMClient, similarity: Callable[[str, str], float] | None = None,
                 min_grounding: float = 0.35, conflict_threshold: float = 0.70):
        self.llm = llm
        self.similarity = similarity
        self.min_grounding = min_grounding
        self.conflict_threshold = conflict_threshold

    def answer(self, question: str, retrieved: list[RetrievedChunk], history: str = "") -> Answer:
        if not retrieved:
            return Answer(AnswerStatus.not_found, prompts.NOT_FOUND_MESSAGE, [], None, None)

        conflicts = detect_conflicts(retrieved, self.similarity, self.conflict_threshold) if self.similarity else []
        retrieved = order_newest_first(retrieved, conflicts)
        labelled, draft, cited = self._generate(question, retrieved, history, conflicts)

        # If the model relied only on outdated text, regenerate without the superseded chunks.
        superseded = {c.older for c in conflicts}
        if draft.answerable and cited and all(labelled[l].chunk.chunk_id in superseded for l in cited):
            fresh = [rc for rc in retrieved if rc.chunk.chunk_id not in superseded]
            labelled, draft, cited = self._generate(question, fresh, history, [])

        citations = [_citation(l, labelled[l]) for l in cited]
        note = draft.conflict_note or _conflict_note(conflicts, labelled)

        if not draft.answerable:
            return Answer(AnswerStatus.insufficient_context, draft.answer.strip() or prompts.NOT_FOUND_MESSAGE,
                          citations, note, None)
        if not cited:  # no valid label: accept only if a retrieved source fully supports the answer
            cited = self._repair_citations(draft.answer, labelled, question) or []
            if not cited:
                return Answer(AnswerStatus.unsupported, prompts.UNSUPPORTED_MESSAGE, [], note, 0.0)
            citations = [_citation(l, labelled[l]) for l in cited]

        support = [format_source(l, labelled[l]) for l in cited]
        score, bad_numbers = grounding_score(draft.answer, support, question)
        if bad_numbers or score < self.min_grounding:
            # Citation repair: small models sometimes cite the wrong label even though the answer
            # is fully supported by another retrieved source. Re-attribute instead of refusing.
            repaired = self._repair_citations(draft.answer, labelled, question)
            if repaired:
                cited = repaired
                citations = [_citation(l, labelled[l]) for l in cited]
                support = [format_source(l, labelled[l]) for l in cited]
                score, bad_numbers = grounding_score(draft.answer, support, question)
        if bad_numbers or score < self.min_grounding:
            extra = f" (unverified values: {', '.join(bad_numbers)})" if bad_numbers else ""
            return Answer(AnswerStatus.unsupported, prompts.UNSUPPORTED_MESSAGE + extra, citations, note, score)
        return Answer(AnswerStatus.answered, draft.answer.strip(), citations, note, score)

    # ------------------------------------------------------------------ internals
    def _repair_citations(self, answer: str, labelled: dict[str, RetrievedChunk], question: str) -> list[str] | None:
        """Smallest set of retrieved sources (1 or 2) that fully supports the answer, if any."""
        labels = list(labelled)
        candidates = [[l] for l in labels] + [[a, b] for i, a in enumerate(labels) for b in labels[i + 1:]]
        best = None
        for combo in candidates:
            score, bad = grounding_score(answer, [format_source(l, labelled[l]) for l in combo], question)
            if not bad and score >= self.min_grounding and (best is None or score > best[0]):
                best = (score, combo)
            if best and len(combo) > len(best[1]):
                break  # prefer a single source when one suffices
        return best[1] if best else None

    def _generate(self, question, retrieved, history, conflicts: list[Conflict]):
        labelled = {f"S{i}": rc for i, rc in enumerate(retrieved, start=1)}
        label_of = {rc.chunk.chunk_id: l for l, rc in labelled.items()}
        newer_for = {c.older: label_of[c.newer] for c in conflicts if c.newer in label_of}
        sources_block = "\n\n".join(format_source(l, rc, newer_for.get(rc.chunk.chunk_id))
                                    for l, rc in labelled.items())
        hist = f"CONVERSATION SO FAR (context only, not a source):\n{history}\n\n" if history else ""
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": prompts.ANSWER_USER.format(
                history=hist, sources=sources_block, question=question)},
        ]
        draft = self.llm.chat_json(messages, AnswerDraft, max_tokens=700)
        # Normalise labels ("[S2]", "s2", "S2") and keep only labels that exist (models invent [S9]).
        raw = [m.upper() for x in draft.cited_sources for m in re.findall(r"[Ss]\d+", x)]
        cited = [l for l in dict.fromkeys(raw + _LABEL.findall(draft.answer)) if l in labelled]
        return labelled, draft, cited


def _conflict_note(conflicts: list[Conflict], labelled: dict[str, RetrievedChunk]) -> str | None:
    if not conflicts:
        return None
    label_of = {rc.chunk.chunk_id: l for l, rc in labelled.items()}
    parts = []
    for c in conflicts:
        new_l, old_l = label_of.get(c.newer, ""), label_of.get(c.older, "")
        parts.append(f"{c.newer_doc} (effective {c.newer_date}) {f'[{new_l}] ' if new_l else ''}differs from "
                     f"older {c.older_doc} (effective {c.older_date}) {f'[{old_l}]' if old_l else ''}".strip())
    return "Sources disagree: " + "; ".join(dict.fromkeys(parts)) + ". The newer document takes precedence."


def _citation(label: str, rc: RetrievedChunk) -> Citation:
    c = rc.chunk
    snippet = c.text if len(c.text) <= 300 else c.text[:297] + "..."
    return Citation(label=label, doc_name=c.doc_name, page=c.page, section=c.section,
                    chunk_id=c.chunk_id, snippet=snippet, effective_date=c.effective_date)
