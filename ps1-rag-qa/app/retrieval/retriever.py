"""Retrieval: dense, BM25, or hybrid (reciprocal rank fusion) + relevance gate + de-duplication."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from ..llm import EmbeddingClient
from ..schemas import RetrievedChunk
from .index import ChunkIndex

logger = logging.getLogger(__name__)
RRF_K = 60  # standard constant from Cormack et al. (2009)
# Identifier-like tokens (error codes, model numbers, emails): embeddings represent them poorly,
# so an exact lexical match on one of them counts as relevant even if dense similarity is low.
_IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{3,}\b|\S+@\S+\.\w+")


def identifier_tokens(text: str) -> set[str]:
    return {t.lower() for t in _IDENTIFIER.findall(text)}


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_dense: float
    gated_out: bool                       # True => nothing passed the relevance threshold
    dropped_duplicates: int = 0
    candidates: list[RetrievedChunk] = field(default_factory=list)  # pre-gate ranking (for eval)


class Retriever:
    def __init__(self, index: ChunkIndex, embedder: EmbeddingClient, settings):
        self.index, self.embedder, self.s = index, embedder, settings

    def retrieve(self, query: str, k: int | None = None, mode: str | None = None,
                 apply_gate: bool = True) -> RetrievalResult:
        k = k or self.s.top_k
        mode = mode or self.s.retrieval_mode
        qvec = np.asarray(self.embedder.embed_query(query), dtype="float32")
        ck = max(self.s.candidate_k, k)

        dense = self.index.dense_search(qvec, ck) if mode in ("dense", "hybrid") else []
        sparse = self.index.bm25_search(query, ck) if mode in ("bm25", "hybrid") else []

        scores: dict[int, RetrievedChunk] = {}
        if mode == "hybrid":
            for rank, (i, s) in enumerate(dense):
                rc = scores.setdefault(i, RetrievedChunk(chunk=self.index.chunks[i].model_copy(deep=True)))
                rc.dense_score, rc.fused_score = s, rc.fused_score + 1 / (RRF_K + rank + 1)
            for rank, (i, s) in enumerate(sparse):
                rc = scores.setdefault(i, RetrievedChunk(chunk=self.index.chunks[i].model_copy(deep=True)))
                rc.bm25_score, rc.fused_score = s, rc.fused_score + self.s.bm25_weight / (RRF_K + rank + 1)
        else:
            for i, s in (dense or sparse):
                key = "dense_score" if mode == "dense" else "bm25_score"
                scores[i] = RetrievedChunk(chunk=self.index.chunks[i].model_copy(deep=True), fused_score=s, **{key: s})

        # Every candidate gets a dense score so the relevance gate is mode-independent.
        for i, rc in scores.items():
            if rc.dense_score is None:
                rc.dense_score = self.index.cosine(qvec, i)

        ranked = sorted(scores.items(), key=lambda kv: kv[1].fused_score, reverse=True)
        candidates = [rc for _, rc in ranked]
        best_dense = max((rc.dense_score for rc in candidates), default=0.0)

        if not apply_gate:
            return RetrievalResult(candidates[:k], best_dense, False, candidates=candidates)

        # ---- relevance gate -------------------------------------------------------------
        ids = identifier_tokens(query)
        lexical = {i for i, rc in ranked if ids and ids & identifier_tokens(rc.chunk.text)}
        if best_dense < self.s.min_dense_score and not lexical:
            logger.info("Relevance gate: best dense %.3f < %.2f -> no relevant context",
                        best_dense, self.s.min_dense_score)
            return RetrievalResult([], best_dense, True, candidates=candidates)
        floor = max(self.s.min_dense_score, best_dense * self.s.relative_score_cutoff)
        relevant = [(i, rc) for i, rc in ranked if rc.dense_score >= floor or i in lexical]

        # ---- near-duplicate collapse ---------------------------------------------------
        selected: list[tuple[int, RetrievedChunk]] = []
        dropped = 0
        for i, rc in relevant:
            dup = next((s for s in selected
                        if self.index.pair_cosine(i, s[0]) >= self.s.near_duplicate_threshold), None)
            if dup:
                dropped += 1
                kept = dup[1].chunk
                if rc.chunk.doc_name != kept.doc_name and rc.chunk.doc_name not in kept.duplicate_of:
                    kept.duplicate_of = kept.duplicate_of + [rc.chunk.doc_name]
                continue
            selected.append((i, rc))
            if len(selected) == k:
                break
        return RetrievalResult([rc for _, rc in selected], best_dense, False, dropped, candidates)
