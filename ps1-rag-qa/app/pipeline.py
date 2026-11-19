"""RAG pipeline orchestration: ingest -> chunk -> embed -> index | query -> rewrite -> retrieve -> answer."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .errors import EmptyDocumentError, EmptyIndexError, ServiceUnavailableError
from .generation.answerer import Answerer
from .generation.conversation import SessionStore, Turn, format_history, rewrite_question
from .ingestion.chunker import chunk_document
from .ingestion.cleaner import clean_pages, extract_effective_date
from .ingestion.loaders import load_document
from .llm import EmbeddingClient, LLMClient, LLMUnavailableError, build_llm
from .retrieval.index import ChunkIndex
from .retrieval.retriever import Retriever
from .schemas import (DocumentRecord, FileIngestResult, QueryRequest, QueryResponse, RetrievalDebug)

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None,
                 embedder: EmbeddingClient | None = None, index: ChunkIndex | None = None):
        self.s = settings or get_settings()
        self.llm = llm or build_llm(self.s)
        self.embedder = embedder or EmbeddingClient(
            self.s.embed_provider, self.s.embed_model, self.s.embed_base_url, self.s.embed_api_key,
            query_prefix=self.s.embed_query_prefix, doc_prefix=self.s.embed_doc_prefix,
            on_cpu=self.s.embed_on_cpu and self.s.embed_provider == "ollama")
        self.index = index or ChunkIndex(self.s.data_dir)
        self.retriever = Retriever(self.index, self.embedder, self.s)
        self.answerer = Answerer(self.llm, similarity=self.index.similarity_by_id)
        self.sessions = SessionStore()

    # ------------------------------------------------------------------ ingestion
    def ingest(self, filename: str, data: bytes) -> FileIngestResult:
        filename = Path(filename).name  # strip any client-supplied path
        ftype, pages, warnings = load_document(filename, data)
        raw_chars = sum(len(p.text) for p in pages)
        pages = clean_pages(pages)
        if not pages:
            raise EmptyDocumentError(f"'{filename}' has no text left after preprocessing")
        if ftype == "pdf":
            empty = sum(1 for p in pages if len(p.text) < 20)
            if empty:
                warnings.append(f"{empty} page(s) had little or no text (possibly scanned images)")
        full_text = "\n".join(p.text for p in pages)
        eff = extract_effective_date(full_text)
        doc_id = hashlib.sha1(data).hexdigest()[:12]

        existing = self.index.docs.get(doc_id)
        if existing:
            return FileIngestResult(filename=filename, ok=True, doc_id=doc_id, chunks=existing.chunks,
                                    warnings=["identical file already indexed; skipped"])
        same_name = self.index.find_by_name(filename)
        if same_name:  # new version of a file with the same name replaces the old one
            self.index.remove_document(same_name.doc_id)
            warnings.append("replaced previous version of this file")

        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        chunks = chunk_document(pages, doc_id, filename, title, self.s.chunk_size,
                                self.s.chunk_overlap, effective_date=eff)
        if not chunks:
            raise EmptyDocumentError(f"'{filename}' produced no usable chunks")
        try:
            vectors = np.asarray(self.embedder.embed_documents([c.embed_text for c in chunks]))
        except LLMUnavailableError as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        record = DocumentRecord(doc_id=doc_id, doc_name=filename, file_type=ftype,
                                pages=len(pages) if ftype == "pdf" else None, chunks=0,
                                chars=raw_chars, effective_date=eff, warnings=warnings)
        added = self.index.add_document(record, chunks, vectors)
        if added < len(chunks):
            warnings.append(f"{len(chunks) - added} chunk(s) were exact duplicates of already-indexed text")
        logger.info("Indexed %s: %d chunks (%d new)", filename, len(chunks), added)
        return FileIngestResult(filename=filename, ok=True, doc_id=doc_id, chunks=added, warnings=warnings)

    # ------------------------------------------------------------------ query
    def query(self, req: QueryRequest) -> QueryResponse:
        if self.index.size == 0:
            raise EmptyIndexError("No documents are indexed yet. Upload PDF or TXT files first.")
        t = {}
        t0 = time.perf_counter()
        sid, session = self.sessions.get_or_create(req.session_id)
        history = format_history(session, self.s.history_turns)
        try:
            standalone = rewrite_question(self.llm, req.question.strip(), history)
            t["rewrite"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            result = self.retriever.retrieve(standalone, req.top_k, req.mode)
            t["retrieval"] = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            # History is used only to rewrite the question. The generator sees the standalone
            # question + sources, so earlier (possibly wrong) answers cannot leak into new ones.
            ans = self.answerer.answer(standalone, result.chunks)
            t["generation"] = (time.perf_counter() - t2) * 1000
        except LLMUnavailableError as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        t["total"] = (time.perf_counter() - t0) * 1000

        session.turns.append(Turn(req.question, ans.text))
        return QueryResponse(
            session_id=sid, question=req.question, standalone_question=standalone,
            status=ans.status, answer=ans.text, citations=ans.citations,
            conflict_note=ans.conflict_note, grounding_score=ans.grounding_score,
            retrieved=[RetrievalDebug(label=f"S{i}", chunk_id=rc.chunk.chunk_id, doc_name=rc.chunk.doc_name,
                                      page=rc.chunk.page, dense_score=_r(rc.dense_score),
                                      bm25_score=_r(rc.bm25_score), fused_score=round(rc.fused_score, 4))
                       for i, rc in enumerate(result.chunks, start=1)],
            timings_ms={k: round(v, 1) for k, v in t.items()},
        )


def _r(x):
    return None if x is None else round(x, 4)
