"""Data models shared across the pipeline and the API."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------- internal
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str                              # text shown to the LLM / user
    embed_text: str                        # text that is embedded (adds doc + section context)
    page: int | None = None                # 1-based page for PDFs
    section: str | None = None             # nearest heading
    effective_date: str | None = None      # ISO date parsed from "Effective:" / "Last updated:"
    duplicate_of: list[str] = Field(default_factory=list)  # other docs containing the same text


class DocumentRecord(BaseModel):
    doc_id: str
    doc_name: str
    file_type: str
    pages: int | None
    chunks: int
    chars: int
    effective_date: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None       # cosine similarity (0..1-ish)
    bm25_score: float | None = None
    fused_score: float = 0.0               # reciprocal-rank-fusion score (hybrid) or raw score


# ----------------------------------------------------------------------------- LLM contract
class AnswerDraft(BaseModel):
    """Structured output the generator must return."""
    answerable: bool = Field(description="false if the context does not contain the answer")
    answer: str = Field(description="answer text with inline [S#] citations, or a short explanation of what is missing")
    cited_sources: list[str] = Field(default_factory=list, description="source labels used, e.g. ['S1','S3']")
    conflict_note: str | None = Field(default=None, description="describe disagreement between sources, if any")


class RewrittenQuery(BaseModel):
    standalone_question: str


# ----------------------------------------------------------------------------- API
class AnswerStatus(str, Enum):
    answered = "answered"
    not_found = "not_found"                # retrieval found nothing relevant (LLM not called)
    insufficient_context = "insufficient_context"  # LLM judged the context insufficient
    unsupported = "unsupported"            # answer failed the citation/grounding check


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="pass the same id for follow-up questions")
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: str | None = Field(default=None, pattern="^(dense|bm25|hybrid)$")


class Citation(BaseModel):
    label: str                             # S1, S2, ...
    doc_name: str
    page: int | None
    section: str | None
    chunk_id: str
    snippet: str
    effective_date: str | None = None


class RetrievalDebug(BaseModel):
    label: str
    chunk_id: str
    doc_name: str
    page: int | None
    dense_score: float | None
    bm25_score: float | None
    fused_score: float


class QueryResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    conflict_note: str | None = None
    grounding_score: float | None = Field(default=None, description="share of answer content words found in cited sources")
    retrieved: list[RetrievalDebug]
    timings_ms: dict[str, float]


class FileIngestResult(BaseModel):
    filename: str
    ok: bool
    doc_id: str | None = None
    chunks: int = 0
    error_code: str | None = None
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    results: list[FileIngestResult]
    total_documents: int
    total_chunks: int
