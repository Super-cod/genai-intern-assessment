"""API response models."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FieldStatus(str, Enum):
    extracted = "extracted"        # found, verified in the document text, parsed, consistent
    not_found = "not_found"        # not present in the document
    ambiguous = "ambiguous"        # present but has several valid readings (e.g. 03/04/2024)
    invalid = "invalid"            # present but could not be parsed / failed schema validation
    unverified = "unverified"      # model returned a value that is NOT in the document -> discarded
    inconsistent = "inconsistent"  # parsed, but contradicts another field (e.g. totals don't add up)


class FieldResult(BaseModel):
    value: Any = None
    status: FieldStatus
    raw_value: str | None = None
    evidence: str | None = None
    page: int | None = None
    source: str | None = Field(default=None, description="text | ocr")
    confidence: float = 0.0
    candidates: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ExtractionStatus(str, Enum):
    complete = "complete"   # all required fields extracted, no blocking issues
    partial = "partial"     # some required fields missing/ambiguous/inconsistent -> needs review


class ExtractionResponse(BaseModel):
    request_id: str
    filename: str
    document_type: str
    status: ExtractionStatus
    needs_review: bool
    data: dict[str, Any] = Field(description="validated structured output; null = not found or not trusted")
    fields: dict[str, FieldResult]
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str]
    ambiguous_fields: list[str]
    problem_fields: list[str] = Field(description="invalid, unverified or inconsistent fields")
    validation_issues: list[str]
    warnings: list[str]
    pages: int
    text_sources: dict[str, int]
    model: str
    timings_ms: dict[str, float]


class ErrorResponse(BaseModel):
    request_id: str | None = None
    error: str
    message: str
