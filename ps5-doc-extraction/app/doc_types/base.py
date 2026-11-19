"""Document-type plug-in contract.

Supporting a new document type (resume, certificate, ...) = adding one module that defines a
`DocumentType`: the raw LLM schema, the field kinds used for parsing, a prompt and business
rules. Intake, OCR, grounding checks, the API and logging are shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field


class RawField(BaseModel):
    """What the LLM returns for one field: the value as printed, plus the exact source text."""
    # No defaults on purpose: required keys force constrained decoding (Ollama `format`) to emit
    # every field explicitly as a value or null, instead of silently omitting it.
    value: str | None = Field(description="value exactly as printed, or null if absent")
    evidence: str | None = Field(description="short verbatim snippet of the document containing the value")


@dataclass
class Issue:
    fields: list[str]
    message: str
    severity: str = "error"   # error -> field marked inconsistent; warning -> informational


@dataclass
class DocumentType:
    name: str
    description: str
    raw_model: type[BaseModel]                 # LLM output contract
    output_model: type[BaseModel]              # validated, typed output
    field_kinds: dict[str, str]                # field -> string | date | amount | currency | id
    required_fields: list[str]
    system_prompt: str
    rules: Callable[[dict], list[Issue]] = lambda data: []
    list_fields: dict[str, dict[str, str]] = field(default_factory=dict)  # e.g. line_items -> kinds
