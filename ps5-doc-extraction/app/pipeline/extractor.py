"""Stage 3 - LLM extraction: locate values and quote evidence (no conversion, no inference)."""
from __future__ import annotations

from ..doc_types import DocumentType
from ..llm import LLMClient


def build_messages(doc_type: DocumentType, text: str) -> list[dict]:
    return [
        {"role": "system", "content": doc_type.system_prompt},
        {"role": "user", "content": f"DOCUMENT TEXT:\n<<<\n{text}\n>>>\n\nExtract the {doc_type.name} fields as JSON."},
    ]


def llm_extract(llm: LLMClient, doc_type: DocumentType, text: str):
    """Returns an instance of doc_type.raw_model (validated, one repair round on bad JSON)."""
    return llm.chat_json(build_messages(doc_type, text), doc_type.raw_model, temperature=0.0, max_tokens=2048)
