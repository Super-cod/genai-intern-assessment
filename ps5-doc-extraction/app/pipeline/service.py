"""Orchestration: intake -> text/OCR -> LLM extraction -> validation -> response, with logging."""
from __future__ import annotations

import json
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler

from ..config import Settings, get_settings
from ..doc_types import REGISTRY
from ..errors import (ExtractionError, ModelOutputError, ModelUnavailableError, NoTextError,
                      UnknownDocumentTypeError)
from ..llm import LLMClient, LLMError, LLMOutputError, LLMUnavailableError, build_llm
from ..schemas import ExtractionResponse, ExtractionStatus, FieldStatus
from .extractor import llm_extract
from .intake import validate_upload
from .text_extraction import extract_text
from .validation import validate_extraction

logger = logging.getLogger("extraction")


def _setup_audit_log(settings: Settings) -> logging.Logger:
    """JSON-lines log of every failed or incomplete extraction (logs/extraction_issues.jsonl)."""
    audit = logging.getLogger(f"extraction.audit.{settings.log_dir}")  # one handler per log directory
    if not audit.handlers:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(settings.log_dir / "extraction_issues.jsonl", maxBytes=5_000_000, backupCount=3,
                                encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        audit.addHandler(h)
        audit.setLevel(logging.INFO)
        audit.propagate = False
    return audit


class ExtractionService:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.s = settings or get_settings()
        self.llm = llm or build_llm(self.s)
        self.audit = _setup_audit_log(self.s)

    def _log(self, event: str, **kw):
        self.audit.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}, default=str))

    def extract(self, filename: str, data: bytes, content_type: str | None = None,
                document_type: str = "invoice") -> ExtractionResponse:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            if document_type not in REGISTRY:
                raise UnknownDocumentTypeError(
                    f"Unknown document_type '{document_type}'. Supported: {', '.join(REGISTRY)}")
            dtype = REGISTRY[document_type]

            doc = validate_upload(filename, data, content_type, self.s.max_upload_mb, self.s.max_pages)
            try:
                t = time.perf_counter()
                extracted = extract_text(doc, self.s.ocr_enabled, self.s.ocr_min_chars_per_page, self.s.ocr_dpi)
                timings["text_extraction"] = (time.perf_counter() - t) * 1000
                n_pages = doc.page_count
            finally:
                doc.close()

            warnings = list(extracted.warnings)
            if extracted.repaired:
                warnings.append("PDF structure was damaged and repaired; some content may be missing")
            text = extracted.full_text
            if len(text.strip()) < 20:
                raise NoTextError(f"'{filename}' contains no extractable text"
                                  + ("" if self.s.ocr_enabled else " (OCR is disabled)"))
            if len(text) > self.s.max_chars_for_llm:
                warnings.append(f"document text truncated from {len(text)} to {self.s.max_chars_for_llm} characters")
                text = text[: self.s.max_chars_for_llm]

            t = time.perf_counter()
            try:
                raw = llm_extract(self.llm, dtype, text)
            except LLMUnavailableError as exc:
                raise ModelUnavailableError(str(exc)) from exc
            except LLMOutputError as exc:
                raise ModelOutputError(str(exc)) from exc
            except LLMError as exc:
                raise ModelUnavailableError(f"LLM call failed: {exc}") from exc
            timings["llm_extraction"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            data_out, fields, items, issues, w = validate_extraction(dtype, raw, extracted)
            warnings += w
            timings["validation"] = (time.perf_counter() - t) * 1000
        except ExtractionError as exc:
            self._log("failed", request_id=rid, filename=filename, error=exc.code, message=exc.message)
            exc.request_id = rid
            raise

        missing = [f for f, r in fields.items() if r.status == FieldStatus.not_found]
        ambiguous = [f for f, r in fields.items() if r.status == FieldStatus.ambiguous]
        problems = [f for f, r in fields.items()
                    if r.status in (FieldStatus.invalid, FieldStatus.unverified, FieldStatus.inconsistent)]
        required_ok = all(fields[f].status == FieldStatus.extracted for f in dtype.required_fields)
        status = ExtractionStatus.complete if required_ok else ExtractionStatus.partial
        needs_review = (not required_ok or bool(ambiguous or problems)
                        or any("[error]" in i for i in issues) or any("OCR" in x for x in warnings))
        timings["total"] = (time.perf_counter() - t0) * 1000

        if status != ExtractionStatus.complete or needs_review:
            self._log("incomplete", request_id=rid, filename=filename, status=status.value,
                      missing_required=[f for f in dtype.required_fields if fields[f].status != FieldStatus.extracted],
                      ambiguous=ambiguous, problems=problems, issues=issues, warnings=warnings)
        logger.info("%s %s: status=%s review=%s %.0fms", rid, filename, status.value, needs_review, timings["total"])

        return ExtractionResponse(
            request_id=rid, filename=filename, document_type=document_type, status=status,
            needs_review=needs_review, data=data_out, fields=fields, line_items=items,
            missing_fields=missing, ambiguous_fields=ambiguous, problem_fields=problems,
            validation_issues=issues, warnings=warnings, pages=n_pages, text_sources=extracted.sources,
            model=self.llm.name, timings_ms={k: round(v, 1) for k, v in timings.items()},
        )
