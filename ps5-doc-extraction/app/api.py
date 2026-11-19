"""FastAPI app. Run: uvicorn app.api:app --port 8001  ->  Swagger UI at http://localhost:8001/docs"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import get_settings
from .doc_types import REGISTRY
from .errors import ExtractionError, FileTooLargeError
from .pipeline.service import ExtractionService
from .schemas import ErrorResponse, ExtractionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Document Extraction API",
    version="1.0.0",
    description=(
        "Upload a PDF invoice and receive validated structured JSON. Every field carries a status "
        "(`extracted`, `not_found`, `ambiguous`, `invalid`, `unverified`, `inconsistent`), the verbatim "
        "evidence it came from, the page, and a confidence score, so uncertainty is visible instead of hidden."
    ),
)
_service: ExtractionService | None = None

ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty file or unknown document type"},
    413: {"model": ErrorResponse, "description": "File larger than the upload limit"},
    415: {"model": ErrorResponse, "description": "Not a PDF"},
    422: {"model": ErrorResponse, "description": "Corrupt, encrypted, too many pages, or no extractable text"},
    502: {"model": ErrorResponse, "description": "LLM returned output that failed schema validation"},
    503: {"model": ErrorResponse, "description": "LLM backend unavailable"},
}


def service() -> ExtractionService:
    global _service
    if _service is None:
        _service = ExtractionService()
    return _service


@app.exception_handler(ExtractionError)
async def extraction_error_handler(_: Request, exc: ExtractionError):
    return JSONResponse(status_code=exc.status_code,
                        content={"request_id": getattr(exc, "request_id", None), "error": exc.code,
                                 "message": exc.message})


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", "model": s.llm.name, "document_types": list(REGISTRY),
            "max_upload_mb": s.s.max_upload_mb, "ocr_enabled": s.s.ocr_enabled}


@app.get("/document-types")
def document_types():
    return {name: {"description": d.description, "required_fields": d.required_fields,
                   "schema": d.output_model.model_json_schema()} for name, d in REGISTRY.items()}


@app.post("/extract", response_model=ExtractionResponse, responses=ERRORS)
async def extract(file: UploadFile = File(..., description="PDF document"),
                  document_type: str = Query("invoice", description="Document type to extract")):
    limit = int(get_settings().max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)  # read at most limit+1 bytes: never buffer huge uploads
    if len(data) > limit:
        raise FileTooLargeError(f"'{file.filename}' exceeds the {get_settings().max_upload_mb} MB limit")
    from starlette.concurrency import run_in_threadpool  # CPU/LLM-bound work off the event loop
    return await run_in_threadpool(service().extract, file.filename or "upload.pdf", data,
                                   file.content_type, document_type)
