"""FastAPI application. Run: uvicorn app.api:app --reload  ->  http://localhost:8000  (UI)  /docs (Swagger)."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import get_settings
from .errors import DocumentNotFoundError, FileTooLargeError, RAGError
from .llm import LLMError
from .pipeline import RAGPipeline
from .schemas import DocumentRecord, FileIngestResult, IngestResponse, QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag.api")

app = FastAPI(title="Document RAG Q&A", version="1.0.0",
              description="Upload PDF/TXT documents and ask grounded, cited questions about them.")
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.exception_handler(RAGError)
async def rag_error_handler(_: Request, exc: RAGError):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.code, "message": exc.message, "details": exc.details})


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError):
    logger.error("LLM failure: %s", exc)
    return JSONResponse(status_code=502, content={"error": "llm_failure", "message": str(exc)})


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    p = pipeline()
    return {"status": "ok", "llm": p.llm.name, "embedding_model": p.embedder.model,
            "documents": len(p.index.docs), "chunks": p.index.size}


@app.post("/documents", response_model=IngestResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    s = get_settings()
    if len(files) > s.max_files_per_upload:
        raise HTTPException(400, f"At most {s.max_files_per_upload} files per request")
    limit = s.max_upload_mb * 1024 * 1024
    results = []
    for f in files:
        name = f.filename or "unnamed"
        try:
            data = await f.read(limit + 1)
            if len(data) > limit:
                raise FileTooLargeError(f"'{name}' exceeds the {s.max_upload_mb} MB limit")
            results.append(pipeline().ingest(name, data))
        except RAGError as exc:  # one bad file must not fail the whole batch
            logger.warning("Ingest failed for %s: %s", name, exc.message)
            results.append(FileIngestResult(filename=name, ok=False, error_code=exc.code, message=exc.message))
    p = pipeline()
    return IngestResponse(results=results, total_documents=len(p.index.docs), total_chunks=p.index.size)


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents():
    return list(pipeline().index.docs.values())


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not pipeline().index.remove_document(doc_id):
        raise DocumentNotFoundError(f"No document with id '{doc_id}'")
    return {"deleted": doc_id}


@app.delete("/documents")
def delete_all_documents():
    pipeline().index.clear()
    return {"deleted": "all"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return pipeline().query(req)


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    return {"reset": pipeline().sessions.reset(session_id)}
