# Project 5 — Intelligent Document Understanding & Structured Data Extraction

A FastAPI service that turns **PDF invoices** (digital or scanned) into **validated, structured JSON**. It doesn't just output JSON: every field reports **whether it was found, where (page and verbatim evidence), how confident the system is, and why it's uncertain**. Values the model invents are detected and discarded.

Runs fully locally (Ollama + RapidOCR). The LLM sits behind a provider-agnostic API layer, so switching to OpenAI-compatible or Anthropic APIs is a configuration change.

| | |
|---|---|
| Architecture diagram + API | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Technical decisions (answers to all 8 reasoning questions) | [docs/DECISIONS.md](docs/DECISIONS.md) |
| Evaluation report | [docs/EVALUATION.md](docs/EVALUATION.md) |
| API documentation | Swagger UI at `/docs`, OpenAPI at `/openapi.json` |
| Sample documents + structured outputs | [samples/](samples/) · [eval/results/](eval/results/) (one JSON output per invoice) |

## Results at a glance

| Metric | Result |
|---|---|
| Field accuracy, 12 invoices × 14 fields | **97.6%** (164/168) with final validation code; 94.6% in the first full run |
| Line items matched | 50/51 (98%) |
| Ambiguous date / currency flagged instead of guessed | ✅ (inv05) |
| Arithmetic errors in the document detected | ✅ (inv06) |
| Scanned PDF via OCR | ✅ 13/14 fields (inv07) |
| Invalid inputs rejected with the correct status code | 9/9 |
| Unit / API tests (no LLM needed) | 53 passing |

## How it works

```
PDF → [1 Intake]          size, magic bytes, encryption, page count     → 400/413/415/422
    → [2 Text / OCR]      text layer per page; OCR only for scanned pages
    → [3 LLM extraction]  values AS PRINTED + verbatim evidence; null if absent; never infer
    → [4 Validation]      grounding (value must exist in the document) → deterministic parsing
                          (dates, amounts, currency) → Pydantic schema → business rules
    → [5 Response]        typed data + per-field status/evidence/page/confidence + needs_review
```

Field statuses: `extracted` · `not_found` · `ambiguous` (with candidates) · `invalid` · `unverified` (model value not in document, so discarded) · `inconsistent` (fails a cross-field rule).

## Setup

```bash
ollama pull qwen2.5:7b

cd ps5-doc-extraction
python -m venv .venv && .venv\Scripts\activate        # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                                 # optional
python samples/generate_samples.py                     # (samples are already included)
```

## Run

```bash
uvicorn app.api:app --port 8001
```

Open <http://localhost:8001/docs> and try `POST /extract` with any file from `samples/`.

```bash
curl -F "file=@samples/inv06_inconsistent_totals.pdf" "http://localhost:8001/extract?document_type=invoice"
```

Response (abridged, inv06):

```json
{
  "status": "partial",
  "needs_review": true,
  "data": {"invoice_number": "QF-1043", "invoice_date": "2024-07-09", "currency": "USD",
           "subtotal": "900.00", "tax_amount": "90.00", "total_amount": "1100.00", "due_date": null, "...": "..."},
  "fields": {
    "total_amount": {"value": 1100.0, "status": "inconsistent", "raw_value": "1,100.00",
                     "evidence": "TOTAL (USD): $1,100.00", "page": 1, "source": "text", "confidence": 0.5,
                     "issues": ["subtotal (900.00) + tax (90.00) = 990.00, but total is 1100.00"]},
    "due_date": {"value": null, "status": "not_found"},
    "...": "..."
  },
  "missing_fields": ["due_date", "purchase_order", "payment_terms"],
  "ambiguous_fields": [],
  "problem_fields": ["subtotal", "tax_amount", "total_amount"],
  "validation_issues": ["[error] subtotal (900.00) + tax (90.00) = 990.00, but total is 1100.00"],
  "text_sources": {"text": 1},
  "timings_ms": {"text_extraction": 6.1, "llm_extraction": 77000.0, "validation": 3.2, "total": 77010.0}
}
```

Error example:

```bash
curl -F "file=@samples/edge_cases/invoice.docx" http://localhost:8001/extract
# 415 {"error": "unsupported_format", "message": "'invoice.docx' is not a PDF (detected ZIP-based office document (docx/xlsx/pptx)). Only PDF documents are supported."}
```

| Code | When |
|---|---|
| 200 | Extraction done (`status`: `complete` or `partial`, uncertainty reported per field) |
| 400 | Empty file, unknown `document_type` |
| 413 | File over `MAX_UPLOAD_MB` (default 10) |
| 415 | Not a PDF (detected from content, not extension) |
| 422 | Corrupt, encrypted, too many pages, or no extractable text |
| 502 | LLM output failed schema validation after a repair attempt |
| 503 | LLM backend unavailable |

## Tests and evaluation

```bash
pytest -q                            # 53 tests, uses a fake LLM (no Ollama needed)
python -m eval.run_eval              # full evaluation with the LLM (~20 min on a laptop GPU)
python -m eval.replay_validation     # re-validate saved LLM outputs with current code (seconds, no LLM)
```

Failed and incomplete extractions are logged to `logs/extraction_issues.jsonl`.

## Switching to a hosted LLM API

```bash
set LLM_PROVIDER=openai & set LLM_MODEL=gpt-4o-mini & set LLM_BASE_URL=https://api.openai.com/v1 & set LLM_API_KEY=sk-...
set LLM_PROVIDER=anthropic & set LLM_MODEL=claude-opus-5-5    # pip install anthropic; uses ANTHROPIC_API_KEY
```

## Project structure

```
app/
  api.py                    FastAPI routes, OpenAPI docs, error mapping
  config.py, errors.py, schemas.py
  llm/                      provider-agnostic LLM client (Ollama, OpenAI-compatible, Anthropic)
  doc_types/                plug-in document types: base contract + invoice (schemas, prompt, rules)
  pipeline/
    intake.py               upload validation
    text_extraction.py      PyMuPDF text layer + per-page RapidOCR fallback
    extractor.py            LLM extraction (quote + evidence)
    normalize.py            deterministic date / amount / currency parsing
    validation.py           grounding, parsing, schema, business rules, field statuses
    service.py              orchestration + audit logging
samples/                    12 invoices + ground_truth.json + edge_cases/ + generator
eval/                       run_eval.py, replay_validation.py, results/
tests/                      unit + API tests
docs/                       ARCHITECTURE.md, DECISIONS.md, EVALUATION.md
```

## Known limitations

See [EVALUATION.md](docs/EVALUATION.md#known-limitations): a synthetic test set, business rules catch arithmetic but not every plausible hallucination, 60–230 s per invoice on a 4 GB laptop GPU (use async jobs or a hosted API in production), and OCR quality on scans.
