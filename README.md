# Generative AI Intern — Technical Assessment Submission

| Part | Folder | What it is |
|---|---|---|
| **Problem 1** — Intelligent Document Retrieval & QA | [`ps1-rag-qa/`](ps1-rag-qa/) | RAG system: PDF/TXT upload, hybrid retrieval, cited answers, refusal when unanswerable, conflict and duplicate handling, conversational follow-ups. FastAPI + web UI. |
| **Problem 5** — Document Understanding & Structured Extraction | [`ps5-doc-extraction/`](ps5-doc-extraction/) | Invoice extraction API: PDF → validated JSON with per-field status, evidence, page and confidence; OCR for scanned pages; hallucination guard; business-rule checks. FastAPI + Swagger. |
| **Mandatory reasoning challenge** | [`reasoning-challenge/RELIABILITY.md`](reasoning-challenge/RELIABILITY.md) | Root causes, debugging, architecture, validation, technique mapping, abstention behaviour and trade-offs for unreliable LLM apps, grounded in failures found while building the two projects. |

## Results summary

| | Problem 1 — RAG Q&A | Problem 5 — Extraction |
|---|---|---|
| Headline quality | 16/16 answerable questions correct and cited; 6/6 unanswerable refused | 97.6% field accuracy (164/168); 50/51 line items |
| Retrieval / parsing | Hit@1 0.94, Hit@3 1.00, MRR 0.97 (hybrid) | Ambiguous dates and currencies flagged; document arithmetic errors detected |
| Failure handling | Conflict between old and new policy detected; corrupt, empty and unsupported files rejected per file | Hallucinated values discarded (`unverified`); 9/9 invalid inputs get the correct HTTP code |
| Tests (fake LLM, no Ollama) | 37 passing | 53 passing |
| Local latency (4 GB GPU) | median 7.6 s per answer | median ~89 s per invoice |

Details and every remaining error: [`ps1-rag-qa/docs/EVALUATION.md`](ps1-rag-qa/docs/EVALUATION.md), [`ps5-doc-extraction/docs/EVALUATION.md`](ps5-doc-extraction/docs/EVALUATION.md).

## Shared design principles

1. **Local first, API-ready.** Both projects run fully offline on Ollama (`qwen2.5:7b`, `embeddinggemma`). All model calls go through one provider-agnostic `LLMClient` (Ollama, any OpenAI-compatible API, or Anthropic Claude) that adds retries with backoff, schema-constrained JSON output, and a validation-and-repair round. Switching to a hosted LLM API is a config change (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`).
2. **The LLM proposes, code verifies.** Citations, numbers, quoted values, dates, amounts and cross-field totals are checked deterministically. Anything that fails is refused or flagged, never passed through silently.
3. **Make uncertainty visible.** Typed statuses (`not_found`, `insufficient_context`, `unsupported` / `ambiguous`, `unverified`, `inconsistent`) instead of confident guesses.
4. **Measured, not assumed.** Each project has an evaluation harness against real local models, and unit tests with a fake LLM that pin the failure handling.

## Quick start

```bash
# prerequisites: Python 3.10+, Ollama
ollama pull qwen2.5:7b
ollama pull embeddinggemma

python -m venv .venv && .venv\Scripts\activate
pip install -r ps1-rag-qa/requirements.txt -r ps5-doc-extraction/requirements.txt

# Problem 1 (UI at http://localhost:8000, Swagger at /docs)
cd ps1-rag-qa && uvicorn app.api:app --port 8000

# Problem 5 (Swagger at http://localhost:8001/docs)
cd ps5-doc-extraction && uvicorn app.api:app --port 8001
```

Tests (no Ollama needed): `pytest -q` in each project folder.
Evaluations (need Ollama): `python -m eval.run_eval` in each project folder.

## Test environment for the reported numbers

Windows 11, Intel i7-12650H, 16 GB RAM, NVIDIA RTX 3050 Laptop (4 GB VRAM), Ollama 0.17, Python 3.11.
