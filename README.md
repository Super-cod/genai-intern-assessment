# Generative AI Intern — Technical Assessment Submission

[![CI](https://github.com/Super-cod/genai-intern-assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/Super-cod/genai-intern-assessment/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains my complete submission for the Generative AI Technical Assessment, implementing **Problem 1** (Intelligent Document Retrieval & QA), **Problem 5** (Document Understanding & Structured Extraction), and the **Mandatory Reasoning Challenge**.

| Part | Folder | What it is | Live Endpoints |
|---|---|---|---|
| **Problem 1** — Intelligent Document Retrieval & QA | [`ps1-rag-qa/`](ps1-rag-qa/) | Local RAG system: PDF/TXT upload, hybrid dense+BM25 retrieval, cited answers, multi-layered refusal, deterministic conflict and duplicate handling, and conversational follow-ups. | Web UI: `http://localhost:8000`<br>Swagger: `/docs` |
| **Problem 5** — Document Understanding & Structured Extraction | [`ps5-doc-extraction/`](ps5-doc-extraction/) | Production-grade invoice extraction API: PDF → validated typed JSON with per-field status, evidence snippet, page, and confidence; OCR for scanned pages; grounding hallucination guards; business-rule checks. | Web UI: `http://localhost:8001`<br>Swagger: `/docs` |
| **Mandatory reasoning challenge** | [`reasoning-challenge/RELIABILITY.md`](reasoning-challenge/RELIABILITY.md) | Comprehensive engineering breakdown: Root causes, debugging methodologies, proposed multi-layered architecture, evaluation strategies, technique mapping, abstention mechanisms, and latency/cost trade-offs. | Technical Whitepaper |

---

## Results Summary

| Metric / Dimension | Problem 1 — RAG Q&A | Problem 5 — Extraction |
|---|---|---|
| **Headline Quality** | **16/16** answerable questions correct and cited; **6/6** unanswerable refused | **97.6%** field accuracy (164/168); **50/51** line items matched |
| **Retrieval / Parsing** | Hit@1 0.94, Hit@3 1.00, MRR 0.97 (hybrid RRF) | Ambiguous dates & currencies surfaced with candidates; document math errors caught |
| **Edge-Case & Failure Handling** | Policy conflicts resolved chronologically; corrupt, empty, and non-PDF files rejected per file | Hallucinated values discarded (`unverified`); 9/9 invalid inputs mapped to typed HTTP codes |
| **Automated Tests** | **38 passing** (fake LLM, no GPU required) | **54 passing** (fake LLM, no GPU required) |
| **Local Latency (4 GB GPU)** | median 7.6 s per answer | median ~89 s per full invoice extraction |

Detailed benchmark reports and ablation tables: [`ps1-rag-qa/docs/EVALUATION.md`](ps1-rag-qa/docs/EVALUATION.md) and [`ps5-doc-extraction/docs/EVALUATION.md`](ps5-doc-extraction/docs/EVALUATION.md).

---

## Shared Design Principles

1. **Local first, API-ready:** Both systems run 100% offline on local Ollama models (`qwen2.5:7b` for generation, `embeddinggemma` for dense retrieval) with PaddleOCR via ONNX for scans. All LLM calls pass through an abstract `LLMClient` with retries, exponential backoff, schema-constrained decoding, and JSON repair. Switching to hosted APIs (OpenAI, Anthropic) requires only environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`).
2. **The LLM proposes, code verifies:** Output values, citations, numbers, dates, amounts, and cross-field sums are verified deterministically by application logic before reaching the user. Anything unverified is flagged or refused.
3. **Make uncertainty visible:** Rather than guessing, both services return typed statuses (`not_found`, `insufficient_context`, `unsupported`, `ambiguous`, `unverified`, `inconsistent`) with exact evidence snippets.
4. **Resilient & cross-platform:** Vector search utilizes FAISS with an automatic pure-NumPy fallback (`IndexFlatIP`), ensuring zero DLL or architecture crashes across Windows, Linux, and macOS.

---

## Quick Start

### Option A: Local Python Setup (Recommended)

```bash
# 1. Prerequisites: Python 3.10+, Ollama
ollama pull qwen2.5:7b
ollama pull embeddinggemma

# 2. Virtual environment setup
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r ps1-rag-qa/requirements.txt -r ps5-doc-extraction/requirements.txt

# 4. Start Problem 1 Service (Web UI at http://localhost:8000)
cd ps1-rag-qa && uvicorn app.api:app --port 8000

# 5. Start Problem 5 Service in another terminal (Web UI at http://localhost:8001)
cd ps5-doc-extraction && uvicorn app.api:app --port 8001
```

### Option B: Docker Compose

```bash
# Run both services simultaneously in containers:
docker compose up --build
```
- Problem 1 Web UI & API: `http://localhost:8000`
- Problem 5 Web UI & API: `http://localhost:8001`

---

## Running the Automated Test Suites

Both test suites use mock LLM responses and run in **~2 seconds without needing a GPU or Ollama running**:

```bash
# Run unit tests across both projects:
pytest -v ps1-rag-qa/tests
pytest -v ps5-doc-extraction/tests
```

To run the offline evaluations against real local models (requires Ollama running):
```bash
python -m eval.run_eval  # inside ps1-rag-qa/ or ps5-doc-extraction/
```

---

## Project Structure

```text
genai-intern-assessment/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix (Py 3.10 & 3.11)
├── docker-compose.yml             # Orchestration for both microservices
├── README.md                      # Monorepo documentation & quickstart
├── reasoning-challenge/
│   └── RELIABILITY.md             # Mandatory reasoning challenge whitepaper
├── ps1-rag-qa/                    # Problem 1: Document Retrieval & QA
│   ├── app/
│   │   ├── api.py                 # FastAPI endpoints & UI server
│   │   ├── generation/            # Answer generation, prompt templates, conflict resolution
│   │   ├── ingestion/             # Loaders, cleaners, and hierarchical chunker
│   │   ├── llm/                   # Provider-agnostic LLM interface (Ollama/OpenAI/Anthropic)
│   │   ├── retrieval/             # Hybrid FAISS/NumPy dense + BM25 sparse index & RRF
│   │   └── static/index.html      # Interactive web UI
│   ├── docs/                      # ARCHITECTURE.md, DECISIONS.md, EVALUATION.md
│   ├── sample_docs/               # Sample policies, manuals, FAQs & edge case files
│   └── tests/                     # 38 pytest test cases
└── ps5-doc-extraction/            # Problem 5: Document Understanding & Extraction
    ├── app/
    │   ├── api.py                 # FastAPI invoice extraction & UI endpoints
    │   ├── doc_types/             # Document schema definitions (Invoices, Receipts)
    │   ├── pipeline/              # OCR, text extraction, normalizers, grounding & rules
    │   └── static/index.html      # Interactive drag-and-drop extraction UI
    ├── docs/                      # ARCHITECTURE.md, DECISIONS.md, EVALUATION.md
    ├── samples/                   # 12 diverse invoice PDFs & edge-case test files
    └── tests/                     # 54 pytest test cases
```

---

## Benchmarked Environment

- **OS:** Windows 11 / Ubuntu 22.04 LTS
- **CPU:** Intel Core i7-12650H
- **RAM:** 16 GB DDR5
- **GPU:** NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM)
- **Ollama:** 0.17+
- **Python:** 3.11.9
