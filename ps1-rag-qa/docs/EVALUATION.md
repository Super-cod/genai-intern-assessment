# Evaluation report — Document RAG Q&A

**Setup.** 6 documents (3 PDF, 3 TXT; 32 chunks after dedup) for a fictional company. The corpus deliberately includes one **conflicting** pair (2024 handbook vs 2025 leave update), one **duplicated** paragraph (the RX-200 warranty appears in both the manual and the FAQ), multi-page PDFs with running headers and footers, and 5 invalid files.
**Models.** LLM `qwen2.5:7b` and embeddings `embeddinggemma`, both local through Ollama.
**Hardware.** Laptop with an RTX 3050 (4 GB VRAM), i7-12650H and 16 GB RAM.
**Reproduce.** `python -m eval.run_eval`. The raw output is in `eval/results/summary.md` and `results.json`.

The 22 test queries (`eval/queries.json`) are:
* 16 answerable: 12 factual, 2 that hit the conflicting documents, 1 that hits the duplicated paragraph, 1 misspelled query, and 1 multi-turn follow-up ("How long does it take to charge **it**?").
* 6 unanswerable: 3 clearly off-topic and 3 **near-topic traps**. The traps are "parental leave" when only annual leave is documented, "RX-**300** warranty" when only the RX-200 is documented, and "can the RX-200 operate underwater?".

## 1. Retrieval quality

A *hit* is a retrieved chunk from the expected document that contains the answer evidence string. Top-5, no relevance gate:

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Dense (EmbeddingGemma) | 0.938 | 1.000 | 1.000 | 0.969 |
| BM25 | 0.812 | 0.875 | 0.938 | 0.859 |
| Hybrid RRF, BM25 weight 1.0 | 0.875 | 1.000 | 1.000 | 0.938 |
| Hybrid RRF, BM25 weight 0.5 | 0.875 | 1.000 | 1.000 | 0.938 |
| **Hybrid RRF, BM25 weight 0.3 (default)** | **0.938** | **1.000** | **1.000** | **0.969** |

What the misses show:
* **BM25 fails on the misspelled query.** "pasword lenght requirment" shares no tokens with the source, so BM25 returns nothing useful. Dense retrieval handles it (cosine 0.58). BM25 also ranks the warranty chunk first for "battery runtime of the RX-200", because "RX-200" appears in it more often.
* **Equal-weight hybrid inherits that BM25 bias** on Q03. Down-weighting BM25 to 0.3 fixes it and keeps the exact-match path for identifiers.
* **The one remaining non-top-1 is Q01**, the conflict question. Both the old and the new leave chunk are retrieved, and the 2024 handbook chunk ranks first. This is expected: the conflict logic in the generation stage puts the newer source first and annotates the older one.

## 2. Relevance gate (refusal before generation)

Best dense cosine per query:

| Group | Range |
|---|---|
| Answerable (16) | **0.42 – 0.76** |
| Clearly off-topic (CEO salary, capital of France) | **0.10 – 0.23**, gated out with no LLM call; answered in 0.3 s |
| Stock price (mentions the company name) | 0.47, passes the gate |
| Near-topic traps (parental leave, RX-300, underwater) | 0.42 – 0.66, pass the gate |

The 0.30 threshold sits in the gap between the two groups. Near-topic questions overlap with answerable ones on score alone, which is why the system has a second, generation-time refusal layer (DECISIONS.md D4).

## 3. Are answers supported by the retrieved sources?

| Metric | Result |
|---|---|
| Answerable questions answered correctly (expected value present) | **16 / 16** |
| …citing an expected source document | **16 / 16** |
| Mean grounding score (share of answer content words found in cited sources) | **0.958** |
| Unanswerable questions refused | **6 / 6** (2 by the gate, 4 by the generator) |
| Conflict note produced for the conflicting-documents question | **1 / 1** |
| Latency per query (local 7B on a 4 GB GPU) | median **7.6 s**, max 17.5 s |

Every answer passed the deterministic checks: cited labels exist, every number appears in a cited source, and grounding is at least 0.35.

## 4. Behaviour when the answer is unavailable

| Query | Path | Response |
|---|---|---|
| "What is the CEO's salary?" | Gate (cosine 0.23) | `not_found` in 0.3 s, LLM not called |
| "What is the parental leave policy?" | Generator saw annual-leave chunks | `insufficient_context` |
| "How long is the warranty on the RX-300?" | Generator saw the RX-200 warranty | `insufficient_context`; it did **not** transfer the RX-200's 2 years |
| "Can the RX-200 operate underwater?" | Generator saw the IP54 / "no wet floors" text | `insufficient_context` (a hedged "no, see IP54" would also have been acceptable) |

## 5. Failure and edge cases found during development (and fixed)

These came from running the evaluation, not from the original design. Each now has a regression test.

| # | Failure observed | Root cause | Fix |
|---|---|---|---|
| 1 | Answered "unused leave cannot be carried over", which is wrong | The model picked the **outdated 2024 handbook** over the 2025 update, ignoring the "prefer newest" prompt rule | Deterministic conflict detection: newer source first, older one labelled OUTDATED, regenerate if only outdated sources are cited (D5) |
| 2 | Follow-up "Can I carry any of it over?" answered from the previous *answer* | Conversation history in the generation prompt distracted the model | History is used only for query rewriting (D6) |
| 3 | Correct answers about the RX-200 rejected as unsupported | The number check read "200" from "RX-200" in the answer as a figure missing from the sources | Numbers inside identifiers are ignored; numbers from the question are allowed |
| 4 | Correct charging answer rejected | The model cited the wrong label (the FAQ) even though the manual chunk supporting it was retrieved | Citation repair: re-attribute to the smallest set of retrieved sources that fully supports the answer; still refuse if none does |
| 5 | Citations dropped | The model returned `"[S2]"` instead of `"S2"` in `cited_sources` | Normalise labels with a regex |
| 6 | "What does E204 mean?" could be gated out | Error codes embed poorly, so dense similarity can fall below the gate | Lexical override: an exact identifier match counts as relevant |
| 7 | Truncated PDF silently accepted | PyMuPDF repairs broken files | A `repaired` warning is returned to the user |
| 8 | cp1252 text decoded as garbage | UTF-16 decoding "succeeds" on almost any even-length input | UTF-16 is tried only when a BOM is present |
| 9 | About 3 s per query embedding | `localhost` → IPv6 fallback on Windows, plus a new TCP connection per call | `127.0.0.1` plus a persistent HTTP client, now about 0.4 s |

**Ingestion edge cases (current behaviour):**

| File | Result |
|---|---|
| `empty.txt` | 422 `no_extractable_text` |
| `not_really_a_pdf.pdf` (text with a .pdf extension) | 422 `corrupt_document`, detected from the magic bytes |
| `scanned_memo.pdf` (image only) | 422 `no_extractable_text`, with a pointer to OCR |
| `spreadsheet.xlsx` | 415 `unsupported_file_type` |
| `corrupted.pdf` (truncated) | Accepted with a "repaired, content may be missing" warning |
| Mixed batch upload | Per-file results; one bad file never fails the batch |

## 6. Limitations

* **Small test set.** 22 queries over 6 short documents is enough to find and fix failure modes, but not to make strong statistical claims. The BM25 weight and the 0.30 gate were tuned on the same queries they are evaluated on, so they need a held-out set.
* The **number-based conflict detector** misses non-numeric contradictions ("allowed" vs "not allowed").
* **Grounding is lexical.** A fluent paraphrase that changes meaning without changing numbers ("must" → "may") can pass. An NLI or LLM-judge check would catch it at extra latency.
* **No OCR** in this service. Scanned PDFs are rejected with a clear error; the extraction service in Project 5 has OCR.
* **Latency** of 7–17 s per answer is dominated by the 7B model partly offloaded to the CPU on a 4 GB GPU. A hosted API (`LLM_PROVIDER=openai|anthropic`) or a larger GPU brings this to about 1–2 s.
* **Sessions are in memory.** A multi-instance deployment would need Redis.

## 7. How retrieval quality could be improved

1. **A cross-encoder reranker** (e.g. `bge-reranker-base`) over the top 20. Its score would also make a better-calibrated relevance gate than raw cosine.
2. **A larger, held-out evaluation set** built from real user questions, with LLM-as-judge scoring of faithfulness and answer relevance (RAGAS-style), tracked in CI.
3. **Metadata filters** (document type, effective date, "current policies only") and **supersedes links** declared at ingest ("this document replaces section 2 of X").
4. **Query expansion / HyDE** for vague queries such as "policy", and multi-query retrieval for questions that compare two documents.
5. **Table-aware PDF parsing** (`pymupdf4llm` or layout models) for documents with complex tables.
