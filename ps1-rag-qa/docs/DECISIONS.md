# Technical decision log — Document RAG Q&A

Each decision lists the alternatives considered and what evidence settled it.

---

## D1. Chunking: structure-aware recursive chunks (900 chars, 150-char sentence overlap, contextual header)

**Choice.** Split at headings first, so a chunk never mixes two sections. Then pack whole paragraphs up to about 900 characters (~200–230 tokens). Split paragraphs that are too long at sentence boundaries. Carry up to 150 characters of whole sentences into the next chunk within the same section. Embed each chunk with a `Document title > Section` prefix.

**Alternatives considered**

| Option | Why not |
|---|---|
| Fixed-size windows (e.g. 512 tokens, 50 overlap) | Cuts across sections and sentences, so "Annual Leave" and "Sick Leave" land in one chunk and its embedding is a blend of two topics. |
| One chunk per page | Pages hold several unrelated sections, and the PDF manual mixes specs, installation and error codes on one page. That produces low-precision retrieval and long, noisy context. |
| Semantic chunking (split where adjacent-sentence embeddings diverge) | It needs an embedding call per sentence at ingest and is harder to debug. Policy and manual documents already have explicit headings, and those are the semantic boundaries. |
| Larger chunks (2000 chars) | Each chunk would cover several facts, so cosine scores get diluted and the "unanswerable" gate becomes less reliable. |

**Why it works for this corpus.** Policy documents, manuals and FAQs are organised by headings, and one heading usually answers one kind of question. Under the chosen chunking every answerable evaluation query has its answer-bearing chunk in the top 3 (Hit@3 = 1.0), and the section label doubles as a readable citation. FAQ questions are detected as headings, so each Q&A pair becomes one chunk. The contextual header fixes the classic failure where a chunk like "Typical runtime is 8 hours…" doesn't name the product it describes.

---

## D2. Embedding model: EmbeddingGemma (768-d) via Ollama, with asymmetric query/document prompts

**Choice.** `embeddinggemma`, a 308M-parameter model, run locally through Ollama. Queries are prefixed with `task: search result | query:` and documents with `title: none | text:`, as the model expects.

**Selection criteria:** (1) runs locally, with no data leaving the machine; (2) strong retrieval quality for its size; (3) small enough to share a 4 GB GPU laptop with a 7B LLM; (4) asymmetric query/document encoding.

| Candidate | Notes |
|---|---|
| `all-minilm` (23M, 384-d) | Fastest, but weaker on paraphrases and question-style queries. |
| **`embeddinggemma`** (308M, 768-d) | Good retrieval quality for its size, task prompts, multilingual. **Chosen.** |
| `nomic-embed-text`, `bge-m3` | Comparable or better quality but larger (bge-m3 is 568M). It would compete with the LLM for VRAM. |
| OpenAI `text-embedding-3-small` | Strong and cheap, but it sends documents to a third party. It's supported through `EMBED_PROVIDER=openai` if that's acceptable. |

**Evidence.** On the 16 answerable queries, dense retrieval alone reaches Hit@1 = 0.94 and Hit@3 = 1.0. It also handles the misspelled query "pasword lenght requirment", which BM25 misses completely. There is a wide gap between answerable queries (best cosine 0.42–0.76) and clearly off-topic ones (0.10–0.23), and the relevance gate relies on that gap.

**A deployment detail that mattered.** On a 4 GB GPU, loading the embedding model evicted the 7B LLM and the reverse, which added about 6 s per query. The embedding model now runs on the CPU (`num_gpu=0`). Calls go to `127.0.0.1` instead of `localhost`, which avoids a ~2 s IPv6 fallback on Windows, and use a persistent HTTP connection. Query embedding dropped from about 3 s to about 0.4 s.

---

## D3. Retrieval: hybrid dense + BM25 with weighted reciprocal rank fusion (BM25 weight 0.3)

**Alternatives measured** (top-5, 16 answerable queries; see EVALUATION.md):

| Mode | Hit@1 | Hit@3 | MRR | Strength | Weakness |
|---|---|---|---|---|---|
| Dense only | 0.94 | 1.00 | 0.97 | paraphrases, typos | exact codes and rare identifiers in large corpora |
| BM25 only | 0.81 | 0.88 | 0.86 | exact terms (E204, emails) | typos (Q13 miss), vocabulary mismatch ("work from home" vs "remotely") |
| Hybrid, BM25 weight 1.0 | 0.88 | 1.00 | 0.94 | | the repeated "RX-200" token pulls the warranty chunk above the specs chunk |
| **Hybrid, BM25 weight 0.3** | **0.94** | **1.00** | **0.97** | dense ranking with a lexical tie-breaker | |

**Decision.** Hybrid with down-weighted BM25. On this small corpus it ties dense-only, but it keeps an exact-match path for identifiers. Those matter more as a corpus grows, because embeddings don't distinguish `E204` from `E207` reliably. **Caveat:** the weight was chosen on the same 16 queries it is evaluated on, so it should be re-validated on a held-out query set with real users' questions.

---

## D4. Unanswerable questions: two-layer refusal (retrieval gate + verified generation)

**Choice.**
1. **Retrieval gate (deterministic, no LLM cost).** If the best cosine is under 0.30 and no identifier matches exactly, return `not_found` immediately.
2. **Generation layer.** The LLM may set `answerable=false`, and its answer must then pass deterministic checks: cited labels exist, every number appears in a cited source, and grounding is at least 0.35. Otherwise the system returns `unsupported` instead of the draft.

**Why two layers.** The gate alone can't catch *near-topic* questions. "What is the parental leave policy?" scores 0.42 against the annual-leave chunks, and "RX-300 warranty" scores 0.66 against the RX-200 warranty. Those have to be refused by the generator reading the context. Relying on the LLM alone is slower and less reliable, because small models tend to answer anyway. The gate handles clearly off-topic questions in about 0.5 s with no generation call.

**Alternatives considered.** A cross-encoder reranker score as the gate is more accurate, but it adds a second model and latency, so it's listed as an improvement. An LLM self-check ("is this supported? yes/no") doubles generation latency on a laptop GPU, and the deterministic number and citation checks catch the most damaging failure (invented figures) for free.

---

## D5. Conflicting and duplicate documents: resolved in application logic, not left to the prompt

**Observation.** The first version only told the model "prefer the most recent source". `qwen2.5:7b` ignored this. It answered the carry-over follow-up from the outdated 2024 handbook ("cannot be carried over") instead of the 2025 update ("up to 5 days"), and it never filled `conflict_note`.

**Choice.** A deterministic `conflicts.py`. It flags pairs of chunks from documents with different effective dates that are about the same topic (cosine ≥ 0.70) and contain different numbers. The system then:
* orders the newer chunk first and labels the older one `WARNING: OUTDATED`;
* always returns a conflict note listing both sources and their dates;
* regenerates without the outdated chunk if the model cited only that chunk.

**Duplicates.** Exact duplicates, judged by the hash of normalised text, are stored once and `duplicate_of` records the other documents. Near-duplicates (cosine ≥ 0.95) are collapsed at query time. Both keep repeated content from filling the top-k slots and pushing out other evidence.

**Trade-off.** The number-difference rule can miss conflicts that aren't numeric ("allowed" vs "not allowed"). It can also flag two related but non-conflicting chunks, but that case costs only an extra note.

---

## D6. Conversation memory used only for query rewriting

**Choice.** History (the last 4 turns) is used to rewrite follow-ups into standalone questions. The generator sees **only** the standalone question and the retrieved sources.

**Evidence.** With history in the generation prompt, the model answered the carry-over follow-up from its own previous answer ("24 days…, which implies leave does not carry over"). Without history it gave the correct, cited "up to 5 days". Keeping prior answers out also stops an earlier mistake from carrying into later turns. A pronoun/length heuristic skips the rewrite call for standalone questions, which saves one LLM round-trip (5–7 s locally) on most turns.

---

## D7. Vector store: FAISS (exact inner product) plus JSON/NumPy persistence

**Choice.** `faiss.IndexFlatIP` over L2-normalised vectors, rebuilt from `vectors.npy` on start-up.

**Why not ChromaDB?** Chroma would also work, but it brings a server or SQLite layer and an ONNX runtime. With FAISS plus BM25 in-process, both indexes sit in one place, and the dedup and relevance logic stays in transparent Python. Exact search is a few milliseconds for well over 100k chunks of this size. Beyond that, the swap would be `IndexHNSWFlat` or a managed vector DB, and only `index.py` would change.
