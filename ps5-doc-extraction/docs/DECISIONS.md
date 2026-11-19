# Technical decision log — Structured Extraction

The assessment asks eight reasoning questions. Each is answered by one of the decisions below, and the mapping is at the end.

---

## D1. Document type and schema: invoices, 14 scalar fields + line items, every field nullable

**Why invoices.** They are the most common document in back-office automation, and they exercise every hard case in the brief:
* layouts vary widely (single column, two column, receipt, multi-page);
* there are typed values (dates, money, currency, tax IDs) where errors are costly;
* the format is international (EU decimal commas, Indian lakh grouping, ambiguous `03/04/2024`, bare `$`);
* there are **natural cross-field checks** (subtotal + tax = total, line items sum to the subtotal, due date not before issue date), which give independent validation of model output.

**How the schema was chosen.** It covers the fields an accounts-payable workflow actually uses: identifiers (number, PO), dates (issue, due), parties (vendor and customer name, address, tax ID), money (currency, subtotal, tax, total), terms, and line items. **Required fields** are the minimum needed to book an invoice (`invoice_number`, `invoice_date`, `vendor_name`, `total_amount`, `currency`). Everything else is optional, because real invoices often omit due dates, POs or tax IDs, and a missing optional field shouldn't make a document "failed".

**Two schemas.** `RawInvoice` is what the LLM returns: strings as printed, plus evidence. `Invoice` is the typed, validated output (`date`, `Decimal`, ISO currency). Keeping them apart is what allows deterministic parsing and grounding in between.

---

## D2. Preventing invented values: the LLM quotes, code verifies

The approach has five layers, cheapest first:

1. **Prompt contract.** Return values *exactly as printed* with a verbatim evidence snippet, `null` when absent, and "never infer or calculate". For example, don't compute a due date from "Net 30" and don't sum line items into a total. Sample `inv09` tests exactly this: it has payment terms but no due date.
2. **Schema-constrained decoding with every key required.** Ollama `format=<JSON schema>` guarantees parseable JSON. An early version gave the fields defaults, so they were optional in the schema. The constrained decoder then let the model **omit 10 of 14 fields** on a clean invoice. Making every key required forces an explicit value or `null` per field.
3. **Grounding check (the main guard).** The *value* must occur in the extracted document text. That means an exact or fuzzy text match, a numerically equal amount (`1,350.00` equals `1350`), or a match inside an evidence snippet that is itself found in the document. Evidence alone isn't enough: "Total: 500" as evidence for a value of 550 fails. Amounts are compared **numerically**, so `155.00` does not "match" inside `1155.00`. A value that fails is **discarded and marked `unverified`**, so a hallucination can't reach `data`.
4. **Line items** must have their description or amount found in the text, otherwise they're dropped with a warning.
5. **Cross-field rules** catch values that are real but misread (D5).

**Alternatives considered.** An LLM self-consistency check (extract twice and compare) doubles the cost and still can't tell you whether a value exists in the document. Asking the model for a confidence score isn't calibrated. Grounding is deterministic, free, and directly answers "did this value come from the document?".

---

## D3. Validation: Pydantic for structure, deterministic parsers for types, rules for consistency

* **Parsing** (`normalize.py`), done in code rather than by the LLM:
  * **Dates:** month names, ISO, `15-Jan-2024`, ordinals. Dotted `12.02.2024` follows the European day-first convention. A slash date where both readings are valid (`03/04/2024`) becomes **`ambiguous` with both candidates**, never a silent guess.
  * **Amounts:** `1,234.56`, `1.234,56`, `1 234,56`, Indian `1,25,000.00`, `Rs.`/`$`/`€` prefixes, parentheses for negatives.
  * **Currency:** ISO codes and unambiguous symbols are mapped. A bare `$` is ambiguous (USD, CAD, AUD…) unless an ISO code appears elsewhere in the document.
* **Pydantic output model** validates types and constraints (ISO currency pattern, non-negative totals). A field that fails is nulled and marked `invalid`, and the rest of the record survives.
* **Business rules** (`invoice_rules`): subtotal + tax = total, line items sum to the subtotal, quantity × unit price = amount, due date not before issue date. Tolerance is max(0.02, 0.5%) for rounding. Violations mark the involved fields `inconsistent` and set `needs_review`. Values are kept, because the document may itself be wrong, as in `inv06`.

---

## D4. OCR only when needed, decided per page

**Rule.** Use the PDF text layer when a page has at least 30 characters of extractable text. OCR runs only when a page has (almost) no text **and** contains images, which is the definition of a scanned page. Pages with neither are reported as empty.

**Why per page.** Mixed PDFs are common (a digital invoice with a scanned delivery note attached). The text layer is exact and costs about 5 ms per page. OCR takes about 10–20 s per page on a CPU and introduces character errors.

**Engine.** RapidOCR (PaddleOCR models on ONNX runtime) is pip-installable with no system Tesseract binary, which matters for a reproducible setup on Windows. The OCR confidence is surfaced: below 0.85 the response warns, and every OCR-sourced field reports `source: "ocr"` with lower confidence (0.7 vs 0.95). Fuzzy grounding (SequenceMatcher ≥ 0.85) tolerates OCR noise such as `INVOICENO:` vs `Invoice No:`.

**Observed limitation** (sample `inv07`): OCR dropped the `£` symbols and one quantity. The system then correctly *doesn't* report a currency that isn't in the OCR text, which counts as a miss against the ground truth. That's the honest outcome. A layout-aware document model (e.g. Donut or LayoutLMv3) or a vision LLM would do better on scans.

---

## D5. Handling different layouts

* PyMuPDF `sort=True` emits text blocks in reading order (top-to-bottom, left-to-right), which handles two-column headers like FROM / INVOICE TO.
* **The LLM handles label variation** (`Bill No.` / `Invoice #` / `Invoice reference`, `Seller` / `Buyer`, `Amount payable` / `Gross total` / `Grand Total`), which is what regex templates can't do.
* **Multi-line values** such as addresses are often not contiguous in layout-sorted text, because a parallel column interleaves with them. Grounding therefore accepts a multi-line value when **each line or comma part** is found. Without this, correct addresses in `inv01` were marked `unverified`.
* Multi-page documents are sent whole, with page markers (`--- page 2 ---`). Each field reports the page its evidence was found on.

**Alternative.** Per-vendor templates or regex give near-perfect precision on known layouts but zero coverage of new ones. They're a good *complement* for high-volume vendors.

---

## D6. Local LLM behind a provider-agnostic client

`qwen2.5:7b` through Ollama is the default. It's strong at structured JSON for its size, Ollama supports grammar-constrained decoding, and invoices never leave the machine, which suits documents containing personal and financial data. The same `LLMClient` interface (retries, backoff, JSON repair) supports OpenAI-compatible APIs and Anthropic through configuration (`LLM_PROVIDER`). The trade-off is latency: about 60–100 s per invoice on a 4 GB laptop GPU, because evidence quotes roughly double the output tokens. Ways to reduce it: a hosted API (a few seconds), a smaller model, or dropping evidence for fields that are verified numerically.

---

## D7. Measuring extraction accuracy

Field-level comparison against hand-written ground truth (`samples/ground_truth.json`), with type-aware matching: exact ISO dates, amounts within ±0.01, identifiers by alphanumeric containment (`EIN: 84-1234567` ≈ `84-1234567`), and fuzzy matching (≥ 0.85) for names and addresses. Every decision falls into **one** category: correct, missed, wrong, hallucinated, or flagged-ambiguous. Expected-ambiguous and expected-inconsistent fields count as correct only when the system flags them so. Line items are scored separately (description + amount). Errors are categorised because their costs differ: a *hallucinated* total is far worse than a *missed* one, which simply goes to human review.

---

## Mapping to the assessment's reasoning questions

| Question | Answered in |
|---|---|
| 1. Document type and schema selection | D1 |
| 2. Missing or ambiguous information | D1 (nullable fields), D3 (ambiguous dates and currency with candidates), status model in ARCHITECTURE.md |
| 3. Preventing invented values | D2 |
| 4. Validating extracted information | D2 (grounding), D3 (parsing, Pydantic, rules) |
| 5. Different layouts | D5 |
| 6. When OCR is required | D4 |
| 7. Measuring accuracy | D7, EVALUATION.md |
| 8. Multiple document types in future | ARCHITECTURE.md "Extending", `doc_types/base.py` |
