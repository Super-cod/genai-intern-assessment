# Extraction evaluation results

Model: `OllamaClient(qwen2.5:7b)` | documents: 12 | fields per document: 14 | field decisions: 168

## Overall

- **Field accuracy: 159/168 = 94.6%**
- Missed: 6 | Wrong value: 1 | Hallucinated: 2 | Flagged ambiguous (not a hit): 0
- Line items matched (description + amount): 50/51 = 98.0%
- Latency per document: median 89.2s, max 234.4s

## Accuracy per field

| Field | Correct | Missed | Wrong | Hallucinated | Flagged | Accuracy |
|---|---|---|---|---|---|---|
| invoice_number | 12 | 0 | 0 | 0 | 0 | 100% |
| invoice_date | 12 | 0 | 0 | 0 | 0 | 100% |
| due_date | 12 | 0 | 0 | 0 | 0 | 100% |
| vendor_name | 12 | 0 | 0 | 0 | 0 | 100% |
| vendor_address | 11 | 1 | 0 | 0 | 0 | 92% |
| vendor_tax_id | 12 | 0 | 0 | 0 | 0 | 100% |
| customer_name | 12 | 0 | 0 | 0 | 0 | 100% |
| customer_address | 11 | 1 | 0 | 0 | 0 | 92% |
| purchase_order | 12 | 0 | 0 | 0 | 0 | 100% |
| currency | 11 | 1 | 0 | 0 | 0 | 92% |
| subtotal | 10 | 1 | 0 | 1 | 0 | 83% |
| tax_amount | 11 | 1 | 0 | 0 | 0 | 92% |
| total_amount | 11 | 1 | 0 | 0 | 0 | 92% |
| payment_terms | 10 | 0 | 1 | 1 | 0 | 83% |

## Per document

| Document | Status | Review | Text source | Fields correct | Items | Latency |
|---|---|---|---|---|---|---|
| inv01_classic_usd.pdf | complete | False | {'text': 1} | 14/14 | 3/3 | 93.5s |
| inv02_eu_two_column.pdf | complete | False | {'text': 1} | 14/14 | 2/2 | 90.8s |
| inv03_missing_fields.pdf | partial | True | {'text': 1} | 13/14 | 2/2 | 70.6s |
| inv04_many_items_gbp.pdf | complete | False | {'text': 1} | 14/14 | 8/8 | 116.4s |
| inv05_ambiguous_date_currency.pdf | partial | True | {'text': 1} | 14/14 | 2/2 | 62.9s |
| inv06_inconsistent_totals.pdf | partial | True | {'text': 1} | 14/14 | 2/2 | 78.1s |
| inv07_scanned.pdf | partial | True | {'ocr': 1} | 13/14 | 1/2 | 80.7s |
| inv08_label_variants.pdf | complete | True | {'text': 1} | 11/14 | 2/2 | 95.3s |
| inv09_terms_no_due_date.pdf | complete | True | {'text': 1} | 13/14 | 1/1 | 87.5s |
| inv10_multi_page.pdf | complete | False | {'text': 2} | 14/14 | 24/24 | 234.4s |
| inv11_receipt_no_items.pdf | complete | False | {'text': 1} | 14/14 | 0/0 | 58.9s |
| inv12_india_gst.pdf | partial | True | {'text': 1} | 11/14 | 3/3 | 104.6s |

## Every non-correct field decision

| Document | Field | Outcome | Expected | Got | Status | Raw | Issue |
|---|---|---|---|---|---|---|---|
| inv03_missing_fields.pdf | subtotal | hallucinated | None | 240.0 | inconsistent | '240.00' | subtotal (240.00) + tax (0) = 240.00, but total is 270.00 |
| inv07_scanned.pdf | currency | missed | GBP | None | unverified | '$' | value does not appear in the document text; discarded as a possible hallucinatio |
| inv08_label_variants.pdf | vendor_address | missed | 10 Anson Road #22-01, Singapore 079903 | None | unverified | '10 Anson Road #22-01 Singapore 079903' | value does not appear in the document text; discarded as a possible hallucinatio |
| inv08_label_variants.pdf | customer_address | missed | 2 Orchard Turn, Singapore 238801 | None | unverified | '2 Orchard Turn Singapore 238801' | value does not appear in the document text; discarded as a possible hallucinatio |
| inv08_label_variants.pdf | payment_terms | hallucinated | None | Pay by: 2 October 2024 | extracted | 'Pay by: 2 October 2024' |  |
| inv09_terms_no_due_date.pdf | payment_terms | wrong | Payment due within 30 days of the invoice date | 30 days of the invoice date. | extracted | '30 days of the invoice date.' |  |
| inv12_india_gst.pdf | subtotal | missed | 110000.0 | None | unverified | 'Rs. 1,10,000.00' | value does not appear in the document text; discarded as a possible hallucinatio |
| inv12_india_gst.pdf | tax_amount | missed | 19800.0 | None | unverified | 'Rs. 19,800.00' | value does not appear in the document text; discarded as a possible hallucinatio |
| inv12_india_gst.pdf | total_amount | missed | 129800.0 | None | unverified | 'Rs. 1,29,800.00' | value does not appear in the document text; discarded as a possible hallucinatio |

## Validation issues raised

- inv03_missing_fields.pdf: [error] subtotal (240.00) + tax (0) = 240.00, but total is 270.00
- inv03_missing_fields.pdf: [error] line items sum to 270.00, but subtotal is 240.00
- inv06_inconsistent_totals.pdf: [error] subtotal (900.00) + tax (90.00) = 990.00, but total is 1100.00
- inv07_scanned.pdf: [warning] line items sum to 800.00, but total is 260.00

## Invalid / unsupported input

| File | Result | HTTP | As expected |
|---|---|---|---|
| blank_page.pdf | no_extractable_text | 422 | ✅ |
| empty.pdf | empty_file | 400 | ✅ |
| encrypted.pdf | encrypted_file | 422 | ✅ |
| garbage_after_header.pdf | corrupt_file | 422 | ✅ |
| invoice.docx | unsupported_format | 415 | ✅ |
| not_a_pdf.pdf | unsupported_format | 415 | ✅ |
| photo.png | unsupported_format | 415 | ✅ |
| truncated.pdf | corrupt_file | 422 | ✅ |
| oversized.pdf (in-memory) | file_too_large | 413 | ✅ |
