# RAG evaluation results

LLM: `ollama:qwen2.5:7b` | embeddings: `embeddinggemma` | chunk_size=900 overlap=150 top_k=5 min_dense_score=0.3

## 1. Retrieval quality (answerable queries, top-5, no gate)

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| dense | 0.938 | 1.0 | 1.0 | 0.969 |
| bm25 | 0.812 | 0.875 | 0.938 | 0.859 |
| hybrid | 0.938 | 1.0 | 1.0 | 0.969 |

Misses / non-top-1 per mode:

- dense: Q01 (conflict) rank=2 top1=`employee_handbook_2024.pdf | 2. Annual Leave`
- bm25: Q03 (factual) rank=4 top1=`customer_support_faq.txt | What warranty does the RX-200 have?`
- bm25: Q05 (factual) rank=2 top1=`customer_support_faq.txt | What warranty does the RX-200 have?`
- bm25: Q13 (typo) rank=None top1=`None`
- hybrid: Q01 (conflict) rank=2 top1=`employee_handbook_2024.pdf | 2. Annual Leave`

## 2. Relevance gate calibration (best dense cosine per query)

| Query | Expected | Best dense | Gated out |
|---|---|---|---|
| Q01 | answer | 0.608 | False |
| Q02 | answer | 0.652 | False |
| Q03 | answer | 0.701 | False |
| Q04 | answer | 0.526 | False |
| Q05 | answer | 0.645 | False |
| Q06 | answer | 0.572 | False |
| Q07 | answer | 0.622 | False |
| Q08 | answer | 0.667 | False |
| Q09 | answer | 0.566 | False |
| Q10 | answer | 0.75 | False |
| Q11 | answer | 0.548 | False |
| Q12 | answer | 0.58 | False |
| Q13 | answer | 0.576 | False |
| Q14 | answer | 0.653 | False |
| Q15 | answer | 0.42 | False |
| Q16 | answer | 0.764 | False |
| U01 | refuse | 0.232 | True |
| U02 | refuse | 0.47 | False |
| U03 | refuse | 0.1 | True |
| U04 | refuse | 0.424 | False |
| U05 | refuse | 0.655 | False |
| U06 | refuse_or_qualified | 0.514 | False |

## 3. End-to-end answers (hybrid retrieval)

- Answerable: correct answer 16/16, cited an expected document 16/16
- Unanswerable: correctly refused 6/6
- Conflict note produced when required: 1/1
- Mean grounding score (answered): 0.958
- Latency per query: median 7.55s, max 17.5s

| ID | Type | Status | OK | Cited | Grounding | Answer |
|---|---|---|---|---|---|---|
| Q01 | conflict | answered | ✅ | S1:leave_policy_update_2025.txt | 1.0 | Full-time employees are entitled to 24 days of paid annual leave per calendar year, as of January 1, 2025 [S1]. |
| Q02 | conflict | answered | ✅ | S1:leave_policy_update_2025.txt | 0.8 | Yes, unused annual leave can be carried over to the next year, up to a maximum of 5 days. |
| Q03 | factual | answered | ✅ | S1:rx200_product_manual.pdf | 1.0 | The typical runtime of the RX-200's battery is 8 hours at 50 percent average payload. |
| Q04 | factual | answered | ✅ | S1:rx200_product_manual.pdf | 0.714 | Error code E204 indicates that the battery temperature exceeded 55 degrees Celsius, causing the robot to stop and return to the dock. Allow it to cool for 30 mi |
| Q05 | factual | answered | ✅ | S1:rx200_product_manual.pdf | 1.0 | The maximum payload of the RX-200 is 150 kg. |
| Q06 | factual | answered | ✅ | S1:it_security_policy.txt | 1.0 | The minimum password length is 14 characters. |
| Q07 | factual | answered | ✅ | S1:it_security_policy.txt | 1.0 | Passwords need to be changed every 180 days. |
| Q08 | factual | answered | ✅ | S1:travel_policy.pdf | 1.0 | The nightly hotel limit for domestic business trips is 180 USD. |
| Q09 | factual | answered | ✅ | S1:employee_handbook_2024.pdf | 1.0 | You have 30 days from the expense date to submit an expense claim. |
| Q10 | duplicate | answered | ✅ | S1:customer_support_faq.txt | 1.0 | The RX-200 is covered by a limited warranty of 2 years from the date of delivery. |
| Q11 | factual | answered | ✅ | S1:employee_handbook_2024.pdf | 1.0 | You can work remotely up to 2 days per week with manager approval. |
| Q12 | factual | answered | ✅ | S1:it_security_policy.txt | 1.0 | Suspected phishing emails should be reported using the 'Report Phishing' button or by forwarding them to security@aster-robotics.example. |
| Q13 | typo | answered | ✅ | S1:it_security_policy.txt | 1.0 | All accounts must use passwords of at least 14 characters. |
| Q14 | factual | answered | ✅ | S1:travel_policy.pdf | 1.0 | The daily meal allowance (per diem) for international trips is 85 USD. |
| Q15 | followup | answered | ✅ | S1:rx200_product_manual.pdf | 0.818 | It takes 1.5 hours to fully charge the RX-200's battery using the fast dock, and 2.5 hours on the standard dock. |
| Q16 | factual | answered | ✅ | S1:customer_support_faq.txt | 1.0 | A single fleet manager instance supports up to 50 RX-200 robots. |
| U01 | unanswerable | not_found | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |
| U02 | unanswerable | insufficient_context | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |
| U03 | unanswerable | not_found | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |
| U04 | near_topic | insufficient_context | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |
| U05 | near_topic | insufficient_context | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |
| U06 | near_topic | insufficient_context | ✅ |  | None | I could not find information about this in the uploaded documents, so I can't answer it without guessing. |

## 4. Ingestion edge cases

| File | Accepted | Result |
|---|---|---|
| corrupted.pdf | True | 1 chunks |
| empty.txt | False | HTTP 422 `no_extractable_text`: 'empty.txt' is empty (0 bytes) |
| not_really_a_pdf.pdf | False | HTTP 422 `corrupt_document`: 'not_really_a_pdf.pdf' has a .pdf extension but is not a valid PDF |
| scanned_memo.pdf | False | HTTP 422 `no_extractable_text`: 'scanned_memo.pdf' contains no extractable text (likely a scanned/image-only PDF; OCR is out of scope for this service - see the extraction service in ps5 for OCR) |
| spreadsheet.xlsx | False | HTTP 415 `unsupported_file_type`: 'spreadsheet.xlsx': only PDF and TXT are supported (got '.xlsx') |
