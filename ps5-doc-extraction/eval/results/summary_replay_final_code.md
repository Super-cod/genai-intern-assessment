# Replay: saved LLM outputs re-validated with current code (no LLM call)

- **Field accuracy: 164/168 = 97.6%**
- Missed: 1 | Wrong: 1 | Hallucinated: 2 | Flagged: 0
- Line items matched: 50/51

| Field | Accuracy |
|---|---|
| invoice_number | 12/12 |
| invoice_date | 12/12 |
| due_date | 12/12 |
| vendor_name | 12/12 |
| vendor_address | 12/12 |
| vendor_tax_id | 12/12 |
| customer_name | 12/12 |
| customer_address | 12/12 |
| purchase_order | 12/12 |
| currency | 11/12 |
| subtotal | 11/12 |
| tax_amount | 12/12 |
| total_amount | 12/12 |
| payment_terms | 10/12 |

| Document | Field | Outcome | Expected | Got | Status |
|---|---|---|---|---|---|
| inv03_missing_fields.pdf | subtotal | hallucinated | None | 240.0 | inconsistent |
| inv07_scanned.pdf | currency | missed | GBP | None | unverified |
| inv08_label_variants.pdf | payment_terms | hallucinated | None | Pay by: 2 October 2024 | extracted |
| inv09_terms_no_due_date.pdf | payment_terms | wrong | Payment due within 30 days of the invoice date | 30 days of the invoice date. | extracted |
