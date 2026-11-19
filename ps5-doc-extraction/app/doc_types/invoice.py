"""Invoice document type: schema, prompt and business rules."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from .base import DocumentType, Issue, RawField


# ----------------------------------------------------------------------------- LLM contract
class RawLineItem(BaseModel):
    description: str | None
    quantity: str | None
    unit_price: str | None
    amount: str | None


class RawInvoice(BaseModel):
    invoice_number: RawField
    invoice_date: RawField
    due_date: RawField
    vendor_name: RawField
    vendor_address: RawField
    vendor_tax_id: RawField
    customer_name: RawField
    customer_address: RawField
    purchase_order: RawField
    currency: RawField
    subtotal: RawField
    tax_amount: RawField
    total_amount: RawField
    payment_terms: RawField
    line_items: list[RawLineItem]


# ----------------------------------------------------------------------------- validated output
class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class Invoice(BaseModel):
    """Final structured output. Every field is nullable: null means 'not found / not trusted'."""
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    vendor_tax_id: str | None = None
    customer_name: str | None = None
    customer_address: str | None = None
    purchase_order: str | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    subtotal: Decimal | None = None
    tax_amount: Decimal | None = None
    total_amount: Decimal | None = None
    payment_terms: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("subtotal", "tax_amount", "total_amount")
    @classmethod
    def non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("amount must not be negative on an invoice")
        return v


FIELD_KINDS = {
    "invoice_number": "id", "invoice_date": "date", "due_date": "date", "vendor_name": "string",
    "vendor_address": "string", "vendor_tax_id": "id", "customer_name": "string",
    "customer_address": "string", "purchase_order": "id", "currency": "currency",
    "subtotal": "amount", "tax_amount": "amount", "total_amount": "amount", "payment_terms": "string",
}
LINE_ITEM_KINDS = {"description": "string", "quantity": "number", "unit_price": "amount", "amount": "amount"}
REQUIRED = ["invoice_number", "invoice_date", "vendor_name", "total_amount", "currency"]

SYSTEM_PROMPT = """You extract data from invoices. You are given the text of ONE invoice (possibly OCR'd).

For each field return {"value": ..., "evidence": ...}:
- "value": the value EXACTLY as printed in the document (keep the original date and number format,
  e.g. "03/04/2024", "1.234,56", "$ 1,200.00"). Do not convert, translate or reformat.
- "evidence": a short snippet (max ~12 words) copied VERBATIM from the document that contains the value,
  e.g. "Invoice No: INV-2024-001".
- If a field is not printed in the document, return {"value": null, "evidence": null}.

Strict rules:
1. NEVER invent, infer or calculate values. Do not compute a due date from payment terms, do not add up
   line items to get a total, do not guess a currency that is not printed.
2. vendor = the company issuing the invoice (seller / from / supplier). customer = who is billed
   (bill to / buyer / client / sold to).
3. vendor_tax_id: the seller's VAT / GST / GSTIN / Tax ID / EIN, if printed.
4. currency: the currency code or symbol as printed (e.g. "USD", "EUR", "€", "$", "₹").
5. tax_amount: the total tax / VAT / GST amount. total_amount: the final amount due (grand total).
6. line_items: one entry per billed row with description, quantity, unit_price, amount as printed
   (null for any column that is not present). Return [] if there are no line items.
Return only JSON."""


def _close(a: Decimal, b: Decimal, abs_tol=Decimal("0.02"), rel_tol=Decimal("0.005")) -> bool:
    return abs(a - b) <= max(abs_tol, abs(b) * rel_tol)


def invoice_rules(d: dict) -> list[Issue]:
    """Cross-field consistency checks. `d` holds parsed values (None if missing/untrusted)."""
    issues: list[Issue] = []
    sub, tax, tot = d.get("subtotal"), d.get("tax_amount"), d.get("total_amount")
    if sub is not None and tot is not None:
        expected = sub + (tax or Decimal("0"))
        if not _close(expected, tot):
            fields = ["subtotal", "tax_amount", "total_amount"] if tax is not None else ["subtotal", "total_amount"]
            issues.append(Issue(fields, f"subtotal ({sub}) + tax ({tax or 0}) = {expected}, but total is {tot}"))
    items = d.get("line_items") or []
    amounts = [li.get("amount") for li in items]
    if items and all(a is not None for a in amounts) and sub is not None:
        s = sum(amounts)
        if not _close(s, sub):
            issues.append(Issue(["line_items", "subtotal"], f"line items sum to {s}, but subtotal is {sub}"))
    elif items and all(a is not None for a in amounts) and sub is None and tot is not None and tax is None:
        s = sum(amounts)
        if not _close(s, tot):
            issues.append(Issue(["line_items", "total_amount"], f"line items sum to {s}, but total is {tot}",
                                severity="warning"))
    for i, li in enumerate(items):
        q, p, a = li.get("quantity"), li.get("unit_price"), li.get("amount")
        if q is not None and p is not None and a is not None and not _close(q * p, a):
            issues.append(Issue(["line_items"], f"line {i + 1}: {q} x {p} = {q * p}, but amount is {a}",
                                severity="warning"))
    inv, due = d.get("invoice_date"), d.get("due_date")
    if inv and due and due < inv:
        issues.append(Issue(["due_date"], f"due date {due} is before invoice date {inv}"))
    return issues


INVOICE = DocumentType(
    name="invoice",
    description="Commercial invoices: parties, dates, amounts, tax, line items",
    raw_model=RawInvoice,
    output_model=Invoice,
    field_kinds=FIELD_KINDS,
    required_fields=REQUIRED,
    system_prompt=SYSTEM_PROMPT,
    rules=invoice_rules,
    list_fields={"line_items": LINE_ITEM_KINDS},
)
