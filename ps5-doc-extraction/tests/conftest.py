"""Fixtures: a scripted fake LLM and in-memory PDF builders (no Ollama needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm.base import LLMClient, LLMUnavailableError  # noqa: E402
from app.pipeline.service import ExtractionService  # noqa: E402

FIELDS = ["invoice_number", "invoice_date", "due_date", "vendor_name", "vendor_address", "vendor_tax_id",
          "customer_name", "customer_address", "purchase_order", "currency", "subtotal", "tax_amount",
          "total_amount", "payment_terms"]


def raw_invoice(line_items=None, **values) -> dict:
    """Build an LLM-style response; values may be 'str' or ('value', 'evidence')."""
    out = {}
    for f in FIELDS:
        v = values.get(f)
        if isinstance(v, tuple):
            out[f] = {"value": v[0], "evidence": v[1]}
        else:
            out[f] = {"value": v, "evidence": v}
    out["line_items"] = line_items or []
    return out


class FakeLLM(LLMClient):
    def __init__(self, responses=None, down=False):
        super().__init__("fake-llm", max_retries=0)
        self.responses = list(responses or [])
        self.down = down
        self.calls = []

    def _chat(self, messages, *, json_schema, temperature, max_tokens):
        if self.down:
            raise LLMUnavailableError("fake LLM down")
        self.calls.append(messages)
        r = self.responses.pop(0) if self.responses else raw_invoice()
        return r if isinstance(r, str) else json.dumps(r)


def make_pdf(lines: list[str], pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        pg = doc.new_page()
        for i, line in enumerate(lines):
            pg.insert_text((50, 60 + i * 16), line, fontsize=10)
    return doc.tobytes()


INVOICE_LINES = [
    "Acme Tools Inc.", "1 Main Street, Springfield, IL 62701", "Tax ID: 12-3456789",
    "Invoice No: A-100", "Date: January 15, 2024", "Due: February 14, 2024",
    "Bill To: Wayne Hardware", "Hammer   2   10.00   20.00", "Wrench   1   30.00   30.00",
    "Subtotal: USD 50.00", "Tax: USD 5.00", "Total: USD 55.00", "Terms: Net 30",
]


def good_raw(**overrides) -> dict:
    vals = dict(invoice_number=("A-100", "Invoice No: A-100"), invoice_date=("January 15, 2024", "Date: January 15, 2024"),
                due_date=("February 14, 2024", "Due: February 14, 2024"), vendor_name="Acme Tools Inc.",
                vendor_address="1 Main Street, Springfield, IL 62701", vendor_tax_id=("12-3456789", "Tax ID: 12-3456789"),
                customer_name=("Wayne Hardware", "Bill To: Wayne Hardware"), currency=("USD", "Total: USD 55.00"),
                subtotal=("50.00", "Subtotal: USD 50.00"), tax_amount=("5.00", "Tax: USD 5.00"),
                total_amount=("55.00", "Total: USD 55.00"), payment_terms=("Net 30", "Terms: Net 30"))
    vals.update(overrides)
    items = [{"description": "Hammer", "quantity": "2", "unit_price": "10.00", "amount": "20.00"},
             {"description": "Wrench", "quantity": "1", "unit_price": "30.00", "amount": "30.00"}]
    return raw_invoice(line_items=items, **vals)


@pytest.fixture
def settings(tmp_path):
    return Settings(log_dir=tmp_path / "logs", max_upload_mb=1)


@pytest.fixture
def make_service(settings):
    def _make(responses=None, down=False):
        return ExtractionService(settings, llm=FakeLLM(responses, down=down))
    return _make
