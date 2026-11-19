import json

import pytest

from app.errors import (EmptyFileError, EncryptedFileError, FileTooLargeError, ModelOutputError,
                        ModelUnavailableError, NoTextError, UnknownDocumentTypeError, UnsupportedFormatError)
from app.schemas import FieldStatus

from .conftest import INVOICE_LINES, good_raw, make_pdf, raw_invoice


def test_clean_invoice_is_complete(make_service):
    svc = make_service([good_raw()])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.status.value == "complete" and not r.needs_review
    assert r.data["invoice_date"] == "2024-01-15" and r.data["total_amount"] == "55.00"
    assert r.fields["total_amount"].evidence == "Total: USD 55.00" and r.fields["total_amount"].page == 1
    assert len(r.line_items) == 2
    assert r.fields["purchase_order"].status == FieldStatus.not_found   # not printed -> not_found, not invented


def test_hallucinated_value_is_discarded(make_service):
    svc = make_service([good_raw(purchase_order=("PO-999", "PO: PO-999"))])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    f = r.fields["purchase_order"]
    assert f.status == FieldStatus.unverified and f.value is None and r.data["purchase_order"] is None


def test_evidence_alone_does_not_validate_a_different_value(make_service):
    svc = make_service([good_raw(total_amount=("65.00", "Total: USD 55.00"))])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.fields["total_amount"].status == FieldStatus.unverified


def test_amount_substring_does_not_ground(make_service):
    lines = [l if not l.startswith("Total") else "Total: USD 1155.00" for l in INVOICE_LINES]
    svc = make_service([good_raw(total_amount=("155.00", None))])
    r = svc.extract("a.pdf", make_pdf(lines))
    assert r.fields["total_amount"].status == FieldStatus.unverified


def test_inconsistent_totals_are_flagged(make_service):
    lines = [l if not l.startswith("Total") else "Total: USD 60.00" for l in INVOICE_LINES]
    svc = make_service([good_raw(total_amount=("60.00", "Total: USD 60.00"))])
    r = svc.extract("a.pdf", make_pdf(lines))
    assert r.fields["total_amount"].status == FieldStatus.inconsistent
    assert r.status.value == "partial" and r.needs_review and any("but total is" in i for i in r.validation_issues)


def test_ambiguous_date_reports_candidates(make_service):
    lines = [l if not l.startswith("Date") else "Date: 03/04/2024" for l in INVOICE_LINES]
    svc = make_service([good_raw(invoice_date=("03/04/2024", "Date: 03/04/2024"), due_date=None)])
    r = svc.extract("a.pdf", make_pdf(lines))
    f = r.fields["invoice_date"]
    assert f.status == FieldStatus.ambiguous and len(f.candidates) == 2 and r.data["invoice_date"] is None
    assert "invoice_date" in r.ambiguous_fields and r.status.value == "partial"


def test_invalid_value_marked_invalid(make_service):
    svc = make_service([good_raw(invoice_date=("Date: January", "Date: January 15, 2024"))])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.fields["invoice_date"].status == FieldStatus.invalid


def test_missing_required_field_gives_partial(make_service):
    svc = make_service([good_raw(invoice_number=None)])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.status.value == "partial" and "invoice_number" in r.missing_fields


def test_ungrounded_line_item_dropped(make_service):
    raw = good_raw()
    raw["line_items"].append({"description": "Chainsaw", "quantity": "1", "unit_price": "999.00", "amount": "999.00"})
    r = make_service([raw]).extract("a.pdf", make_pdf(INVOICE_LINES))
    assert len(r.line_items) == 2 and any("Chainsaw" in w for w in r.warnings)


def test_malformed_llm_json_is_repaired_once(make_service):
    svc = make_service(["this is not json", good_raw()])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.status.value == "complete" and len(svc.llm.calls) == 2


def test_persistently_bad_llm_output_is_502(make_service):
    with pytest.raises(ModelOutputError):
        make_service(["nope", "still nope"]).extract("a.pdf", make_pdf(INVOICE_LINES))


def test_llm_down_is_503(make_service):
    with pytest.raises(ModelUnavailableError):
        make_service(down=True).extract("a.pdf", make_pdf(INVOICE_LINES))


@pytest.mark.parametrize("name,data,exc", [
    ("e.pdf", b"", EmptyFileError),
    ("x.pdf", b"hello, I am text", UnsupportedFormatError),
    ("x.docx", b"PK\x03\x04" + b"0" * 50, UnsupportedFormatError),
    ("big.pdf", b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024), FileTooLargeError),
])
def test_invalid_uploads(make_service, name, data, exc):
    with pytest.raises(exc):
        make_service().extract(name, data)


def test_encrypted_pdf(make_service, tmp_path):
    import pymupdf
    doc = pymupdf.open(stream=make_pdf(INVOICE_LINES), filetype="pdf")
    p = tmp_path / "enc.pdf"
    doc.save(p, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(EncryptedFileError):
        make_service().extract("enc.pdf", p.read_bytes())


def test_blank_pdf_has_no_text(make_service):
    with pytest.raises(NoTextError):
        make_service().extract("blank.pdf", make_pdf([]))


def test_unknown_document_type(make_service):
    with pytest.raises(UnknownDocumentTypeError):
        make_service().extract("a.pdf", make_pdf(INVOICE_LINES), document_type="passport")


def test_failures_and_incomplete_results_are_logged(make_service, settings):
    svc = make_service([good_raw(invoice_number=None)])
    svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    with pytest.raises(EmptyFileError):
        svc.extract("e.pdf", b"")
    for h in svc.audit.handlers:
        h.flush()
    lines = [json.loads(l) for l in (settings.log_dir / "extraction_issues.jsonl").read_text().splitlines()]
    events = {l["event"] for l in lines}
    assert {"incomplete", "failed"} <= events


def test_address_rejoined_differently_from_layout_is_grounded(make_service):
    # two-column layout: address lines interleaved with the other column's text
    lines = ["Seller                      Buyer", "10 Anson Road #22-01        2 Orchard Turn",
             "Singapore 079903             Singapore 238801"] + INVOICE_LINES[3:]
    svc = make_service([good_raw(vendor_address="10 Anson Road #22-01 Singapore 079903", vendor_name=None)])
    r = svc.extract("a.pdf", make_pdf(lines))
    assert r.fields["vendor_address"].status == FieldStatus.extracted


def test_free_text_made_of_absent_words_is_not_grounded(make_service):
    svc = make_service([good_raw(vendor_address="99 Imaginary Boulevard, Atlantis")])
    r = svc.extract("a.pdf", make_pdf(INVOICE_LINES))
    assert r.fields["vendor_address"].status == FieldStatus.unverified
