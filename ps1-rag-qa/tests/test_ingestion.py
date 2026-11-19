import pytest

from app.errors import CorruptDocumentError, EmptyDocumentError, UnsupportedFileError
from app.ingestion.chunker import chunk_document, is_heading
from app.ingestion.cleaner import clean_pages, extract_effective_date, normalize
from app.ingestion.loaders import PageText, load_document

from .conftest import HANDBOOK


def test_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_document("sheet.xlsx", b"PK\x03\x04")


def test_rejects_fake_pdf():
    with pytest.raises(CorruptDocumentError):
        load_document("fake.pdf", b"just text")


def test_rejects_empty_file():
    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"")


def test_rejects_whitespace_only_txt():
    with pytest.raises(EmptyDocumentError):
        load_document("blank.txt", b"   \n\n  ")


def test_txt_cp1252_fallback():
    ftype, pages, warnings = load_document("legacy.txt", "Caf\xe9 policy applies.".encode("cp1252"))
    assert ftype == "txt" and "Café" in pages[0].text and warnings == []


def test_normalize_dehyphenates_and_collapses_spaces():
    assert normalize("retriev-\nal   works") == "retrieval works"


def test_repeated_headers_and_page_numbers_removed():
    pages = [PageText(i, f"ACME Internal Header\nContent of page {i} is unique text here.\nPage {i} of 3")
             for i in range(1, 4)]
    cleaned = clean_pages(pages)
    joined = "\n".join(p.text for p in cleaned)
    assert "ACME Internal Header" not in joined and "Page 2 of 3" not in joined
    assert "Content of page 2" in joined


def test_effective_date_extraction():
    assert extract_effective_date("Policy\nEffective: January 1, 2025\n...") == "2025-01-01"
    assert extract_effective_date("Last updated: 2024-06-01") == "2024-06-01"
    assert extract_effective_date("no date here") is None


def test_heading_detection():
    assert is_heading("2. Annual Leave")
    assert is_heading("Remote Work")
    assert not is_heading("Employees may work remotely up to 2 days per week.")
    assert not is_heading("Effective: January 1, 2025")


def test_chunks_respect_sections_and_carry_metadata():
    pages = clean_pages([PageText(None, HANDBOOK)])
    chunks = chunk_document(pages, "d1", "handbook.txt", "Handbook", effective_date="2024-01-01")
    sections = [c.section for c in chunks]
    assert "Annual Leave" in sections and "Remote Work" in sections
    leave = next(c for c in chunks if c.section == "Annual Leave")
    assert "20 days" in leave.text and "remotely" not in leave.text   # no section mixing
    assert leave.embed_text.startswith("Handbook > Annual Leave")      # contextual header
    assert leave.effective_date == "2024-01-01"


def test_long_paragraph_split_with_size_limit_and_overlap():
    para = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(80))
    chunks = chunk_document([PageText(1, para)], "d", "long.txt", "Long", chunk_size=300, overlap=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 80 + 2 for c in chunks)
    # overlap: the last sentence of chunk i reappears at the start of chunk i+1
    last_sentence = chunks[0].text.split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1].text
