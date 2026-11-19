"""Text extraction from PDF and TXT files."""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from ..errors import CorruptDocumentError, EmptyDocumentError, UnsupportedFileError

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
fitz.TOOLS.mupdf_display_errors(False)  # we surface problems as warnings/errors ourselves


@dataclass
class PageText:
    page: int | None   # 1-based page number (None for TXT)
    text: str


def detect_type(filename: str, data: bytes) -> str:
    """Decide file type from extension AND content, so a renamed file is caught."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"'{filename}': only PDF and TXT are supported (got '{ext or 'no extension'}')")
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF-"):
        raise CorruptDocumentError(f"'{filename}' has a .pdf extension but is not a valid PDF")
    if ext == ".txt" and b"\x00" in data[:4096]:
        raise UnsupportedFileError(f"'{filename}' looks like a binary file, not plain text")
    return ext[1:]


def load_pdf(data: bytes, filename: str) -> tuple[list[PageText], list[str]]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for broken files
        raise CorruptDocumentError(f"Could not open PDF '{filename}': {exc}") from exc
    if doc.needs_pass:
        raise CorruptDocumentError(f"'{filename}' is password protected")
    pages = []
    repaired = bool(getattr(doc, "is_repaired", False))
    try:
        for i, page in enumerate(doc, start=1):
            # sort=True orders text blocks top-to-bottom, left-to-right (helps multi-column layouts)
            pages.append(PageText(page=i, text=page.get_text("text", sort=True)))
    except Exception as exc:
        raise CorruptDocumentError(f"Failed while reading '{filename}': {exc}") from exc
    finally:
        doc.close()
    if not any(p.text.strip() for p in pages):
        raise EmptyDocumentError(
            f"'{filename}' contains no extractable text (likely a scanned/image-only PDF; "
            "OCR is out of scope for this service - see the extraction service in ps5 for OCR)")
    warnings = ["PDF structure was damaged and has been repaired; some content may be missing"] if repaired else []
    return pages, warnings


def load_txt(data: bytes, filename: str) -> list[PageText]:
    # utf-16 only with a BOM: without one it "successfully" decodes almost any byte string into garbage.
    encodings = ("utf-16",) if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else ("utf-8-sig", "cp1252")
    for enc in encodings:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes almost anything
        raise CorruptDocumentError(f"Could not decode '{filename}' as text")
    if not text.strip():
        raise EmptyDocumentError(f"'{filename}' is empty")
    return [PageText(page=None, text=text)]


def load_document(filename: str, data: bytes) -> tuple[str, list[PageText], list[str]]:
    """Returns (file_type, pages, warnings)."""
    if not data:
        raise EmptyDocumentError(f"'{filename}' is empty (0 bytes)")
    ftype = detect_type(filename, data)
    if ftype == "pdf":
        pages, warnings = load_pdf(data, filename)
        return ftype, pages, warnings
    return ftype, load_txt(data, filename), []
