"""Stage 1 - intake validation: size, format (by content, not just extension), integrity."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from ..errors import (CorruptFileError, EmptyFileError, EncryptedFileError, FileTooLargeError,
                      TooManyPagesError, UnsupportedFormatError)

pymupdf.TOOLS.mupdf_display_errors(False)

# Magic bytes of common formats people upload by mistake -> clearer error messages.
_KNOWN_SIGNATURES = {
    b"PK\x03\x04": "ZIP-based office document (docx/xlsx/pptx)",
    b"\x89PNG": "PNG image",
    b"\xff\xd8\xff": "JPEG image",
    b"GIF8": "GIF image",
    b"\xd0\xcf\x11\xe0": "legacy Microsoft Office document",
}


def validate_upload(filename: str, data: bytes, content_type: str | None, max_mb: float,
                    max_pages: int) -> pymupdf.Document:
    """Return an opened PDF document or raise a typed error."""
    if not data:
        raise EmptyFileError(f"'{filename}' is empty (0 bytes)")
    if len(data) > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"'{filename}' is {len(data) / 1e6:.1f} MB; the limit is {max_mb} MB")

    head = data.lstrip()[:8]
    if not head.startswith(b"%PDF-"):
        kind = next((v for k, v in _KNOWN_SIGNATURES.items() if data.startswith(k)), None)
        ext = Path(filename).suffix.lower() or "no extension"
        detail = f"detected {kind}" if kind else f"extension {ext}, content-type {content_type or 'unknown'}"
        raise UnsupportedFormatError(f"'{filename}' is not a PDF ({detail}). Only PDF documents are supported.")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptFileError(f"'{filename}' could not be parsed as a PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise EncryptedFileError(f"'{filename}' is password-protected; remove the password and retry")
    if doc.page_count == 0:
        doc.close()
        raise CorruptFileError(f"'{filename}' has no pages")
    if doc.page_count > max_pages:
        n = doc.page_count
        doc.close()
        raise TooManyPagesError(f"'{filename}' has {n} pages; the limit is {max_pages}")
    return doc
