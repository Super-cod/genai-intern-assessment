"""Stage 2 - text extraction with per-page OCR fallback.

Decision rule (docs/DECISIONS.md, D4): a page goes to OCR only when its embedded text layer is
(nearly) empty AND it contains images, i.e. it is a scan/photo. Digital PDFs use the text
layer, which is exact and ~100x faster than OCR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    number: int
    text: str
    source: str            # "text" | "ocr" | "empty"
    ocr_confidence: float | None = None


@dataclass
class ExtractedDocument:
    pages: list[PageResult]
    warnings: list[str] = field(default_factory=list)
    repaired: bool = False

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in self.pages if p.text.strip())

    @property
    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pages:
            out[p.source] = out.get(p.source, 0) + 1
        return out


class OCREngine:
    """Lazy RapidOCR wrapper (ONNX runtime, pip-installable, no system Tesseract needed)."""

    def __init__(self):
        self._engine = None
        self.available = True
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except Exception:  # pragma: no cover - optional dependency
            self.available = False

    def __call__(self, image: np.ndarray) -> tuple[str, float]:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        result, _ = self._engine(image)
        if not result:
            return "", 0.0
        # result: [ [box(4 points), text, score], ... ] -> rebuild lines by vertical position
        items = sorted(((b[0][1], b[0][0], t, s) for b, t, s in result), key=lambda x: (round(x[0] / 12), x[1]))
        lines, current_y, buf = [], None, []
        for y, x, t, s in items:
            if current_y is not None and abs(y - current_y) > 12:
                lines.append("  ".join(buf))
                buf = []
            buf.append(t)
            current_y = y if current_y is None or abs(y - current_y) > 12 else current_y
        if buf:
            lines.append("  ".join(buf))
        conf = float(np.mean([s for *_, s in items]))
        return "\n".join(lines), conf


_ocr = OCREngine()


def extract_text(doc: pymupdf.Document, ocr_enabled: bool = True, min_chars: int = 30,
                 dpi: int = 200) -> ExtractedDocument:
    pages: list[PageResult] = []
    warnings: list[str] = []
    for page in doc:
        n = page.number + 1
        try:
            text = page.get_text("text", sort=True).strip()
        except Exception as exc:
            warnings.append(f"page {n}: text layer unreadable ({exc})")
            text = ""
        has_images = bool(page.get_images(full=False))
        if len(text) >= min_chars:
            pages.append(PageResult(n, text, "text"))
            continue
        if not has_images:
            pages.append(PageResult(n, text, "text" if text else "empty"))
            continue
        # Scanned page
        if not (ocr_enabled and _ocr.available):
            warnings.append(f"page {n} is a scanned image but OCR is disabled/unavailable")
            pages.append(PageResult(n, text, "empty"))
            continue
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        ocr_text, conf = _ocr(img)
        pages.append(PageResult(n, ocr_text, "ocr", round(conf, 3)))
        if conf < 0.85:
            warnings.append(f"page {n}: low OCR confidence ({conf:.2f}); values may be misread")
    return ExtractedDocument(pages, warnings, bool(getattr(doc, "is_repaired", False)))
