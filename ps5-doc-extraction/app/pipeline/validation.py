"""Stage 4 - validation: grounding, typed parsing, schema validation, cross-field rules.

Every value the LLM returns must survive four gates before it is reported as `extracted`:
  1. grounding  - the quoted evidence (or the value itself) must actually occur in the document
                  text; otherwise it is treated as a hallucination and discarded (`unverified`);
  2. parsing    - dates/amounts/currencies are parsed by deterministic code (`invalid` /
                  `ambiguous` on failure);
  3. schema     - the assembled record is validated with the Pydantic output model;
  4. rules      - cross-field consistency (totals, dates) -> `inconsistent`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher

from pydantic import ValidationError

from ..doc_types import DocumentType
from ..schemas import FieldResult, FieldStatus
from .normalize import Parsed, normalize_text, parse_amount, parse_currency, parse_date, parse_number
from .text_extraction import ExtractedDocument

_NUM_TOKEN = re.compile(r"\d[\d.,']*\d|\d")                          # 1,234.56 / 1.234,56 / 42
_SPACED_NUM = re.compile(r"(?<![\d.,])\d{1,3}(?: \d{3})+(?:[.,]\d{1,2})?(?![\d])")  # 1 234,56


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Location:
    page: int
    source: str
    exact: bool


class Grounder:
    """Finds where (and whether) a quoted snippet or value occurs in the extracted document."""

    def __init__(self, doc: ExtractedDocument, fuzzy_threshold: float = 0.85):
        self.pages = [(p.number, p.source, _norm(p.text)) for p in doc.pages if p.text.strip()]
        self.fuzzy_threshold = fuzzy_threshold
        self._numbers = {n: self._page_numbers(t) for n, _, t in self.pages}

    @staticmethod
    def _page_numbers(text: str) -> set[Decimal]:
        out = set()
        for tok in _NUM_TOKEN.findall(text) + _SPACED_NUM.findall(text):
            p = parse_amount(tok)
            if p.ok and p.value is not None:
                out.add(p.value)
        return out

    def locate_text(self, snippet: str | None) -> Location | None:
        if not snippet or len(snippet.strip()) < 2:
            return None
        s = _norm(snippet)
        for n, src, text in self.pages:
            if s in text:
                return Location(n, src, True)
        # Fuzzy match tolerates OCR noise and minor whitespace/punctuation differences.
        for n, src, text in self.pages:
            if self._fuzzy_contains(s, text):
                return Location(n, src, False)
        return None

    def locate_parts(self, value: str) -> Location | None:
        """Multi-line values (addresses) are often not contiguous in layout-sorted text, because
        text from a parallel column is interleaved. Accept if every line/comma part is found."""
        parts = [p.strip() for p in re.split(r"[\n,]", value) if len(p.strip()) >= 3]
        if len(parts) < 2:
            return None
        locs = [self.locate_text(p) for p in parts]
        if all(locs):
            return Location(locs[0].page, locs[0].source, False)
        return None

    def locate_tokens(self, value: str) -> Location | None:
        """Last resort for free-text fields (addresses, names): every word of the value occurs on
        one page. Handles values the model re-joined differently from the interleaved layout text."""
        toks = [t for t in re.findall(r"[a-z0-9#]+", _norm(value)) if len(t) > 1]
        if len(toks) < 3:
            return None
        for n, src, text in self.pages:
            page_toks = set(re.findall(r"[a-z0-9#]+", text))
            if all(t in page_toks for t in toks):
                return Location(n, src, False)
        return None

    def locate_amount(self, value: Decimal) -> Location | None:
        for n, src, _ in self.pages:
            if value in self._numbers[n] or -value in self._numbers[n]:
                return Location(n, src, True)
        return None

    def _fuzzy_contains(self, s: str, text: str) -> bool:
        L = len(s)
        if L < 4 or L > 200:
            return False
        step = max(1, L // 8)
        for i in range(0, max(1, len(text) - L + 1), step):
            window = text[i:i + L + 4]
            if SequenceMatcher(None, s, window).ratio() >= self.fuzzy_threshold:
                return True
        return False


def _parse(kind: str, raw: str, full_text: str) -> Parsed:
    if kind == "date":
        return parse_date(raw)
    if kind == "amount":
        return parse_amount(raw)
    if kind == "number":
        return parse_number(raw)
    if kind == "currency":
        return parse_currency(raw, full_text)
    value = normalize_text(re.sub(r"\s*\n\s*", ", ", raw.strip()).replace(",,", ","))  # multi-line -> "a, b"
    return Parsed(value) if value else Parsed(None, ok=False, issue="empty value")


def _jsonable(v):
    if isinstance(v, Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def validate_scalar(name: str, kind: str, raw_field, grounder: Grounder, full_text: str) -> tuple[FieldResult, object]:
    raw_value = raw_field.value.strip() if raw_field.value and raw_field.value.strip() else None
    if raw_value is None or raw_value.lower() in {"null", "none", "n/a", "not found", "-"}:
        return FieldResult(status=FieldStatus.not_found), None

    # 1) grounding: the VALUE must be in the document - either directly, as a numerically equal
    #    amount, or inside an evidence snippet that is itself verified. (Evidence alone is not
    #    enough: "Total: 500" as evidence for a value of 550 must fail.)
    parsed = _parse(kind, raw_value, full_text)
    if kind == "amount":  # numeric comparison: "150.00" must not match inside "1150.00"
        loc = grounder.locate_amount(parsed.value) if parsed.value is not None else None
    else:
        loc = grounder.locate_text(raw_value) or grounder.locate_parts(raw_value)
        if loc is None and kind == "string":
            loc = grounder.locate_tokens(raw_value)
    if loc is None and kind != "amount" and raw_field.evidence and _norm(raw_value) in _norm(raw_field.evidence):
        loc = grounder.locate_text(raw_field.evidence)
    if loc is None and kind == "currency":
        # currency symbols are 1 char ("$", "€"): check them (or the parsed ISO code) directly
        if raw_value in full_text or (parsed.value and re.search(rf"\b{parsed.value}\b", full_text)):
            loc = Location(grounder.pages[0][0], grounder.pages[0][1], True)
    if loc is None:
        return FieldResult(status=FieldStatus.unverified, raw_value=raw_value, evidence=raw_field.evidence,
                           issues=["value does not appear in the document text; discarded as a possible "
                                   "hallucination"]), None

    base = dict(raw_value=raw_value, evidence=raw_field.evidence, page=loc.page, source=loc.source)
    # 2) parsing
    if parsed.ambiguous:
        return FieldResult(status=FieldStatus.ambiguous, candidates=parsed.candidates,
                           issues=[parsed.issue], confidence=0.3, **base), None
    if not parsed.ok or parsed.value is None:
        return FieldResult(status=FieldStatus.invalid, issues=[parsed.issue or "unparseable"],
                           confidence=0.2, **base), None
    confidence = 0.95 if loc.exact and loc.source == "text" else 0.8 if loc.source == "text" else 0.7
    return FieldResult(value=_jsonable(parsed.value), status=FieldStatus.extracted,
                       confidence=confidence, **base), parsed.value


def validate_line_items(raw_items, kinds: dict[str, str], grounder: Grounder, full_text: str):
    items_json, items_typed, warnings = [], [], []
    for i, item in enumerate(raw_items, start=1):
        raw = item.model_dump()
        if not any(raw.values()):
            continue
        desc = raw.get("description")
        amount = parse_amount(raw["amount"]).value if raw.get("amount") else None
        grounded = (desc and grounder.locate_text(desc)) or (amount is not None and grounder.locate_amount(amount))
        if not grounded:
            warnings.append(f"line item {i} ('{desc}') not found in document text; discarded")
            continue
        typed, js, issues = {}, {}, []
        for k, kind in kinds.items():
            if raw.get(k) in (None, ""):
                typed[k] = js[k] = None
                continue
            p = _parse(kind, raw[k], full_text)
            if p.ok and p.value is not None:
                typed[k], js[k] = p.value, _jsonable(p.value)
            else:
                typed[k] = js[k] = None
                issues.append(f"{k}: {p.issue}")
        if issues:
            js["issues"] = issues
        items_json.append(js)
        items_typed.append(typed)
    return items_json, items_typed, warnings


def validate_extraction(doc_type: DocumentType, raw, doc: ExtractedDocument):
    """Returns (data, fields, line_items_json, issues, warnings)."""
    grounder = Grounder(doc)
    full_text = doc.full_text
    fields: dict[str, FieldResult] = {}
    typed: dict[str, object] = {}
    for name, kind in doc_type.field_kinds.items():
        fields[name], typed[name] = validate_scalar(name, kind, getattr(raw, name), grounder, full_text)

    items_json, warnings = [], []
    for list_name, kinds in doc_type.list_fields.items():
        items_json, items_typed, w = validate_line_items(getattr(raw, list_name), kinds, grounder, full_text)
        typed[list_name] = items_typed
        warnings += w

    # 3) schema validation of the assembled record; a failing field is nulled and marked invalid
    data = {k: v for k, v in typed.items()}
    for _ in range(len(data)):
        try:
            model = doc_type.output_model.model_validate(data)
            break
        except ValidationError as exc:
            for err in exc.errors():
                f = str(err["loc"][0])
                if f in fields:
                    fields[f].status = FieldStatus.invalid
                    fields[f].value = None
                    fields[f].confidence = 0.2
                    fields[f].issues.append(f"schema validation: {err['msg']}")
                data[f] = [] if f in doc_type.list_fields else None
    else:  # pragma: no cover
        model = doc_type.output_model()

    # 4) cross-field business rules
    issues = doc_type.rules(data)
    messages = []
    for iss in issues:
        messages.append(f"[{iss.severity}] {iss.message}")
        if iss.severity == "error":
            for f in iss.fields:
                if f in fields and fields[f].status == FieldStatus.extracted:
                    fields[f].status = FieldStatus.inconsistent
                    fields[f].confidence = min(fields[f].confidence, 0.5)
                    fields[f].issues.append(iss.message)
    return model.model_dump(mode="json"), fields, items_json, messages, warnings
