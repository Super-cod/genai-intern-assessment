"""Field-level evaluation of the extraction pipeline against samples/ground_truth.json.

Outcome per (document, field):
  correct       value matches (or: expected null and nothing returned; expected ambiguous/inconsistent and flagged so)
  missed        expected a value, system returned none (not_found / unverified / invalid)
  wrong         system returned a different value
  hallucinated  expected null (not printed) but system returned a value
  flagged       expected a clear value but system marked it ambiguous (safe, but not a hit)

Usage (from ps5-doc-extraction/):  python -m eval.run_eval
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.doc_types.invoice import FIELD_KINDS  # noqa: E402
from app.errors import ExtractionError  # noqa: E402
from app.pipeline.service import ExtractionService  # noqa: E402

SAMPLES = ROOT / "samples"
OUT = ROOT / "eval" / "results"
HAS_VALUE = {"extracted", "inconsistent"}

EDGE_EXPECT = {
    "not_a_pdf.pdf": "unsupported_format", "empty.pdf": "empty_file", "invoice.docx": "unsupported_format",
    "photo.png": "unsupported_format", "encrypted.pdf": "encrypted_file", "blank_page.pdf": "no_extractable_text",
    "garbage_after_header.pdf": "corrupt_file|no_extractable_text", "truncated.pdf": "corrupt_file|no_extractable_text|ok",
}


def _alnum(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match(kind: str, expected, got) -> bool:
    if got is None:
        return False
    if kind == "amount":
        return abs(float(expected) - float(got)) <= 0.011
    if kind in ("date", "currency"):
        return str(expected) == str(got)
    if kind == "id":
        a, b = _alnum(expected), _alnum(got)
        return bool(a) and (a == b or a in b or b in a and len(b) >= len(a) - 2)
    a, b = _alnum(expected), _alnum(got)
    return a == b or (a in b) or SequenceMatcher(None, a, b).ratio() >= 0.85


def score_field(kind, expected, result) -> str:
    status, value = result["status"], result["value"]
    if isinstance(expected, dict):  # expected special status
        exp_status = expected["status"]
        if status == exp_status and ("value" not in expected or match(kind, expected["value"], value)):
            return "correct"
        return "wrong" if status in HAS_VALUE else "missed"
    if expected is None:
        return "hallucinated" if status in HAS_VALUE and value is not None else "correct"
    if status == "ambiguous":
        return "flagged"
    if status not in HAS_VALUE:
        return "missed"
    return "correct" if match(kind, expected, value) else "wrong"


def score_items(expected: list, got: list) -> tuple[int, int]:
    """(matched items, expected items): an item matches if description and amount match."""
    used, hits = set(), 0
    for e in expected:
        for j, g in enumerate(got):
            if j in used:
                continue
            if g.get("amount") is not None and abs(float(g["amount"]) - e["amount"]) <= 0.011 and \
                    match("string", e["description"], g.get("description") or ""):
                used.add(j)
                hits += 1
                break
    return hits, len(expected)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))
    svc = ExtractionService()
    per_field = defaultdict(Counter)
    per_doc, rows, latencies = [], [], []
    item_hits = item_total = 0
    for fname, exp in truth.items():
        print(f"extracting {fname} ...", flush=True)
        resp = svc.extract(fname, (SAMPLES / fname).read_bytes(), "application/pdf")
        r = resp.model_dump(mode="json")
        latencies.append(r["timings_ms"]["total"] / 1000)
        doc_counts = Counter()
        for field, kind in FIELD_KINDS.items():
            outcome = score_field(kind, exp[field], r["fields"][field])
            per_field[field][outcome] += 1
            doc_counts[outcome] += 1
            if outcome != "correct":
                rows.append({"doc": fname, "field": field, "outcome": outcome, "expected": exp[field],
                             "got": r["fields"][field]["value"], "status": r["fields"][field]["status"],
                             "raw": r["fields"][field]["raw_value"], "issues": r["fields"][field]["issues"]})
        h, t = score_items(exp["line_items"], r["line_items"])
        item_hits, item_total = item_hits + h, item_total + t
        per_doc.append({"doc": fname, "status": r["status"], "needs_review": r["needs_review"],
                        "sources": r["text_sources"], "correct": doc_counts["correct"], "fields": len(FIELD_KINDS),
                        "items": f"{h}/{t}", "latency_s": round(latencies[-1], 1),
                        "issues": r["validation_issues"], "warnings": r["warnings"]})
        (OUT / f"{Path(fname).stem}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    edge = []
    for f in sorted((SAMPLES / "edge_cases").glob("*")):
        try:
            resp = svc.extract(f.name, f.read_bytes(), None)
            got, http = "ok", 200
        except ExtractionError as exc:
            got, http = exc.code, exc.status_code
        edge.append({"file": f.name, "result": got, "http": http,
                     "as_expected": got in EDGE_EXPECT.get(f.name, "").split("|")})
    big = b"%PDF-1.4\n" + b"0" * int(svc.s.max_upload_mb * 1024 * 1024 + 10)
    try:
        svc.extract("oversized.pdf", big)
        edge.append({"file": "oversized.pdf (in-memory)", "result": "ok", "http": 200, "as_expected": False})
    except ExtractionError as exc:
        edge.append({"file": "oversized.pdf (in-memory)", "result": exc.code, "http": exc.status_code,
                     "as_expected": exc.code == "file_too_large"})

    total = Counter()
    for c in per_field.values():
        total.update(c)
    n = sum(total.values())
    L = ["# Extraction evaluation results", "",
         f"Model: `{svc.llm.name}` | documents: {len(truth)} | fields per document: {len(FIELD_KINDS)} | "
         f"field decisions: {n}", "",
         "## Overall", "",
         f"- **Field accuracy: {total['correct']}/{n} = {total['correct'] / n:.1%}**",
         f"- Missed: {total['missed']} | Wrong value: {total['wrong']} | Hallucinated: {total['hallucinated']} | "
         f"Flagged ambiguous (not a hit): {total['flagged']}",
         f"- Line items matched (description + amount): {item_hits}/{item_total} = {item_hits / max(item_total, 1):.1%}",
         f"- Latency per document: median {statistics.median(latencies):.1f}s, max {max(latencies):.1f}s", "",
         "## Accuracy per field", "", "| Field | Correct | Missed | Wrong | Hallucinated | Flagged | Accuracy |",
         "|---|---|---|---|---|---|---|"]
    for f in FIELD_KINDS:
        c = per_field[f]
        tot = sum(c.values())
        L.append(f"| {f} | {c['correct']} | {c['missed']} | {c['wrong']} | {c['hallucinated']} | {c['flagged']} | "
                 f"{c['correct'] / tot:.0%} |")
    L += ["", "## Per document", "", "| Document | Status | Review | Text source | Fields correct | Items | Latency |",
          "|---|---|---|---|---|---|---|"]
    for d in per_doc:
        L.append(f"| {d['doc']} | {d['status']} | {d['needs_review']} | {d['sources']} | {d['correct']}/{d['fields']} | "
                 f"{d['items']} | {d['latency_s']}s |")
    L += ["", "## Every non-correct field decision", "", "| Document | Field | Outcome | Expected | Got | Status | Raw | Issue |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        iss = (r["issues"][0] if r["issues"] else "").replace("|", "/")[:80]
        L.append(f"| {r['doc']} | {r['field']} | {r['outcome']} | {r['expected']} | {r['got']} | {r['status']} | "
                 f"{str(r['raw'])[:40]!r} | {iss} |")
    L += ["", "## Validation issues raised", ""]
    for d in per_doc:
        for i in d["issues"] + d["warnings"]:
            L.append(f"- {d['doc']}: {i}")
    L += ["", "## Invalid / unsupported input", "", "| File | Result | HTTP | As expected |", "|---|---|---|---|"]
    for e in edge:
        L.append(f"| {e['file']} | {e['result']} | {e['http']} | {'✅' if e['as_expected'] else '❌'} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "scores.json").write_text(json.dumps({"per_field": per_field, "per_doc": per_doc, "errors": rows,
                                                 "edge": edge}, indent=2, default=str), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
