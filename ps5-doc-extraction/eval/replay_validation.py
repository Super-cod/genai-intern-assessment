"""Re-validate saved LLM outputs with the current validation code (no LLM call).

Each eval/results/<doc>.json contains the model's raw value + evidence for every field. This
script rebuilds the raw LLM response from those files and runs it through the CURRENT
text-extraction + validation stages, then scores it with the same rules as run_eval.py.

Use it to measure the effect of validation-layer changes on identical model outputs
(isolates code changes from LLM sampling), and when the full LLM run is too heavy for the machine.

Usage (from ps5-doc-extraction/):  python -m eval.replay_validation
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from app.doc_types import REGISTRY  # noqa: E402
from app.doc_types.invoice import FIELD_KINDS  # noqa: E402
from app.pipeline.text_extraction import extract_text  # noqa: E402
from app.pipeline.validation import validate_extraction  # noqa: E402
from eval.run_eval import score_field, score_items  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SAMPLES = ROOT / "samples"


def _s(v):
    return None if v is None else str(v)


def rebuild_raw(saved: dict):
    dtype = REGISTRY["invoice"]
    fields = {f: {"value": saved["fields"][f]["raw_value"], "evidence": saved["fields"][f]["evidence"]}
              for f in FIELD_KINDS}
    # line items were saved after parsing; feed them back as strings
    fields["line_items"] = [{k: _s(li.get(k)) for k in ("description", "quantity", "unit_price", "amount")}
                            for li in saved["line_items"]]
    return dtype.raw_model.model_validate(fields)


def main():
    truth = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))
    total, per_field, rows = Counter(), {f: Counter() for f in FIELD_KINDS}, []
    item_hits = item_total = 0
    for fname, exp in truth.items():
        saved = json.loads((RESULTS / f"{Path(fname).stem}.json").read_text(encoding="utf-8"))
        doc = pymupdf.open(SAMPLES / fname)
        extracted = extract_text(doc)
        doc.close()
        _, fields, items, issues, _ = validate_extraction(REGISTRY["invoice"], rebuild_raw(saved), extracted)
        fields = {k: v.model_dump(mode="json") for k, v in fields.items()}
        correct = 0
        for f, kind in FIELD_KINDS.items():
            o = score_field(kind, exp[f], fields[f])
            total[o] += 1
            per_field[f][o] += 1
            correct += o == "correct"
            if o != "correct":
                rows.append(f"| {fname} | {f} | {o} | {exp[f]} | {fields[f]['value']} | {fields[f]['status']} |")
        h, t = score_items(exp["line_items"], items)
        item_hits, item_total = item_hits + h, item_total + t
        print(f"{fname:40} {correct}/14  items {h}/{t}")
    n = sum(total.values())
    lines = ["# Replay: saved LLM outputs re-validated with current code (no LLM call)", "",
             f"- **Field accuracy: {total['correct']}/{n} = {total['correct'] / n:.1%}**",
             f"- Missed: {total['missed']} | Wrong: {total['wrong']} | Hallucinated: {total['hallucinated']} | "
             f"Flagged: {total['flagged']}",
             f"- Line items matched: {item_hits}/{item_total}", "",
             "| Field | Accuracy |", "|---|---|"]
    lines += [f"| {f} | {c['correct']}/{sum(c.values())} |" for f, c in per_field.items()]
    lines += ["", "| Document | Field | Outcome | Expected | Got | Status |", "|---|---|---|---|---|---|"] + rows
    (RESULTS / "summary_replay_final_code.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
