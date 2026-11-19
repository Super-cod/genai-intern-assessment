"""Evaluation harness for the RAG system.

Measures
  1. Retrieval quality for dense / BM25 / hybrid (Hit@1, Hit@3, Hit@5, MRR) on answerable queries.
     A hit = a retrieved chunk from an expected document that contains the evidence string.
  2. Relevance-gate calibration: best dense score per query (answerable vs unanswerable).
  3. End-to-end answers (hybrid): correct status, answer keywords, citation correctness,
     grounding score, conflict note, latency.
  4. Ingestion edge cases (corrupt / empty / scanned / unsupported files).

Usage (from ps1-rag-qa/):  python -m eval.run_eval            # everything (needs Ollama)
                           python -m eval.run_eval --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.errors import RAGError  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402
from app.schemas import QueryRequest  # noqa: E402

OUT = ROOT / "eval" / "results"
DOCS = ROOT / "sample_docs"
REFUSAL_STATUSES = {"not_found", "insufficient_context", "unsupported"}


def build_pipeline() -> RAGPipeline:
    p = RAGPipeline(Settings(data_dir=ROOT / "data_eval"))
    p.index.clear()
    for f in sorted(DOCS.glob("*.*")):
        if f.suffix in (".pdf", ".txt"):
            p.ingest(f.name, f.read_bytes())
    return p


def kw_match(text: str, patterns: list[str]) -> bool:
    return all(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_hit(chunk, q) -> bool:
    docs = {chunk.doc_name, *chunk.duplicate_of}
    return bool(docs & set(q["expected_docs"])) and q["evidence"].lower() in chunk.text.lower()


def eval_retrieval(p: RAGPipeline, queries: list[dict]) -> dict:
    answerable = [q for q in queries if q["expect"] == "answer"]
    report = {}
    for mode in ("dense", "bm25", "hybrid"):
        rows = []
        for q in answerable:
            query = q["question"] if q["type"] != "followup" else q["history"][0] + " " + q["question"]
            res = p.retriever.retrieve(query, k=5, mode=mode, apply_gate=False)
            rank = next((i + 1 for i, rc in enumerate(res.chunks) if is_hit(rc.chunk, q)), None)
            rows.append({"id": q["id"], "type": q["type"], "rank": rank,
                         "top1": f"{res.chunks[0].chunk.doc_name} | {res.chunks[0].chunk.section}" if res.chunks else None})
        n = len(rows)
        report[mode] = {
            "hit@1": round(sum(r["rank"] == 1 for r in rows) / n, 3),
            "hit@3": round(sum(r["rank"] is not None and r["rank"] <= 3 for r in rows) / n, 3),
            "hit@5": round(sum(r["rank"] is not None for r in rows) / n, 3),
            "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
            "misses": [r for r in rows if r["rank"] != 1],
        }
    return report


def eval_gate(p: RAGPipeline, queries: list[dict]) -> list[dict]:
    rows = []
    for q in queries:
        res = p.retriever.retrieve(q["question"], k=5, mode="hybrid", apply_gate=True)
        rows.append({"id": q["id"], "expect": q["expect"], "best_dense": round(res.best_dense, 3),
                     "gated_out": res.gated_out})
    return rows


def eval_e2e(p: RAGPipeline, queries: list[dict]) -> list[dict]:
    rows = []
    for q in queries:
        sid = None
        for h in q.get("history", []):
            sid = p.query(QueryRequest(question=h, session_id=sid)).session_id
        t = time.perf_counter()
        r = p.query(QueryRequest(question=q["question"], session_id=sid))
        latency = time.perf_counter() - t
        status = r.status.value
        cited_docs = {c.doc_name for c in r.citations}
        row = {"id": q["id"], "type": q["type"], "question": q["question"],
               "standalone_question": r.standalone_question, "status": status, "answer": r.answer,
               "citations": [f"{c.label}:{c.doc_name}" for c in r.citations],
               "grounding": r.grounding_score, "conflict_note": r.conflict_note,
               "latency_s": round(latency, 1)}
        if q["expect"] == "answer":
            row["status_ok"] = status == "answered"
            row["answer_ok"] = status == "answered" and kw_match(r.answer, q["answer_keywords"])
            row["citation_ok"] = bool(cited_docs & set(q["expected_docs"]))
            if q.get("expect_conflict_note"):
                row["conflict_note_ok"] = bool(r.conflict_note)
        elif q["expect"] == "refuse":
            row["status_ok"] = row["answer_ok"] = status in REFUSAL_STATUSES
        else:  # refuse_or_qualified
            ok = status in REFUSAL_STATUSES or kw_match(r.answer, q["qualified_keywords"])
            row["status_ok"] = row["answer_ok"] = ok
        rows.append(row)
        print(f"  {q['id']:4} {status:22} ok={row['answer_ok']!s:5} {latency:5.1f}s  {r.answer[:90]}")
    return rows


def eval_edge_cases(p: RAGPipeline) -> list[dict]:
    rows = []
    for f in sorted((DOCS / "edge_cases").glob("*")):
        try:
            res = p.ingest(f.name, f.read_bytes())
            rows.append({"file": f.name, "accepted": True, "chunks": res.chunks, "warnings": res.warnings})
        except RAGError as exc:
            rows.append({"file": f.name, "accepted": False, "error": exc.code, "http": exc.status_code,
                         "message": exc.message})
    return rows


def write_summary(ret, gate, e2e, edge, settings):
    lines = ["# RAG evaluation results", "",
             f"LLM: `{settings.llm_provider}:{settings.llm_model}` | embeddings: `{settings.embed_model}` | "
             f"chunk_size={settings.chunk_size} overlap={settings.chunk_overlap} top_k={settings.top_k} "
             f"min_dense_score={settings.min_dense_score}", "",
             "## 1. Retrieval quality (answerable queries, top-5, no gate)", "",
             "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for m, r in ret.items():
        lines.append(f"| {m} | {r['hit@1']} | {r['hit@3']} | {r['hit@5']} | {r['mrr']} |")
    lines += ["", "Misses / non-top-1 per mode:", ""]
    for m, r in ret.items():
        for x in r["misses"]:
            lines.append(f"- {m}: {x['id']} ({x['type']}) rank={x['rank']} top1=`{x['top1']}`")
    lines += ["", "## 2. Relevance gate calibration (best dense cosine per query)", "",
              "| Query | Expected | Best dense | Gated out |", "|---|---|---|---|"]
    lines += [f"| {g['id']} | {g['expect']} | {g['best_dense']} | {g['gated_out']} |" for g in gate]
    if e2e:
        ans = [r for r in e2e if r["id"].startswith("Q")]
        una = [r for r in e2e if r["id"].startswith("U")]
        acc = lambda rows, k: f"{sum(bool(r.get(k)) for r in rows)}/{len(rows)}"
        cn = [r for r in ans if "conflict_note_ok" in r]
        lat = [r["latency_s"] for r in e2e]
        lines += ["", "## 3. End-to-end answers (hybrid retrieval)", "",
                  f"- Answerable: correct answer {acc(ans, 'answer_ok')}, cited an expected document {acc(ans, 'citation_ok')}",
                  f"- Unanswerable: correctly refused {acc(una, 'answer_ok')}",
                  f"- Conflict note produced when required: {acc(cn, 'conflict_note_ok')}",
                  f"- Mean grounding score (answered): "
                  f"{round(statistics.mean([r['grounding'] for r in e2e if r['grounding'] is not None] or [0]), 3)}",
                  f"- Latency per query: median {statistics.median(lat)}s, max {max(lat)}s", "",
                  "| ID | Type | Status | OK | Cited | Grounding | Answer |", "|---|---|---|---|---|---|---|"]
        for r in e2e:
            a = r["answer"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['id']} | {r['type']} | {r['status']} | {'✅' if r['answer_ok'] else '❌'} | "
                         f"{', '.join(r['citations'])} | {r['grounding']} | {a[:160]} |")
    lines += ["", "## 4. Ingestion edge cases", "", "| File | Accepted | Result |", "|---|---|---|"]
    for e in edge:
        res = f"{e['chunks']} chunks" if e["accepted"] else f"HTTP {e['http']} `{e['error']}`: {e['message']}"
        lines.append(f"| {e['file']} | {e['accepted']} | {res} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = json.loads((ROOT / "eval" / "queries.json").read_text(encoding="utf-8"))
    print("Indexing sample documents...")
    p = build_pipeline()
    print("Retrieval evaluation...")
    ret = eval_retrieval(p, queries)
    for m, r in ret.items():
        print(f"  {m:6} hit@1={r['hit@1']} hit@3={r['hit@3']} hit@5={r['hit@5']} mrr={r['mrr']}")
    gate = eval_gate(p, queries)
    e2e = []
    if not args.retrieval_only:
        print("End-to-end evaluation (this calls the LLM)...")
        e2e = eval_e2e(p, queries)
    edge = eval_edge_cases(p)
    (OUT / "results.json").write_text(json.dumps(
        {"retrieval": ret, "gate": gate, "e2e": e2e, "edge_cases": edge}, indent=2), encoding="utf-8")
    write_summary(ret, gate, e2e, edge, p.s)
    print(f"Wrote {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
