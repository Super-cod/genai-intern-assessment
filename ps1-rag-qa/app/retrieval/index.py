"""Chunk index: FAISS (dense, cosine via inner product on L2-normalised vectors) + BM25 (sparse).

Persistence: chunks/doc records as JSON and vectors as .npy in DATA_DIR. The FAISS index is
rebuilt from the vectors on load (exact IndexFlatIP search is fast for <100k chunks and
avoids index/metadata drift).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ..ingestion.chunker import content_hash
from ..schemas import Chunk, DocumentRecord

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the and or of to in on for is are was were be been it its this that with as at by
from what which who whom how when where why do does did can could should would will shall may
i we you they he she our your their me my us them there here than then so if not no""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class ChunkIndex:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype="float32")
        self.docs: dict[str, DocumentRecord] = {}
        self._hash_to_idx: dict[str, int] = {}
        self._id_to_idx: dict[str, int] = {}
        self._faiss: faiss.IndexFlatIP | None = None
        self._bm25: BM25Okapi | None = None
        self.load()

    # ------------------------------------------------------------------ mutation
    def add_document(self, record: DocumentRecord, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add chunks; exact-duplicate chunks (same normalised text) are stored once and the
        extra document is recorded in `duplicate_of`. Returns number of chunks actually added."""
        with self.lock:
            vectors = _normalize(np.asarray(vectors, dtype="float32"))
            keep_chunks, keep_vecs = [], []
            for ch, vec in zip(chunks, vectors):
                h = content_hash(ch.text)
                if h in self._hash_to_idx:
                    existing = self.chunks[self._hash_to_idx[h]]
                    if ch.doc_name not in existing.duplicate_of and ch.doc_name != existing.doc_name:
                        existing.duplicate_of.append(ch.doc_name)
                    continue
                self._hash_to_idx[h] = len(self.chunks) + len(keep_chunks)
                keep_chunks.append(ch)
                keep_vecs.append(vec)
            if keep_chunks:
                new = np.vstack(keep_vecs)
                self.vectors = new if self.vectors.size == 0 else np.vstack([self.vectors, new])
                self.chunks.extend(keep_chunks)
            record.chunks = len(keep_chunks)
            self.docs[record.doc_id] = record
            self._rebuild()
            self.save()
            return len(keep_chunks)

    def remove_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            name = self.docs.pop(doc_id).doc_name
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype="float32")
            for c in self.chunks:
                if name in c.duplicate_of:
                    c.duplicate_of.remove(name)
            self._rebuild()
            self.save()
            return True

    def clear(self) -> None:
        with self.lock:
            self.chunks, self.docs = [], {}
            self.vectors = np.zeros((0, 0), dtype="float32")
            self._rebuild()
            self.save()

    def find_by_name(self, doc_name: str) -> DocumentRecord | None:
        return next((d for d in self.docs.values() if d.doc_name == doc_name), None)

    # ------------------------------------------------------------------ search
    @property
    def size(self) -> int:
        return len(self.chunks)

    def dense_search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self.lock:
            if not self._faiss or self.size == 0:
                return []
            q = _normalize(np.asarray(qvec, dtype="float32").reshape(1, -1))
            scores, idxs = self._faiss.search(q, min(k, self.size))
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self.lock:
            if not self._bm25 or self.size == 0:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            order = np.argsort(-scores)[:k]
            return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def cosine(self, qvec: np.ndarray, idx: int) -> float:
        q = _normalize(np.asarray(qvec, dtype="float32").reshape(1, -1))[0]
        return float(np.dot(self.vectors[idx], q))

    def pair_cosine(self, i: int, j: int) -> float:
        return float(np.dot(self.vectors[i], self.vectors[j]))

    def similarity_by_id(self, chunk_a: str, chunk_b: str) -> float:
        i, j = self._id_to_idx.get(chunk_a), self._id_to_idx.get(chunk_b)
        return 0.0 if i is None or j is None else self.pair_cosine(i, j)

    # ------------------------------------------------------------------ persistence
    def _rebuild(self) -> None:
        self._hash_to_idx = {content_hash(c.text): i for i, c in enumerate(self.chunks)}
        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        if self.size == 0:
            self._faiss, self._bm25 = None, None
            return
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])

    def save(self) -> None:
        (self.data_dir / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks]), encoding="utf-8")
        (self.data_dir / "docs.json").write_text(
            json.dumps({k: v.model_dump() for k, v in self.docs.items()}), encoding="utf-8")
        np.save(self.data_dir / "vectors.npy", self.vectors)

    def load(self) -> None:
        cpath, dpath, vpath = (self.data_dir / n for n in ("chunks.json", "docs.json", "vectors.npy"))
        if not (cpath.exists() and dpath.exists() and vpath.exists()):
            return
        self.chunks = [Chunk(**c) for c in json.loads(cpath.read_text(encoding="utf-8"))]
        self.docs = {k: DocumentRecord(**v) for k, v in json.loads(dpath.read_text(encoding="utf-8")).items()}
        self.vectors = np.load(vpath).astype("float32")
        if len(self.chunks) != len(self.vectors):  # corrupted store -> start clean rather than lie
            self.chunks, self.docs, self.vectors = [], {}, np.zeros((0, 0), dtype="float32")
        self._rebuild()


def _normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms
