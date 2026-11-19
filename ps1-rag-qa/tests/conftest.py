"""Test fixtures: deterministic fake embedder + scripted fake LLM (no Ollama needed)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm.base import LLMClient, LLMUnavailableError  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402
from app.retrieval.index import tokenize  # noqa: E402

DIM = 8192


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get high cosine similarity."""
    model = "fake-embed"

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(DIM, dtype="float32")
        for w in tokenize(text):  # stopwords removed
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM(LLMClient):
    """Returns queued responses (str or dict); records the prompts it received."""

    def __init__(self, responses=None, down=False):
        super().__init__("fake-llm", max_retries=0)
        self.responses = list(responses or [])
        self.calls: list[list[dict]] = []
        self.down = down

    def _chat(self, messages, *, json_schema, temperature, max_tokens):
        if self.down:
            raise LLMUnavailableError("fake LLM is down")
        self.calls.append(messages)
        r = self.responses.pop(0) if self.responses else {"answerable": False, "answer": "", "cited_sources": []}
        return r if isinstance(r, str) else json.dumps(r)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", min_dense_score=0.15, embed_on_cpu=False)


@pytest.fixture
def make_pipeline(settings):
    def _make(responses=None, down=False):
        return RAGPipeline(settings, llm=FakeLLM(responses, down=down), embedder=FakeEmbedder())
    return _make


HANDBOOK = """Company Handbook
Effective: January 1, 2024

Annual Leave
Employees are entitled to 20 days of paid annual leave per year. Unused leave cannot be carried over.

Remote Work
Employees may work remotely up to 2 days per week with manager approval.
"""

UPDATE = """Leave Policy Update
Effective: January 1, 2025

Annual Leave
Employees are entitled to 24 days of paid annual leave per year. Up to 5 unused days may be carried over.
"""

MANUAL = """Robot Manual

Battery
The robot battery provides 8 hours of runtime. Charging takes 2.5 hours on the standard dock.

Error Codes
E204 means the battery is overheating. Let the robot cool for 30 minutes.
"""
