"""Session memory and follow-up question rewriting."""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from ..llm import LLMClient, LLMError
from ..schemas import RewrittenQuery
from . import prompts

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Session:
    turns: deque = field(default_factory=lambda: deque(maxlen=20))
    last_used: float = field(default_factory=time.time)


class SessionStore:
    """In-memory sessions with TTL eviction. Swap for Redis in a multi-instance deployment."""

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000):
        self._s: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.ttl, self.max_sessions = ttl_seconds, max_sessions

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        with self._lock:
            self._evict()
            sid = session_id or uuid.uuid4().hex[:12]
            sess = self._s.setdefault(sid, Session())
            sess.last_used = time.time()
            return sid, sess

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._s.pop(session_id, None) is not None

    def _evict(self):
        now = time.time()
        for sid in [k for k, v in self._s.items() if now - v.last_used > self.ttl]:
            del self._s[sid]
        while len(self._s) > self.max_sessions:
            del self._s[min(self._s, key=lambda k: self._s[k].last_used)]


def format_history(session: Session, n_turns: int) -> str:
    turns = list(session.turns)[-n_turns:]
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer[:500]}" for t in turns)


_REFERRING = re.compile(
    r"\b(it|its|it's|they|them|their|this|that|these|those|he|she|his|her|there|same|above|"
    r"previous|former|latter|one|ones|also|else|more|again|what about|how about|and)\b", re.I)


def needs_rewrite(question: str) -> bool:
    """Cheap heuristic: only follow-ups that refer back (pronouns, 'what about', very short
    questions) need the extra LLM call. Saves one LLM round-trip on standalone questions."""
    return len(question.split()) <= 4 or bool(_REFERRING.search(question))


def rewrite_question(llm: LLMClient, question: str, history: str) -> str:
    """Turn a follow-up ("how long does it take to charge?") into a standalone retrieval query."""
    if not history or not needs_rewrite(question):
        return question
    try:
        out = llm.chat_json(
            [{"role": "system", "content": prompts.REWRITE_SYSTEM},
             {"role": "user", "content": prompts.REWRITE_USER.format(history=history, question=question)}],
            RewrittenQuery, max_tokens=120)
        rewritten = out.standalone_question.strip()
        return rewritten if 3 <= len(rewritten) <= 500 else question
    except LLMError as exc:
        # Degrade gracefully: retrieval with the raw question is still better than failing.
        logger.warning("Query rewrite failed (%s); using raw question", exc)
        return question
