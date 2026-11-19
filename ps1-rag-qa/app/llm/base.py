"""Provider-agnostic LLM interface.

Every part of the pipeline talks to `LLMClient`, never to a vendor SDK directly.
Swapping the local Ollama model for a hosted API (OpenAI-compatible or Anthropic)
is a configuration change (`LLM_PROVIDER`), not a code change.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM provider fails (network, timeout, bad status)."""


class LLMUnavailableError(LLMError):
    """Raised when the provider cannot be reached at all."""


class LLMOutputError(LLMError):
    """Raised when the model keeps returning output that does not match the schema."""


Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLMClient(ABC):
    """Minimal chat interface shared by all providers."""

    def __init__(self, model: str, timeout: float = 120.0, max_retries: int = 2, seed: int = 42):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.model})"

    @abstractmethod
    def _chat(self, messages: list[Message], *, json_schema: dict | None, temperature: float,
              max_tokens: int) -> str:
        """Single provider call. Must raise LLMError / LLMUnavailableError on failure."""

    def chat(self, messages: list[Message], *, json_schema: dict | None = None,
             temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Chat call with retry + exponential backoff on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._chat(messages, json_schema=json_schema, temperature=temperature,
                                  max_tokens=max_tokens)
            except LLMUnavailableError:
                raise  # no point retrying a server that is not running
            except LLMError as exc:
                last_exc = exc
                wait = 1.5 * (2 ** attempt)
                logger.warning("LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                               attempt + 1, self.max_retries + 1, exc, wait)
                time.sleep(wait)
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_exc}")

    def chat_json(self, messages: list[Message], schema_model: type[T], *,
                  temperature: float = 0.0, max_tokens: int = 1024, repair_attempts: int = 1) -> T:
        """Ask for JSON matching `schema_model`, validate with Pydantic, and repair once.

        If the model returns malformed JSON or JSON that fails validation, the validation
        error is fed back to the model so it can correct itself (bounded by repair_attempts).
        """
        schema = schema_model.model_json_schema()
        convo = list(messages)
        last_err = ""
        for attempt in range(repair_attempts + 1):
            raw = self.chat(convo, json_schema=schema, temperature=temperature, max_tokens=max_tokens)
            try:
                return schema_model.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:800]
                logger.warning("Invalid structured output (attempt %d): %s", attempt + 1, last_err)
                convo = list(messages) + [
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": (
                        "Your previous reply was not valid JSON for the required schema. "
                        f"Error: {last_err}\nReturn ONLY corrected JSON matching this schema:\n"
                        f"{json.dumps(schema)}")},
                ]
        raise LLMOutputError(f"Model output failed schema validation: {last_err}")

    def health(self) -> bool:
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=4)
            return True
        except LLMError:
            return False


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences and leading prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])  # may raise ValueError -> caller repairs
    raise ValueError("No JSON object found in model output")
