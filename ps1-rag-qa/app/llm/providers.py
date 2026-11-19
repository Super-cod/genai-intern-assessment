"""Concrete LLM / embedding providers.

- OllamaClient            : local models (default; no data leaves the machine)
- OpenAICompatibleClient  : any OpenAI-style /chat/completions API (OpenAI, Groq, Together,
                            vLLM, LM Studio, ...)
- AnthropicClient         : Claude via the official `anthropic` SDK (optional dependency)
"""
from __future__ import annotations

import json
import logging

import httpx

from .base import LLMClient, LLMError, LLMUnavailableError, Message

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- Ollama
class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=self.timeout)  # keep-alive: avoids per-call connect cost

    def _chat(self, messages, *, json_schema, temperature, max_tokens) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": self.seed},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama constrained decoding to the JSON schema
        try:
            r = self._http.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Ollama not reachable at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama timed out after {self.timeout}s") from exc
        if r.status_code == 404:
            raise LLMUnavailableError(f"Model '{self.model}' not found in Ollama (run: ollama pull {self.model})")
        if r.status_code >= 400:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleClient(LLMClient):
    def __init__(self, model: str, base_url: str, api_key: str = "", **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.Client(timeout=self.timeout)

    def _chat(self, messages, *, json_schema, temperature, max_tokens) -> str:
        msgs = list(messages)
        payload = {"model": self.model, "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens, "seed": self.seed}
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
            msgs.insert(0, {"role": "system",
                            "content": "Respond only with JSON matching this schema: " + json.dumps(json_schema)})
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = self._http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"LLM API not reachable at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM API timed out after {self.timeout}s") from exc
        if r.status_code in (401, 403):
            raise LLMUnavailableError(f"LLM API rejected credentials (HTTP {r.status_code})")
        if r.status_code >= 400:
            raise LLMError(f"LLM API HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- Anthropic (Claude)
class AnthropicClient(LLMClient):
    """Claude via the official SDK. Install with `pip install anthropic`; set ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-opus-5-5", **kw):
        super().__init__(model, **kw)
        try:
            import anthropic  # optional dependency
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError("pip install anthropic to use LLM_PROVIDER=anthropic") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)  # retries handled in base

    def _chat(self, messages, *, json_schema, temperature, max_tokens) -> str:
        a = self._anthropic
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if json_schema is not None:
            system += ("\n\nRespond only with a single JSON object matching this JSON schema "
                       "(no prose, no code fences):\n" + json.dumps(json_schema))
        kwargs = dict(
            model=self.model,
            max_tokens=max(max_tokens, 4096),  # headroom: adaptive thinking shares this budget
            messages=convo,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if system:
            kwargs["system"] = system
        try:
            resp = self._client.beta.messages.create(**kwargs)
        except a.AuthenticationError as exc:
            raise LLMUnavailableError(f"Anthropic auth failed: {exc}") from exc
        except a.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limited: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic HTTP {exc.status_code}: {exc}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("Anthropic model declined the request")
        return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- Embeddings
class EmbeddingClient:
    """Embeddings via Ollama (/api/embed) or an OpenAI-compatible /embeddings endpoint.

    `query_prefix` / `doc_prefix` support asymmetric retrieval prompts (e.g. EmbeddingGemma
    expects 'task: search result | query: ' for queries and 'title: none | text: ' for docs).
    """

    def __init__(self, provider: str, model: str, base_url: str, api_key: str = "",
                 query_prefix: str = "", doc_prefix: str = "", timeout: float = 120.0,
                 batch_size: int = 32, on_cpu: bool = False):
        self.provider, self.model, self.base_url = provider, model, base_url.rstrip("/")
        self.api_key, self.timeout, self.batch_size = api_key, timeout, batch_size
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.on_cpu = on_cpu
        self._http = httpx.Client(timeout=timeout)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                if self.provider == "ollama":
                    body = {"model": self.model, "input": batch}
                    if self.on_cpu:
                        body["options"] = {"num_gpu": 0}
                    r = self._http.post(f"{self.base_url}/api/embed", json=body)
                    r.raise_for_status()
                    out.extend(r.json()["embeddings"])
                else:
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    r = self._http.post(f"{self.base_url}/embeddings", headers=headers,
                                        json={"model": self.model, "input": batch})
                    r.raise_for_status()
                    out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
            except httpx.ConnectError as exc:
                raise LLMUnavailableError(f"Embedding service not reachable at {self.base_url}") from exc
            except httpx.HTTPError as exc:
                raise LLMError(f"Embedding request failed: {exc}") from exc
        return out


def build_llm(settings) -> LLMClient:
    p = settings.llm_provider.lower()
    common = dict(timeout=settings.llm_timeout, max_retries=settings.llm_max_retries)
    if p == "ollama":
        return OllamaClient(settings.llm_model, settings.llm_base_url, **common)
    if p in ("openai", "openai_compatible"):
        return OpenAICompatibleClient(settings.llm_model, settings.llm_base_url, settings.llm_api_key, **common)
    if p == "anthropic":
        return AnthropicClient(settings.llm_model, **common)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
