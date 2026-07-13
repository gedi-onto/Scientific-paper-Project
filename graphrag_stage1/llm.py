"""LLM transport abstraction.

The pipeline needs exactly one capability from a language model: given a prompt
and a JSON Schema, return schema-valid JSON. That contract is the ``LLMClient``
protocol below. Stage 1 and Stage 2 depend only on the protocol, so a host
platform can plug in its own model (Anthropic, OpenAI, vLLM, a mock, ...) by
implementing ``complete`` -- no localhost or Ollama assumption is baked in.

``OllamaClient`` is the batteries-included default (local ``qwen3:8b`` via the
Ollama HTTP API), constructed from environment variables for backward
compatibility with the original scripts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import requests

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract the pipeline requires of a language model.

    Implement this to use your own provider. ``complete`` must return a Python
    dict parsed from schema-valid JSON. ``stronger`` is an optional hint that
    the caller wants a more capable model for this call (Stage 2's
    ``REPROCESS_WITH_STRONGER_MODEL`` route); clients that do not distinguish
    tiers may ignore it.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier for the default model, recorded in provenance."""
        ...

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        ...


@dataclass
class OllamaClient:
    """Default ``LLMClient`` backed by a local Ollama server.

    All knobs are explicit so nothing is read from module globals at call time;
    use :meth:`from_env` to reproduce the original environment-variable defaults.
    """

    model: str = "qwen3:8b"
    stronger_model: str | None = None
    url: str = "http://localhost:11434/api/generate"
    timeout: int = 180
    retries: int = 2
    temperature: float = 0.0
    label: str = "LLM"  # used only in error messages

    @property
    def model_id(self) -> str:
        return self.model

    def resolved_model(self, stronger: bool = False) -> str:
        """The model name actually used for a call, honouring the escalation hint."""
        if stronger and self.stronger_model:
            return self.stronger_model
        return self.model

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        model = self.resolved_model(stronger)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "format": schema,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()["response"].strip()
                raw = _THINK_RE.sub("", raw).strip()
                return json.loads(raw)
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Ollama {self.label} failed after {self.retries + 1} attempts"
        ) from last_error

    @classmethod
    def from_env(
        cls,
        model_env: str,
        *,
        default_model: str = "qwen3:8b",
        stronger_env: str | None = None,
        label: str = "LLM",
    ) -> "OllamaClient":
        """Build an OllamaClient from environment variables (legacy defaults)."""
        model = os.getenv(model_env, default_model)
        stronger = os.getenv(stronger_env, model) if stronger_env else None
        return cls(
            model=model,
            stronger_model=stronger,
            url=os.getenv("OLLAMA_URL", cls.url),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", str(cls.timeout))),
            retries=int(os.getenv("OLLAMA_RETRIES", str(cls.retries))),
            label=label,
        )


def resolved_model_name(client: LLMClient, stronger: bool = False) -> str:
    """Model name used for a call, for provenance stamping.

    Falls back to ``model_id`` for clients that do not expose tiered models.
    """
    resolver = getattr(client, "resolved_model", None)
    if callable(resolver):
        return resolver(stronger)
    return client.model_id


class BoundedClient:
    """Wrap any ``LLMClient`` to cap the number of concurrent ``complete`` calls.

    When paragraphs and statements are processed in parallel, the total number
    of in-flight model calls is ``paragraph_workers x statement_workers`` -- easy
    to blow past a provider's rate limit. Wrapping the client in a ``BoundedClient``
    enforces one global ceiling regardless of how wide the pipeline fans out, so
    you can parallelize freely and still respect the limit.

        client = BoundedClient(MyClient(), max_concurrency=16)

    Pass ``share_limit_with`` to make two clients (e.g. a Stage 1 and a Stage 2
    model) draw from one shared ceiling instead of each getting their own.
    """

    def __init__(
        self,
        inner: LLMClient,
        max_concurrency: int,
        *,
        share_limit_with: "BoundedClient | None" = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._inner = inner
        if share_limit_with is not None:
            self._semaphore = share_limit_with._semaphore
            self.max_concurrency = share_limit_with.max_concurrency
        else:
            self._semaphore = threading.Semaphore(max_concurrency)
            self.max_concurrency = max_concurrency

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def resolved_model(self, stronger: bool = False) -> str:
        return resolved_model_name(self._inner, stronger)

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        with self._semaphore:
            return self._inner.complete(prompt, schema, stronger=stronger)
