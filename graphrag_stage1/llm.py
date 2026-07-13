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
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _retry_after_seconds(response, default: int) -> float:
    """How long the server told us to wait, honouring Retry-After when present."""
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass  # Retry-After may be an HTTP date; fall back to the default
    return float(default)


def _extract_json(raw: str) -> str:
    """Pull the JSON object out of a response that may not be pure JSON.

    Backends that grammar-constrain output (local Ollama) always return bare JSON and
    this is a no-op. Backends that only *hint* at the schema (Ollama Cloud, most hosted
    APIs) may wrap it in ```json fences or bracket it with prose, so fall back to the
    outermost balanced {...} rather than failing the whole call.
    """
    text = _FENCE_RE.sub("", raw).strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text  # let json.loads raise -- the caller retries, then reports honestly


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
    # Ollama defaults to a 4096-token context. A single Stage 2 frame fits (~2.2k
    # prompt + ~0.8k output), but a BATCHED Stage 2 call does not -- N frames need
    # roughly N x 800 output tokens, and anything past the window is silently
    # truncated into invalid JSON. Set this when batching.
    num_ctx: int | None = None
    # Hosted backends (Ollama Cloud, any metered endpoint) answer 429 when throttled.
    # These control how patiently we wait it out; they are separate from `retries`,
    # which exists for transport errors.
    rate_limit_retries: int = 5
    rate_limit_backoff: int = 30  # seconds, when the server sends no Retry-After
    label: str = "LLM"  # used only in error messages

    @property
    def model_id(self) -> str:
        return self.model

    def resolved_model(self, stronger: bool = False) -> str:
        """The model name actually used for a call, honouring the escalation hint."""
        if stronger and self.stronger_model:
            return self.stronger_model
        return self.model

    def schema_is_enforced(self, model: str) -> bool:
        """Does this model's backend actually grammar-constrain output to ``format``?

        Local Ollama compiles the JSON schema into a decoding grammar, so off-schema
        output is impossible. Ollama **Cloud** treats ``format`` as a hint only -- a
        cloud model happily answers a schema-constrained call with prose ("u1: ...")
        or with a flattened object. Detect that and restate the contract in the prompt.
        """
        return "-cloud" not in model and not model.endswith(":cloud")

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        model = self.resolved_model(stronger)
        options: dict = {"temperature": self.temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if not self.schema_is_enforced(model):
            prompt = (
                f"{prompt}\n\n"
                "Respond with a single JSON object and NOTHING else -- no prose, no "
                "explanation, no markdown code fences. It must validate against this "
                f"JSON Schema exactly, including every required key:\n{json.dumps(schema)}"
            )
        last_error: Exception | None = None
        attempt = 0
        rate_limit_waits = 0
        while attempt <= self.retries:
            try:
                response = requests.post(
                    self.url,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "format": schema,
                        "options": options,
                    },
                    timeout=self.timeout,
                )
                # A hosted backend (Ollama Cloud, any metered API) answers 429 when the
                # rate limit or quota is hit. That is not a transport blip: retrying it
                # on the ordinary 1/2/4s backoff just burns the remaining retries and
                # loses the paragraph. Wait as long as the server asks, and do not count
                # it against `retries` -- being throttled is not a failure.
                if response.status_code == 429:
                    if rate_limit_waits >= self.rate_limit_retries:
                        raise RuntimeError(
                            f"Ollama {self.label}: rate limited (429) and still throttled "
                            f"after {rate_limit_waits} waits -- quota is likely exhausted"
                        )
                    delay = _retry_after_seconds(response, default=self.rate_limit_backoff)
                    rate_limit_waits += 1
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                raw = response.json()["response"].strip()
                raw = _THINK_RE.sub("", raw).strip()
                return json.loads(_extract_json(raw))
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                attempt += 1
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
