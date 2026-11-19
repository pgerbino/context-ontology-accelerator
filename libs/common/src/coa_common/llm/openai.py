# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible LLM and embedding clients (Kilo AI Gateway).

Implements the :class:`~coa_common.llm.protocol.LLMClient` and
:class:`~coa_common.llm.protocol.Embedder` protocols against any OpenAI-compatible
``/chat/completions`` and ``/embeddings`` endpoint — by default the Kilo AI
Gateway (``https://ai-gateway.kilo.ai/api/v1``).

The constructor signatures mirror ``BedrockClient`` / ``BedrockEmbedder`` so the
provider factory can swap implementations without touching call sites. Bedrock-only
options (region, guardrail ids, component, dimensions) are accepted and ignored
where they have no OpenAI equivalent; the guardrail behavior is a no-op here (see
plan Phase 3 for the local guardrail seam).

Semantics are kept at parity with ``coa_common.bedrock.BedrockClient.invoke``:
the same truncation error, input-size cap, code-fence stripping, JSON parsing,
usage accounting, and empty-response handling.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import requests

from coa_common.bedrock import (
    BedrockInvocationResult,
    BedrockTruncationError,
    InputTooLargeError,
    PromptSegment,
    prompt_text,
)

DEFAULT_BASE_URL = "https://ai-gateway.kilo.ai/api/v1"
DEFAULT_CHAT_MODEL = "~deepseek/deepseek-flash-latest"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_MAX_INPUT_CHARS = 200_000
_DEFAULT_READ_TIMEOUT = 90
_DEFAULT_EMBED_READ_TIMEOUT = 60
_DEFAULT_EMBED_BATCH_SIZE = 256
_OPENAI_MAX_EMBED_BATCH = 2048


def _post_json(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    """POST a JSON payload to an OpenAI-compatible endpoint and return the parsed body."""
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str) -> float | None:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return None


def _first_env(default: str, *names: str) -> str:
    """Return the first non-empty env var among ``names`` (KILO_* first), else ``default``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _first_env_int(default: int, *names: str) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            return _env_int(value, default)
    return default


def _first_env_float(*names: str) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _normalize_base_url(url: str) -> str:
    """Append the OpenAI version segment when the base URL lacks it.

    ``https://api.kilo.ai`` → ``https://api.kilo.ai/v1``; a URL already carrying
    the version segment (``https://ai-gateway.kilo.ai/api/v1``) is unchanged.
    """
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


class OpenAICompatibleClient:
    """Chat/completion client over an OpenAI-compatible endpoint.

    Args:
        region: Accepted for ``BedrockClient`` compatibility; unused here.
        model_id: Bedrock model id used as the model fallback when neither the
            ``model`` kwarg nor ``COA_CHAT_MODEL`` is set.
        read_timeout: HTTP read timeout in seconds (default 90).
        guardrail_id: Accepted for ``BedrockClient`` compatibility; guardrails
            are a no-op in the OpenAI provider (see plan Phase 3).
        guardrail_version: Accepted for compatibility; unused here.
        max_input_chars: Maximum combined prompt length in characters
            (default 200,000), matching ``BedrockClient``.
        component: Accepted for ``BedrockClient`` compatibility; unused here.
        base_url: OpenAI-compatible base URL; defaults to ``COA_LLM_BASE_URL``
            then ``https://ai-gateway.kilo.ai/api/v1``.
        api_key: Bearer token; defaults to ``COA_LLM_API_KEY`` then ``"local"``.
        model: Gateway model id (e.g. ``anthropic/claude-haiku-4.5``); defaults
            to ``COA_CHAT_MODEL``, then ``model_id``, then the default.
    """

    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
        read_timeout: int | None = None,
        guardrail_id: str | None = None,
        guardrail_version: str | None = None,
        max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
        component: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Resolve endpoint/model config (see class Args)."""
        self.base_url = _normalize_base_url(
            base_url or _first_env(DEFAULT_BASE_URL, "KILO_BASE_URL", "COA_LLM_BASE_URL")
        )
        self.api_key = api_key or _first_env("local", "KILO_API_KEY", "COA_LLM_API_KEY")
        self.model = model or _first_env(
            model_id or DEFAULT_CHAT_MODEL, "KILO_MODEL", "COA_CHAT_MODEL", "COA_LLM_MODEL"
        )
        self._read_timeout = (
            read_timeout
            if read_timeout is not None
            else _first_env_int(_DEFAULT_READ_TIMEOUT, "KILO_TIMEOUT", "COA_CHAT_TIMEOUT", "COA_LLM_TIMEOUT")
        )
        self._default_max_tokens = _first_env_int(4096, "KILO_MAX_TOKENS", "COA_CHAT_MAX_TOKENS")
        self._default_temperature = _first_env_float("KILO_TEMPERATURE", "COA_CHAT_TEMPERATURE")
        self._max_input_chars = max_input_chars
        self._component = component
        self._guardrail_id = guardrail_id
        self._region = region

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str | Sequence[PromptSegment],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> BedrockInvocationResult:
        """Call the model via ``/chat/completions`` and return a parsed result.

        ``user_prompt`` is either a plain string or an ordered list of
        :class:`PromptSegment`; segments are joined in order (the guardrail
        guarded/unguarded distinction is a Bedrock concept and has no effect
        here). ``max_tokens``/``temperature`` default to the gateway env
        (``KILO_MAX_TOKENS``/``KILO_TEMPERATURE``, or their ``COA_*`` aliases).
        """
        user_text = prompt_text(user_prompt)
        input_chars = len(system_prompt) + len(user_text)
        if input_chars > self._max_input_chars:
            raise InputTooLargeError(
                f"Combined prompt is {input_chars} chars, exceeds limit of {self._max_input_chars}"
            )
        max_tokens = max_tokens or self._default_max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
        }
        effective_temperature = temperature if temperature is not None else self._default_temperature
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        start = time.monotonic()
        response = _post_json(
            f"{self.base_url}/chat/completions",
            self.api_key,
            payload,
            self._read_timeout,
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = response.get("usage", {}) or {}
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        choice = response["choices"][0]

        if choice.get("finish_reason") == "length":
            raise BedrockTruncationError(
                model_id=self.model,
                output_tokens=output_tokens,
                max_tokens=max_tokens,
            )

        content = (choice.get("message", {}) or {}).get("content") or ""
        if not content.strip():
            raise ValueError("OpenAI returned empty response")
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            result = json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid OpenAI response format: {e}") from e
        return BedrockInvocationResult(
            result=result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )


class OpenAICompatibleEmbedder:
    """Embedding client over an OpenAI-compatible endpoint.

    Args:
        model_id: Bedrock model id used as the model fallback when neither the
            ``model`` kwarg nor ``COA_EMBED_MODEL`` is set.
        dimensions: Accepted for ``BedrockEmbedder`` compatibility; the output
            dimensionality is fixed by the served model and is not requested.
        region: Accepted for ``BedrockEmbedder`` compatibility; unused here.
        max_workers: Accepted for ``BedrockEmbedder`` compatibility; unused here
            (batches are sent sequentially, order preserved).
        read_timeout: HTTP read timeout in seconds (default 60).
        batch_size: Texts per ``/embeddings`` request (clamped to [1, 2048]).
            Defaults to ``COA_EMBED_BATCH_SIZE`` then 256.
        cost_tracker: Optional per-job usage accumulator; records input tokens
            under the ``"embed"`` stage on every successful call.
        base_url: OpenAI-compatible base URL; defaults to ``COA_LLM_BASE_URL``
            then ``https://ai-gateway.kilo.ai/api/v1``.
        api_key: Bearer token; defaults to ``COA_LLM_API_KEY`` then ``"local"``.
        model: Gateway embedding model id; defaults to ``COA_EMBED_MODEL``,
            then ``model_id``, then the default.
    """

    def __init__(
        self,
        model_id: str | None = None,
        dimensions: int | None = None,
        region: str | None = None,
        max_workers: int = 10,
        read_timeout: int | None = None,
        batch_size: int | None = None,
        cost_tracker=None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Resolve endpoint/model config and clamp the batch size (see class Args)."""
        self.model = model or _first_env(
            model_id or DEFAULT_EMBED_MODEL, "KILO_EMBED_MODEL", "COA_EMBED_MODEL", "COA_LLM_EMBED_MODEL"
        )
        self.dimensions = dimensions
        embed_base = base_url or _first_env("", "KILO_EMBED_BASE_URL", "COA_LLM_EMBED_BASE_URL")
        if not embed_base:
            embed_base = _first_env(DEFAULT_BASE_URL, "KILO_BASE_URL", "COA_LLM_BASE_URL")
        self._base_url = _normalize_base_url(embed_base)
        self._api_key = api_key or _first_env("local", "KILO_API_KEY", "COA_LLM_API_KEY")
        self._read_timeout = (
            read_timeout
            if read_timeout is not None
            else _first_env_int(
                _DEFAULT_EMBED_READ_TIMEOUT, "KILO_TIMEOUT", "COA_LLM_TIMEOUT", "BEDROCK_EMBED_READ_TIMEOUT"
            )
        )
        self._batch_size = max(
            1,
            min(
                _OPENAI_MAX_EMBED_BATCH,
                batch_size
                if batch_size is not None
                else _first_env_int(_DEFAULT_EMBED_BATCH_SIZE, "KILO_EMBED_BATCH_SIZE", "COA_EMBED_BATCH_SIZE"),
            ),
        )
        self._cost_tracker = cost_tracker

    def set_cost_tracker(self, cost_tracker) -> None:
        """Attach (or clear) a per-job cost tracker after construction."""
        self._cost_tracker = cost_tracker

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        chunks = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        vectors: list[list[float]] = []
        for chunk in chunks:
            payload = {"model": self.model, "input": chunk[0] if len(chunk) == 1 else chunk}
            response = _post_json(f"{self._base_url}/embeddings", self._api_key, payload, self._read_timeout)
            data = response.get("data", [])
            if len(data) != len(chunk):
                raise ValueError(f"OpenAI returned {len(data)} embeddings for {len(chunk)} inputs (model={self.model})")
            vectors.extend(item["embedding"] for item in data)
            self._record_cost(chunk)
        return vectors

    @staticmethod
    def _input_tokens(text: str) -> int:
        # Best-effort ~4 chars/token estimate; OpenAI-compatible embedding
        # responses do not include a token count.
        return max(1, math.ceil(len(text) / 4))

    def _record_cost(self, texts: list[str]) -> None:
        if self._cost_tracker is None:
            return
        self._cost_tracker.record("embed", self.model, sum(self._input_tokens(t) for t in texts), 0)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (symmetric — no input-type distinction)."""
        return self._embed_texts([text])[0]

    def embed_document(self, text: str) -> list[float]:
        """Embed indexable content (symmetric — no input-type distinction)."""
        return self._embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents, order preserved, one vector per input."""
        return self._embed_texts(texts)


def make_openai_llama_index_embedding(
    model: str | None = None,
    dimensions: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """A LlamaIndex ``BaseEmbedding`` backed by :class:`OpenAICompatibleEmbedder`.

    Mirrors ``coa_common.embeddings.make_llama_index_embedding``: pass an instance
    (not a model-id string) so retrieval uses the same OpenAI-compatible endpoint
    and model as ingestion. Imported lazily because only the graphrag paths
    depend on ``llama_index``. The adapter must be picklable (graphrag pickles
    ``embed_model`` to worker processes), so it is a module-level class
    reconstructed from plain fields.
    """
    cls = _llama_openai_embedder_cls()
    return cls(
        model_name=model or _first_env(DEFAULT_EMBED_MODEL, "KILO_EMBED_MODEL", "COA_EMBED_MODEL"),
        embed_model=model or _first_env(DEFAULT_EMBED_MODEL, "KILO_EMBED_MODEL", "COA_EMBED_MODEL"),
        embed_dimensions=dimensions,
        embed_base_url=base_url,
        embed_api_key=api_key,
    )


def make_openai_llama_index_llm(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """A LlamaIndex ``OpenAI`` LLM pointed at the configured OpenAI-compatible endpoint.

    Mirrors ``make_openai_llama_index_embedding`` for the LLM side: pass an
    INSTANCE (not a model string) so the graphrag toolkit's ``to_llm`` returns it
    unchanged instead of building a Bedrock Converse client. Imported lazily
    because only the graphrag paths depend on ``llama_index``.
    """
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=model or _first_env(DEFAULT_CHAT_MODEL, "KILO_MODEL", "COA_CHAT_MODEL", "COA_LLM_MODEL"),
        api_key=api_key or _first_env("local", "KILO_API_KEY", "COA_LLM_API_KEY"),
        api_base=_normalize_base_url(base_url or _first_env(DEFAULT_BASE_URL, "KILO_BASE_URL", "COA_LLM_BASE_URL")),
    )


# Cache the dynamically-built adapter class at module scope so instances pickle
# by qualified name (coa_common.llm.openai.<cls>) across processes.
_LLAMA_ADAPTER_CLS: Any = None


def _llama_openai_embedder_cls() -> Any:
    global _LLAMA_ADAPTER_CLS
    if _LLAMA_ADAPTER_CLS is not None:
        return _LLAMA_ADAPTER_CLS

    from llama_index.core.base.embeddings.base import BaseEmbedding
    from pydantic import PrivateAttr

    class OpenAIEmbedderLlamaIndex(BaseEmbedding):
        """Picklable LlamaIndex adapter over :class:`OpenAICompatibleEmbedder`.

        The embedder is rebuilt lazily from plain fields, so instances carry no
        unpicklable client across process boundaries.
        """

        embed_model: str
        embed_dimensions: int | None = None
        embed_base_url: str | None = None
        embed_api_key: str | None = None
        _embedder: OpenAICompatibleEmbedder | None = PrivateAttr(default=None)

        def _get_embedder(self) -> OpenAICompatibleEmbedder:
            if self._embedder is None:
                self._embedder = OpenAICompatibleEmbedder(
                    model=self.embed_model,
                    dimensions=self.embed_dimensions,
                    base_url=self.embed_base_url,
                    api_key=self.embed_api_key,
                )
            return self._embedder

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_embedder().embed_query(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_embedder().embed_query(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._get_embedder().embed_document(text)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_embedder().embed_document(text)

    # Expose at module scope so pickle can resolve it by qualified name.
    OpenAIEmbedderLlamaIndex.__module__ = __name__
    OpenAIEmbedderLlamaIndex.__qualname__ = "OpenAIEmbedderLlamaIndex"
    globals()["OpenAIEmbedderLlamaIndex"] = OpenAIEmbedderLlamaIndex
    _LLAMA_ADAPTER_CLS = OpenAIEmbedderLlamaIndex
    return _LLAMA_ADAPTER_CLS


# Names this module builds on first access instead of at import time.
_LAZY_ATTRS = frozenset({"OpenAIEmbedderLlamaIndex"})

if not TYPE_CHECKING:
    # Hidden from type checkers so a typo in an import is still reported as a
    # missing attribute instead of silently typing as ``Any``.

    def __getattr__(name: str) -> Any:
        """Build the lazy LlamaIndex adapter class when it is looked up by name.

        Pickle stores an adapter instance as
        ``coa_common.llm.openai.OpenAIEmbedderLlamaIndex``. A worker started with
        the ``spawn`` or ``forkserver`` method imports this module fresh and never
        calls :func:`make_openai_llama_index_embedding`, so the name would not
        exist there and unpickling would fail, killing the worker pool. This hook
        (PEP 562; only consulted when normal lookup fails) builds and caches the
        class on first access.
        """
        if name in _LAZY_ATTRS:
            return _llama_openai_embedder_cls()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
