# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible (Kilo AI Gateway) serve LLM client.

Implements the same public surface as :class:`coa_serve.clients.bedrock.BedrockLLMClient`
(``converse``, ``converse_stream``, ``embed``, ``health_check``, ``close``,
``model_id``) so tier2/tier3 consumers are unchanged, but over an
OpenAI-compatible ``/chat/completions`` endpoint — by default the Kilo AI
Gateway (``COA_LLM_BASE_URL``). No AWS is involved in the LLM path.

Provider semantics that have no OpenAI equivalent degrade explicitly:

* **Guardrails** are a no-op locally: outcomes are always
  :attr:`GuardrailOutcome.NONE` and streams never raise
  :class:`GuardrailBlockedError`. When a ``guardrail_id`` is configured a
  decision metric (``ALLOW``) is still emitted so guardrail dashboards keep
  counting the guarded path.
* **Streaming** uses the OpenAI SSE protocol (``data:`` lines, ``[DONE]``
  terminator); usage is requested via ``stream_options.include_usage`` and
  recorded the same way as Bedrock Converse usage.
* **``effort``** (a Bedrock ``output_config`` concept) is accepted and ignored.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
import structlog
from coa_common import resolve_region
from coa_common.constants import DEFAULT_EMBED_MODEL_ID
from coa_common.guardrail_metrics import COMPONENT_NL_TO_SPARQL, DECISION_ALLOW, emit_guardrail_decision
from coa_common.llm import get_embedder
from coa_common.llm.openai import DEFAULT_BASE_URL, DEFAULT_CHAT_MODEL, _first_env, _first_env_int, _normalize_base_url

from ..llm_usage import REASON_PROVIDER_ERROR, current_tracker, record_usage
from .base import STOP_REASON_MAX_TOKENS, ConverseResult, GuardrailOutcome, instrumented

logger = structlog.get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=10, thread_name_prefix="openai-llm")

# Timeout for queue.get() to prevent deadlock if thread dies without signaling.
_STREAM_QUEUE_TIMEOUT_S = 300


def _post_json(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    """POST a JSON payload and return the parsed JSON body (raises on HTTP errors)."""
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


class OpenAICompatibleLLMClient:
    """Serve LLM client over an OpenAI-compatible endpoint (Kilo AI Gateway)."""

    def __init__(
        self,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        region: str | None = None,
        guardrail_version: str | None = None,
    ):
        """Configure the client's model, embedder, and endpoint.

        Args:
            model_id: Chat model id. Defaults to ``COA_CHAT_MODEL``, then the
                ``BEDROCK_MODEL_ID`` env, then a built-in default.
            embed_model_id: Embedding model id. Defaults to ``BEDROCK_EMBED_MODEL_ID``
                then ``DEFAULT_EMBED_MODEL_ID``.
            region: Accepted for ``BedrockLLMClient`` compatibility; unused here
                (the endpoint is chosen by ``COA_LLM_BASE_URL``).
            guardrail_version: Accepted for ``BedrockLLMClient`` compatibility;
                guardrails are a no-op in the OpenAI provider.
        """
        self._model_id = model_id or _first_env(DEFAULT_CHAT_MODEL, "KILO_MODEL", "COA_CHAT_MODEL", "COA_LLM_MODEL")
        self._embed_model_id = embed_model_id or _first_env(
            DEFAULT_EMBED_MODEL_ID, "KILO_EMBED_MODEL", "COA_EMBED_MODEL", "BEDROCK_EMBED_MODEL_ID"
        )
        self._base_url = _normalize_base_url(_first_env(DEFAULT_BASE_URL, "KILO_BASE_URL", "COA_LLM_BASE_URL"))
        self._api_key = _first_env("local", "KILO_API_KEY", "COA_LLM_API_KEY")
        self._read_timeout = _first_env_int(
            120, "KILO_TIMEOUT", "COA_CHAT_TIMEOUT", "COA_LLM_TIMEOUT", "BEDROCK_READ_TIMEOUT"
        )
        # Query embeddings go through the shared provider-aware embedder so serve
        # uses the SAME model as ingestion.
        self._embedder = get_embedder(
            model_id=self._embed_model_id, region=region or os.environ.get("BEDROCK_REGION", resolve_region())
        )
        self._guardrail_version = guardrail_version or os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
        logger.info(
            "openai_client_configured",
            model_id=self._model_id,
            embed_model_id=self._embed_model_id,
            base_url=self._base_url,
        )

    @property
    def model_id(self) -> str:
        """Return the default chat model id for this client."""
        return self._model_id

    async def close(self) -> None:
        """No-op: the requests-based transport holds no persistent client."""

    def _build_payload(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        guard_content: str | None,
        model_id: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the ``/chat/completions`` payload shared by converse() and converse_stream()."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user = prompt
        if guard_content:
            # No guardContent block exists in the OpenAI protocol; the guarded
            # text is sent as part of the user turn.
            user = f"{user}\n\n{guard_content}"
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": model_id or self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stream:
            payload["stream"] = True
            # Without include_usage the final chunk carries no token counts, so
            # streamed usage would be unrecordable.
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _emit_allow_decision(guardrail_id: str | None, latency_ms: float) -> None:
        """Emit the guardrail-decision metric for a locally-unguarded call.

        The OpenAI provider never intervenes; when a guardrail is configured the
        decision is always ALLOW so the dashboard still counts the guarded path
        instead of silently dropping the metric.
        """
        if not guardrail_id:
            return
        emit_guardrail_decision(
            component=COMPONENT_NL_TO_SPARQL,
            blocked=False,
            latency_ms=latency_ms,
            transport="emf",
            decision=DECISION_ALLOW,
        )

    @instrumented("openai")
    async def converse(
        self,
        prompt: str,
        *,
        system: str | None = None,
        guardrail_id: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        guard_content: str | None = None,
        model_id: str | None = None,
        effort: str | None = None,
    ) -> ConverseResult:
        """Send a single-turn prompt via the OpenAI-compatible chat completions API.

        Args:
            prompt: The main prompt text (context plus question, or just context).
            system: Optional system instruction text.
            guardrail_id: Accepted for ``BedrockLLMClient`` compatibility; the
                local provider never blocks (an ALLOW decision is emitted).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature; omitted when None.
            top_p: Nucleus-sampling cutoff; omitted when None.
            guard_content: Text appended to the user turn (no guardContent block
                exists in the OpenAI protocol).
            model_id: Per-call model override; defaults to the client's model.
            effort: Accepted for ``BedrockLLMClient`` compatibility; ignored.

        Returns:
            A ConverseResult with the generated text and outcome NONE.

        Raises:
            ValueError: If the response contains no text content.
        """
        payload = self._build_payload(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            guard_content=guard_content,
            model_id=model_id,
            stream=False,
        )
        tracker = current_tracker()
        handle = tracker.register_call("converse", payload["model"]) if tracker is not None else None

        def _call() -> dict:
            try:
                response = _post_json(f"{self._base_url}/chat/completions", self._api_key, payload, self._read_timeout)
            except BaseException:
                if handle is not None:
                    tracker.fail_call(handle, REASON_PROVIDER_ERROR)
                raise
            if handle is not None:
                record_usage(tracker, handle, response.get("usage"))
            return response

        loop = asyncio.get_running_loop()
        start = time.monotonic()
        response = await loop.run_in_executor(_EXECUTOR, _call)
        latency_ms = (time.monotonic() - start) * 1000

        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason")
        stop_reason = STOP_REASON_MAX_TOKENS if finish_reason == "length" else (finish_reason or "end_turn")
        text = (choice.get("message", {}) or {}).get("content") or ""
        if not text:
            raise ValueError(f"No text content in OpenAI response: finish_reason={finish_reason}")

        if stop_reason == STOP_REASON_MAX_TOKENS:
            logger.warning(
                "openai_output_truncated",
                model=payload["model"],
                max_tokens=max_tokens,
                text_chars=len(text),
            )

        self._emit_allow_decision(guardrail_id, latency_ms)
        return ConverseResult(text=text, outcome=GuardrailOutcome.NONE, stop_reason=stop_reason)

    async def converse_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        guardrail_id: str | None = None,
        max_tokens: int = 4096,
        guard_content: str | None = None,
        model_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream text tokens from the OpenAI-compatible chat completions API.

        Yields text fragments as they arrive (SSE ``data:`` events). The local
        provider never blocks, so :class:`GuardrailBlockedError` is never raised.

        Uses a thread pool executor to iterate the blocking ``requests`` stream,
        bridging to async via loop.call_soon_threadsafe + asyncio.Queue.
        """
        payload = self._build_payload(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=None,
            top_p=None,
            guard_content=guard_content,
            model_id=model_id,
            stream=True,
        )

        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stream_holder: dict[str, Any] = {}
        abandoned = threading.Event()
        usage_holder: dict[str, Any] = {}
        tracker = current_tracker()
        handle = tracker.register_call("converse_stream", payload["model"]) if tracker is not None else None

        def _run_stream() -> None:
            try:
                resp = requests.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self._read_timeout,
                    stream=True,
                )
                resp.raise_for_status()
                stream_holder["resp"] = resp
                if abandoned.is_set():
                    resp.close()
                    if handle is not None:
                        tracker.fail_call(handle, REASON_PROVIDER_ERROR)
                    return
                for line in resp.iter_lines():
                    if abandoned.is_set():
                        resp.close()
                        if handle is not None:
                            tracker.fail_call(handle, REASON_PROVIDER_ERROR)
                        return
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except ValueError:
                            continue
                        choices = event.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta", {}) or {}
                            text = delta.get("content") or ""
                            if text:
                                loop.call_soon_threadsafe(queue.put_nowait, text)
                        if event.get("usage"):
                            usage_holder["usage"] = event["usage"]
                if handle is not None and usage_holder.get("usage") is not None:
                    record_usage(tracker, handle, usage_holder["usage"])
                elif handle is not None:
                    from ..llm_usage import REASON_MISSING_USAGE

                    tracker.fail_call(handle, REASON_MISSING_USAGE)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:
                if handle is not None:
                    tracker.fail_call(handle, REASON_PROVIDER_ERROR)
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        loop.run_in_executor(_EXECUTOR, _run_stream)
        stream_start = time.monotonic()
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=_STREAM_QUEUE_TIMEOUT_S)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except TimeoutError:
            raise TimeoutError("OpenAI stream timed out waiting for next token") from None
        finally:
            abandoned.set()
            resp = stream_holder.get("resp")
            if resp is not None:
                resp.close()

        self._emit_allow_decision(guardrail_id, (time.monotonic() - stream_start) * 1000)

    @instrumented("openai")
    async def embed(self, text: str) -> list[float]:
        """Embed query text into a vector using the shared provider-aware embedder.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, self._embedder.embed_query, text)

    async def health_check(self) -> dict[str, Any]:
        """Probe endpoint reachability via a test embedding call.

        Returns:
            A status dict: ``{"status": "ok", ...}`` on success, otherwise
            ``{"status": "error", ...}``. This method never raises.
        """
        try:
            await self.embed("health")
            return {"status": "ok", "model": self._model_id}
        except Exception:
            return {"status": "error", "detail": "Health check failed"}
