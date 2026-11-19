# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the serve OpenAI-compatible (Kilo) LLM client.

Covers the provider-facing surface that tier2/tier3 consumers use through
``ClientSet.llm``: single-shot ``converse``, SSE ``converse_stream``, shared
``embed``, guardrail no-op semantics, and usage accounting. Mirrors
``test_bedrock.py`` in shape so the two providers are held to the same contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from coa_serve.clients.base import (
    STOP_REASON_MAX_TOKENS,
    GuardrailOutcome,
)
from coa_serve.clients.openai import OpenAICompatibleLLMClient


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeStreamResponse:
    def __init__(self, lines: list[bytes], status: int = 200) -> None:
        self._lines = lines
        self.status_code = status
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        return iter(self._lines)

    def close(self) -> None:
        self.closed = True


def _client(**kwargs) -> OpenAICompatibleLLMClient:
    kwargs.setdefault("model_id", "anthropic/claude-haiku-4.5")
    return OpenAICompatibleLLMClient(**kwargs)


def _chat_payload(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_converse_posts_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_chat_payload("SELECT 1")))

    def _post(url, headers=None, json=None, timeout=None, stream=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return fake(url, headers=headers, json=json, timeout=timeout, stream=stream)

    monkeypatch.setattr("coa_serve.clients.openai.requests.post", _post)
    client = _client()
    result = await client.converse("What tables?", system="You are a helper")

    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer local"
    body = captured["json"]
    assert body["model"] == "anthropic/claude-haiku-4.5"
    assert body["messages"] == [
        {"role": "system", "content": "You are a helper"},
        {"role": "user", "content": "What tables?"},
    ]
    assert result.text == "SELECT 1"
    assert result.outcome is GuardrailOutcome.NONE
    assert result.guardrail_blocked is False
    assert result.truncated is False


@pytest.mark.asyncio
async def test_converse_honours_sampling_and_per_call_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_chat_payload("x")))

    def _post(url, headers=None, json=None, timeout=None, stream=None):
        captured["json"] = json
        return fake(url, headers=headers, json=json, timeout=timeout, stream=stream)

    monkeypatch.setattr("coa_serve.clients.openai.requests.post", _post)
    client = _client()
    result = await client.converse(
        "q", system=None, max_tokens=1024, temperature=0.0, top_p=0.5, guard_content="g", model_id="other-model"
    )
    body = captured["json"]
    assert body["model"] == "other-model"
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.0
    assert body["top_p"] == 0.5
    assert result.text == "x"


@pytest.mark.asyncio
async def test_converse_truncation_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeResponse(_chat_payload("partial", finish_reason="length")),
    )
    result = await _client().converse("q")
    assert result.truncated is True
    assert result.stop_reason == STOP_REASON_MAX_TOKENS


@pytest.mark.asyncio
async def test_converse_empty_text_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeResponse(_chat_payload("")),
    )
    with pytest.raises(ValueError, match="No text content"):
        await _client().converse("q")


@pytest.mark.asyncio
async def test_converse_guardrail_id_emits_allow_and_never_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    decisions: list[dict] = []
    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeResponse(_chat_payload("ok")),
    )
    monkeypatch.setattr(
        "coa_serve.clients.openai.emit_guardrail_decision",
        lambda **kw: decisions.append(kw),
    )
    result = await _client().converse("q", guardrail_id="gr-1")
    assert result.outcome is GuardrailOutcome.NONE
    assert decisions and decisions[0]["decision"] == "ALLOW"


@pytest.mark.asyncio
async def test_converse_records_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from coa_serve.llm_usage import LlmUsageTracker, bind_tracker

    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeResponse(_chat_payload("x")),
    )
    tracker = LlmUsageTracker()
    with bind_tracker(tracker):
        await _client().converse("q")
    snapshot = tracker.snapshot()
    assert snapshot.complete is True
    assert snapshot.input_tokens == 10
    assert snapshot.output_tokens == 5


@pytest.mark.asyncio
async def test_converse_stream_yields_tokens_and_records_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
        b"data: [DONE]",
    ]

    def _post(url, headers=None, json=None, timeout=None, stream=None):
        captured["json"] = json
        return _FakeStreamResponse(lines)

    monkeypatch.setattr("coa_serve.clients.openai.requests.post", _post)
    from coa_serve.llm_usage import LlmUsageTracker, bind_tracker

    tracker = LlmUsageTracker()
    with bind_tracker(tracker):
        client = _client()
        tokens = [t async for t in client.converse_stream("q")]
        assert tokens == ["Hel", "lo"]
        assert captured["json"]["stream"] is True
        assert captured["json"]["stream_options"] == {"include_usage": True}
    snapshot = tracker.snapshot()
    assert snapshot.complete is True
    assert snapshot.input_tokens == 3
    assert snapshot.output_tokens == 2


@pytest.mark.asyncio
async def test_converse_stream_never_raises_guardrail_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}',
        b"data: [DONE]",
    ]
    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeStreamResponse(lines),
    )
    tokens = [t async for t in _client().converse_stream("q", guardrail_id="gr-1")]
    assert tokens == ["a"]


def test_model_id_prefers_env_when_kwarg_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_CHAT_MODEL", "anthropic/claude-sonnet-5")
    client = OpenAICompatibleLLMClient()
    assert client.model_id == "anthropic/claude-sonnet-5"


@pytest.mark.asyncio
async def test_embed_uses_shared_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse({"data": [{"embedding": [0.1, 0.2]}]}))

    def _post(url, headers=None, json=None, timeout=None, stream=None):
        captured["json"] = json
        return fake(url, headers=headers, json=json, timeout=timeout, stream=stream)

    monkeypatch.setattr("coa_serve.clients.openai.requests.post", _post)
    client = _client()
    assert await client.embed("hello") == [0.1, 0.2]
    assert captured["json"]["input"] == "hello"


@pytest.mark.asyncio
async def test_health_check_reports_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_serve.clients.openai.requests.post",
        lambda *a, **k: _FakeResponse({"data": [{"embedding": [0.0]}]}),
    )
    status = await _client().health_check()
    assert status["status"] == "ok"
