# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the provider-agnostic LLM seam (Phase 0/1 of the Kilo OpenAI shim plan).

Covers:
- env-driven provider factory (default = Bedrock, no behavior change)
- protocol conformance of both providers
- OpenAI-compatible chat client semantics (parity with BedrockClient.invoke)
- OpenAI-compatible embedder semantics (parity with BedrockEmbedder)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from coa_common.bedrock import (
    BedrockClient,
    BedrockInvocationResult,
    BedrockTruncationError,
    InputTooLargeError,
    PromptSegment,
)
from coa_common.embeddings import BedrockEmbedder
from coa_common.llm import (
    Embedder,
    LLMClient,
    OpenAICompatibleClient,
    OpenAICompatibleEmbedder,
    get_embedder,
    get_llama_index_embedding,
    get_llama_index_llm,
    get_llm_client,
    make_openai_llama_index_embedding,
)

# --- Factory ---------------------------------------------------------------


def test_factory_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COA_LLM_PROVIDER", raising=False)
    assert isinstance(get_llm_client(), OpenAICompatibleClient)


def test_factory_defaults_embedder_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COA_LLM_PROVIDER", raising=False)
    assert isinstance(get_embedder(), OpenAICompatibleEmbedder)


def test_factory_selects_bedrock_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_LLM_PROVIDER", "bedrock")
    assert isinstance(get_llm_client(), BedrockClient)


def test_factory_selects_openai_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("COA_LLM_BASE_URL", "http://localhost:9000/v1")
    client = get_llm_client()
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "http://localhost:9000/v1"


def test_factory_embedder_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_LLM_PROVIDER", "openai")
    assert isinstance(get_embedder(), OpenAICompatibleEmbedder)


def test_bedrock_client_conforms_to_protocol() -> None:
    assert isinstance(BedrockClient(), LLMClient)
    assert isinstance(BedrockEmbedder(), Embedder)


def test_openai_clients_conform_to_protocol() -> None:
    assert isinstance(OpenAICompatibleClient(), LLMClient)
    assert isinstance(OpenAICompatibleEmbedder(), Embedder)


# --- OpenAI chat client ----------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def _client(**kwargs) -> OpenAICompatibleClient:
    kwargs.setdefault("base_url", "http://gateway.test/v1")
    kwargs.setdefault("api_key", "test-key")
    return OpenAICompatibleClient(**kwargs)


def _completion_payload(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def test_openai_chat_posts_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_completion_payload('{"ok": true}')))

    def _post(url: str, headers: dict | None = None, json: dict | None = None, timeout: float | None = None) -> object:
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return fake(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr("coa_common.llm.openai.requests.post", _post)
    client = _client(model="anthropic/claude-haiku-4.5")
    result = client.invoke("You are a helper", "Answer this", max_tokens=2048)

    assert captured["url"] == "http://gateway.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    body = captured["json"]
    assert body["model"] == "anthropic/claude-haiku-4.5"
    assert body["max_tokens"] == 2048
    assert body["messages"] == [
        {"role": "system", "content": "You are a helper"},
        {"role": "user", "content": "Answer this"},
    ]
    assert result.result == {"ok": True}
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert isinstance(result, BedrockInvocationResult)
    assert result.latency_ms >= 0


def test_openai_chat_prompt_segments_are_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_completion_payload("[]")))

    def _post(url: str, headers: dict | None = None, json: dict | None = None, timeout: float | None = None) -> object:
        captured["json"] = json
        return fake(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr("coa_common.llm.openai.requests.post", _post)
    client = _client()
    client.invoke("sys", [PromptSegment("Table: ", guarded=False), PromptSegment("orders")])
    assert captured["json"]["messages"][1]["content"] == "Table: orders"


def test_openai_chat_truncation_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _completion_payload("partial", finish_reason="length")
    payload["usage"] = {"prompt_tokens": 5, "completion_tokens": 2048}
    monkeypatch.setattr(
        "coa_common.llm.openai.requests.post",
        lambda *a, **k: _FakeResponse(payload),
    )
    with pytest.raises(BedrockTruncationError) as raised:
        _client().invoke("sys", "u", max_tokens=2048)
    assert raised.value.output_tokens == 2048
    assert raised.value.max_tokens == 2048
    assert raised.value.stop_reason == "max_tokens"


def test_openai_chat_strips_code_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_common.llm.openai.requests.post",
        lambda *a, **k: _FakeResponse(_completion_payload('```json\n{"a": 1}\n```')),
    )
    assert _client().invoke("sys", "u").result == {"a": 1}


def test_openai_chat_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_common.llm.openai.requests.post",
        lambda *a, **k: _FakeResponse(_completion_payload("not json")),
    )
    with pytest.raises(ValueError, match="Invalid .* response format"):
        _client().invoke("sys", "u")


def test_openai_chat_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "coa_common.llm.openai.requests.post",
        lambda *a, **k: _FakeResponse(_completion_payload("")),
    )
    with pytest.raises(ValueError, match="empty"):
        _client().invoke("sys", "u")


def test_openai_chat_input_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InputTooLargeError):
        _client(max_input_chars=10).invoke("sys", "u" * 50)


def test_openai_chat_accepts_bedrock_kwargs() -> None:
    client = OpenAICompatibleClient(
        region="us-east-1",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        read_timeout=120,
        guardrail_id="gr-123",
        guardrail_version="DRAFT",
        component="enrichment",
        base_url="http://gateway.test/v1",
    )
    assert client.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_openai_chat_env_model_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_CHAT_MODEL", "anthropic/claude-sonnet-5")
    client = _client(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert client.model == "anthropic/claude-sonnet-5"


# --- OpenAI embedder -------------------------------------------------------


def _embedding_payload(vector: list[float], index: int = 0) -> dict:
    return {"object": "list", "data": [{"object": "embedding", "index": index, "embedding": vector}]}


def test_embed_query_posts_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_embedding_payload([0.1, 0.2])))

    def _post(url: str, headers: dict | None = None, json: dict | None = None, timeout: float | None = None) -> object:
        captured.update(url=url, headers=headers, json=json)
        return fake(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr("coa_common.llm.openai.requests.post", _post)
    emb = OpenAICompatibleEmbedder(base_url="http://gateway.test/v1", api_key="k", model="embed-model")
    assert emb.embed_query("hello") == [0.1, 0.2]
    assert captured["url"] == "http://gateway.test/v1/embeddings"
    assert captured["json"] == {"model": "embed-model", "input": "hello"}


def test_embed_documents_batches_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    vectors_by_text = {"a": [1.0], "b": [2.0], "c": [3.0]}

    def _post(url: str, headers: dict | None = None, json: dict | None = None, timeout: float | None = None) -> object:
        calls.append(json or {})
        inputs: list[str] = (json or {}).get("input", [])
        payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vectors_by_text[t]} for i, t in enumerate(inputs)
            ],
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("coa_common.llm.openai.requests.post", _post)
    emb = OpenAICompatibleEmbedder(base_url="http://gateway.test/v1", api_key="k", model="m", batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert out == [[1.0], [2.0], [3.0]]
    assert [c["input"] for c in calls] == [["a", "b"], "c"]


def test_embed_records_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    from coa_common.bedrock_metrics import CostTracker

    monkeypatch.setattr(
        "coa_common.llm.openai.requests.post",
        lambda *a, **k: _FakeResponse(_embedding_payload([0.5])),
    )
    tracker = CostTracker()
    emb = OpenAICompatibleEmbedder(base_url="http://gateway.test/v1", api_key="k", model="m", cost_tracker=tracker)
    emb.embed_query("hello")
    assert tracker.input_tokens_by_stage["embed"] > 0


def test_embed_accepts_bedrock_kwargs() -> None:
    emb = OpenAICompatibleEmbedder(
        model_id="us.cohere.embed-v4:0",
        dimensions=1024,
        region="us-east-1",
        max_workers=4,
        read_timeout=60,
        batch_size=10,
        base_url="http://gateway.test/v1",
    )
    assert emb.model == "us.cohere.embed-v4:0"
    assert emb.dimensions == 1024


def test_embed_env_model_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_EMBED_MODEL", "nomic-embed-text")
    emb = OpenAICompatibleEmbedder(model_id="us.cohere.embed-v4:0", base_url="http://gateway.test/v1")
    assert emb.model == "nomic-embed-text"


# --- llama_index embedding adapter (GraphRAG / kg-build) -------------------


def test_get_llama_index_embedding_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from llama_index.core.base.embeddings.base import BaseEmbedding

    monkeypatch.delenv("COA_LLM_PROVIDER", raising=False)
    adapter = get_llama_index_embedding(base_url="http://gateway.test/v1", api_key="k", model="m")
    assert isinstance(adapter, BaseEmbedding)
    assert adapter.__class__.__name__ == "OpenAIEmbedderLlamaIndex"


def test_get_llama_index_embedding_bedrock_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_LLM_PROVIDER", "bedrock")
    adapter = get_llama_index_embedding(model_id="us.cohere.embed-v4:0", dimensions=1024)
    assert adapter.__class__.__name__ == "BedrockEmbedderLlamaIndex"


def test_openai_llama_index_adapter_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    fake = MagicMock(return_value=_FakeResponse(_embedding_payload([0.25, 0.5])))

    def _post(url: str, headers: dict | None = None, json: dict | None = None, timeout: float | None = None) -> object:
        captured["json"] = json
        return fake(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr("coa_common.llm.openai.requests.post", _post)
    adapter = make_openai_llama_index_embedding(base_url="http://gateway.test/v1", api_key="k", model="m")
    assert adapter._get_text_embedding("hello") == [0.25, 0.5]
    assert captured["json"]["model"] == "m"
    assert adapter._get_query_embedding("q") == [0.25, 0.5]


# --- llama_index LLM (GraphRAG extraction/response) ------------------------


def test_get_llama_index_llm_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from llama_index.llms.openai import OpenAI

    monkeypatch.delenv("COA_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("COA_LLM_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("COA_LLM_API_KEY", "k")
    llm = get_llama_index_llm(model="anthropic/claude-haiku-4.5")
    assert isinstance(llm, OpenAI)
    assert llm.model == "anthropic/claude-haiku-4.5"
    assert llm.api_key == "k"
    assert llm.api_base == "http://gateway.test/v1"


def test_get_llama_index_llm_none_for_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COA_LLM_PROVIDER", "bedrock")
    assert get_llama_index_llm() is None
