# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-agnostic LLM/embedding seam (Kilo OpenAI shim).

``coa_common.bedrock`` and ``coa_common.embeddings`` remain the Bedrock
implementations and keep their public surface unchanged; this package adds the
protocol those implementations conform to, an OpenAI-compatible implementation
(Kilo AI Gateway — the default provider), and an env-driven factory so callers
can switch providers without changing call sites.

The default provider is ``openai`` (Kilo OpenAI-compatible, no AWS in the LLM
path); ``COA_LLM_PROVIDER=bedrock`` opts back in to the legacy Bedrock
implementation.
"""

from __future__ import annotations

import os
from typing import Any

from coa_common.bedrock import BedrockClient
from coa_common.embeddings import BedrockEmbedder, make_llama_index_embedding
from coa_common.llm.openai import (
    OpenAICompatibleClient,
    OpenAICompatibleEmbedder,
    make_openai_llama_index_embedding,
    make_openai_llama_index_llm,
)
from coa_common.llm.protocol import Embedder, LLMClient

__all__ = [
    "Embedder",
    "LLMClient",
    "OpenAICompatibleClient",
    "OpenAICompatibleEmbedder",
    "get_embedder",
    "get_llama_index_embedding",
    "get_llama_index_llm",
    "get_llm_client",
    "make_openai_llama_index_embedding",
    "make_openai_llama_index_llm",
]


def _provider() -> str:
    return os.environ.get("COA_LLM_PROVIDER", "openai").strip().lower()


def get_llm_client(**kwargs: Any) -> LLMClient:
    """Build a chat/completion client for the configured provider.

    ``COA_LLM_PROVIDER`` selects the implementation: ``openai`` (the default) is
    an :class:`OpenAICompatibleClient` pointed at the Kilo AI Gateway
    (``COA_LLM_BASE_URL``); ``bedrock`` returns the legacy
    :class:`~coa_common.bedrock.BedrockClient`. Bedrock-only kwargs (region,
    guardrail ids, component) are accepted by both implementations, so callers
    construct the client exactly as they do today.
    """
    if _provider() == "bedrock":
        return BedrockClient(**kwargs)
    return OpenAICompatibleClient(**kwargs)


def get_embedder(**kwargs: Any) -> Embedder:
    """Build an embedding client for the configured provider (see :func:`get_llm_client`)."""
    if _provider() == "bedrock":
        return BedrockEmbedder(**kwargs)
    return OpenAICompatibleEmbedder(**kwargs)


def get_llama_index_embedding(**kwargs: Any) -> Any:
    """Build a LlamaIndex ``BaseEmbedding`` for the configured provider.

    Provider-aware twin of :func:`get_embedder` for the llama_index integration
    points (GraphRAG lexical retrieval, doc-kg-build ingestion): ``openai`` (the
    default) returns an adapter over :class:`OpenAICompatibleEmbedder`, ``bedrock``
    returns the legacy ``coa_common.embeddings.make_llama_index_embedding``
    adapter. Kwargs mirror ``make_llama_index_embedding`` (model_id, dimensions,
    region) and additionally accept the OpenAI endpoint overrides (base_url,
    api_key, model).
    """
    if _provider() == "bedrock":
        return make_llama_index_embedding(**kwargs)
    openai_kwargs = dict(kwargs)
    model_id = openai_kwargs.pop("model_id", None)
    # ``region`` is Bedrock-only; the OpenAI endpoint is chosen by base_url.
    openai_kwargs.pop("region", None)
    openai_kwargs.setdefault("model", model_id)
    return make_openai_llama_index_embedding(**openai_kwargs)


def get_llama_index_llm(**kwargs: Any) -> Any | None:
    """Build a LlamaIndex LLM for the configured provider (GraphRAG paths).

    ``openai`` (the default) returns a LlamaIndex ``OpenAI`` instance pointed at
    the Kilo AI Gateway — pass it to ``GraphRAGConfig.extraction_llm`` /
    ``response_llm`` so the toolkit's ``to_llm`` returns it unchanged instead of
    constructing a Bedrock Converse client. ``bedrock`` returns ``None``, leaving
    the toolkit's legacy Bedrock default in place. Kwargs mirror the OpenAI LLM
    (model, api_key, base_url).
    """
    if _provider() == "bedrock":
        return None
    return make_openai_llama_index_llm(**kwargs)
