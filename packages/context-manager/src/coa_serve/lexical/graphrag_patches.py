# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve-side graphrag-toolkit patch: request ``embedding`` in ``_source``.

Why this exists
---------------
``TopicBeamSearch`` expands the beam by fetching each neighbour topic's *stored*
embedding by id (``topic_beam_search.py:148`` -> ``OpenSearchIndex.get_embeddings``
-> ``get_all_embeddings`` -> ``paginated_search``). The toolkit's
``_to_get_embedding_result`` then reads ``source['embedding']``
(``opensearch_vector_indexes.py:643``) with a bare subscript.

On AOSS **NEXTGEN** collections the ``knn_vector`` field is NOT returned in the
default ``_source`` projection, so that subscript raises ``KeyError: 'embedding'``.
The exception escapes ``get_embeddings`` and is swallowed one level up by
``TopicBeamSearch._get_embeddings``' broad ``except Exception`` (logged only as
``Failed to fetch stored topic embeddings``). Every neighbour is then treated as
"no stored embedding" and skipped, so the beam collapses to its seed set and the
retriever returns ~1 node regardless of how good the seeds are.

Measured on the live sec10q collection: a terms query for known topic ids matches
3/3 docs but ``'embedding' in hit['_source']`` is ``False``; adding an explicit
``_source`` include returns the full 1024-dim vector for 5681/5681 topic docs.

The fix: explicitly project the fields ``_to_get_embedding_result`` consumes. This
is a read-path-only patch (serve never writes vectors), and it leaves the
``ids_only`` branch untouched since that path deliberately suppresses ``_source``.

Reported upstream; remove this module once the toolkit projects the vector field
itself. Applied lazily (never at import) to preserve the botocore sync-only
invariant documented in ``strategies.py``.
"""

from __future__ import annotations

import time

import structlog

logger = structlog.get_logger(__name__)

# Fields ``_to_get_embedding_result`` reads: source['id'], source['value'],
# source['embedding'], source['metadata']['_node_content'].
_EMBEDDING_SOURCE_INCLUDES = ["id", "value", "embedding", "metadata"]

_patched = False


def patch_paginated_search_source() -> None:
    """Make ``paginated_search`` project ``embedding`` into ``_source``.

    Idempotent and failure-tolerant: if the toolkit's internals have moved, this
    logs and returns rather than breaking retrieval (the caller degrades to the
    pre-patch behavior, i.e. a narrow beam, not an error).
    """
    global _patched
    if _patched:
        return
    try:
        import graphrag_toolkit.lexical_graph.storage.vector.opensearch_vector_indexes as ovi
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        logger.debug("graphrag_embedding_source_patch_skipped", reason="module not importable")
        return

    original = getattr(ovi.OpenSearchIndex, "paginated_search", None)
    if original is None:
        logger.warning("graphrag_embedding_source_patch_skipped", reason="paginated_search missing")
        return

    def paginated_search(self, query, page_size=10000, max_pages=None, ids_only=False):
        # ids_only deliberately sets "_source": False upstream; only widen the
        # projection for the embedding-fetch path.
        if ids_only:
            yield from original(self, query, page_size=page_size, max_pages=max_pages, ids_only=True)
            return

        client = self.client._os_client
        if client is None:
            return

        search_after = None
        page = 0
        while True:
            body = {
                "size": page_size,
                "query": query,
                "sort": [{"_id": "asc"}],
                # The one line this patch exists for: AOSS NEXTGEN omits
                # knn_vector from the default projection.
                "_source": {"includes": _EMBEDDING_SOURCE_INCLUDES},
            }
            if search_after:
                body["search_after"] = search_after

            # Preserve upstream's index-not-found retry (a namespace can be queried
            # before its topic index materializes); everything else propagates.
            not_found_retries = 0
            response = None
            while response is None:
                try:
                    response = client.search(index=self.underlying_index_name(), body=body)
                except NotFoundError:
                    not_found_retries += 1
                    if not_found_retries > 3:
                        raise
                    time.sleep(5)

            hits = response["hits"]["hits"]
            if not hits:
                break

            yield hits

            search_after = hits[-1]["sort"]
            page += 1
            if max_pages and page >= max_pages:
                break

    ovi.OpenSearchIndex.paginated_search = paginated_search
    _patched = True
    logger.info("graphrag_embedding_source_patch_applied", includes=_EMBEDDING_SOURCE_INCLUDES)


_llm_usage_patched = False
# Set on each wrapper so a re-application never wraps a wrapper (which would record
# every call twice) and so tests can restore the true original.
_ORIGINAL_ATTR = "__coa_llm_usage_original__"
# BedrockConverse entry points that bypass ``chat`` and whose usage this patch does
# not read. GraphRAG's keyword provider uses none of them (LLMCache.predict ->
# LLM.predict -> chat); a call through one is reported, never silently uncounted.
_UNACCOUNTED_LLM_METHODS = ("stream_chat", "achat", "astream_chat", "astream_complete", "acomplete")


def _usage_from_raw(raw: object) -> dict | None:
    """Extract a ``record_usage``-compatible usage dict from a ChatResponse.raw.

    Bedrock Converse returns ``raw`` as the Converse response dict carrying a
    ``usage`` block (``inputTokens``/``outputTokens``). The OpenAI-compatible LLM
    returns ``raw`` as the OpenAI ``ChatCompletion`` object, whose ``usage`` is a
    pydantic model (``prompt_tokens``/``completion_tokens``). Normalize both.
    """
    if isinstance(raw, dict):
        usage = raw.get("usage")
        return usage if isinstance(usage, dict) else None
    usage = getattr(raw, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return vars(usage) if hasattr(usage, "__dict__") else None


def patch_llm_usage_accounting() -> None:
    """Record GraphRAG's LlamaIndex LLM calls in the request's LLM usage tracker.

    GraphRAG's keyword provider calls its LLM through ``LLMCache.predict``, which
    returns only text. The usage lives one level down: ``chat`` returns
    ``ChatResponse(raw=<provider response>)`` carrying the provider's usage block
    (Bedrock Converse ``inputTokens``/``outputTokens``, or OpenAI-compatible
    ``prompt_tokens``/``completion_tokens``). Wrapping ``chat`` sits BELOW the
    cache, so a cache hit makes no provider call and correctly costs nothing.

    The patch covers whichever LlamaIndex LLM the provider is using — the legacy
    ``BedrockConverse`` and/or ``OpenAI`` — so a call is never unpatched.

    A provider call is attributed to the tracker bound on the calling thread (serve
    binds it in the lexical worker; see ``llm_usage.graphrag_retrieval``). A call
    seen with neither a bound tracker nor a deliberate-untracked marker while
    tracked GraphRAG retrieval is in flight marks those requests incomplete
    instead of being lost.

    Idempotent and failure-tolerant like :func:`patch_paginated_search_source`: if
    LlamaIndex has moved, this logs and returns, and :func:`llm_usage_patch_active`
    stays False so the lexical retriever reports tracked requests incomplete rather
    than letting GraphRAG's LLM spend go uncounted.
    """
    global _llm_usage_patched
    if _llm_usage_patched:
        return

    from ..llm_usage import (
        REASON_PROVIDER_ERROR,
        REASON_UNSUPPORTED_CALL,
        current_tracker,
        is_untracked_graphrag_retrieval,
        note_unattributed_graphrag_call,
        record_usage,
    )

    classes: list[type] = []
    try:
        from llama_index.llms.bedrock_converse import BedrockConverse

        classes.append(BedrockConverse)
    except ImportError:
        pass
    try:
        from llama_index.llms.openai import OpenAI

        classes.append(OpenAI)
    except ImportError:
        pass
    classes = [cls for cls in classes if getattr(cls, "chat", None) is not None]
    if not classes:
        logger.warning("graphrag_llm_usage_patch_skipped", reason="no patcheable LLM class")
        return

    def _original(cls: type, name: str):
        method = getattr(cls, name, None)
        return getattr(method, _ORIGINAL_ATTR, method)

    def _bound(original, instance):
        # LlamaIndex wraps these methods in its instrumentation dispatcher, a
        # descriptor that only binds correctly through ``__get__`` — calling the
        # class attribute with an explicit ``self`` drops it.
        return original.__get__(instance, type(instance))

    def chat(original_chat):
        def wrapper(self, messages, **kwargs):
            tracker = current_tracker()
            if tracker is None:
                if not is_untracked_graphrag_retrieval():
                    note_unattributed_graphrag_call()
                return _bound(original_chat, self)(messages, **kwargs)
            handle = tracker.register_call("graphrag_llm", str(getattr(self, "model", "")))
            try:
                response = _bound(original_chat, self)(messages, **kwargs)
            except BaseException:
                tracker.fail_call(handle, REASON_PROVIDER_ERROR)
                raise
            record_usage(tracker, handle, _usage_from_raw(getattr(response, "raw", None)))
            return response

        wrapper.__name__ = "chat"
        setattr(wrapper, _ORIGINAL_ATTR, original_chat)
        return wrapper

    def _unaccounted(name: str, original):
        def method(self, *args, **kwargs):
            tracker = current_tracker()
            if tracker is None:
                if not is_untracked_graphrag_retrieval():
                    note_unattributed_graphrag_call()
            else:
                tracker.mark_incomplete(REASON_UNSUPPORTED_CALL)
            return _bound(original, self)(*args, **kwargs)

        method.__name__ = name
        setattr(method, _ORIGINAL_ATTR, original)
        return method

    for cls in classes:
        original_chat = _original(cls, "chat")
        if original_chat is None:
            continue
        cls.chat = chat(original_chat)
        for name in _UNACCOUNTED_LLM_METHODS:
            original = _original(cls, name)
            if original is not None:
                setattr(cls, name, _unaccounted(name, original))

    _llm_usage_patched = True
    logger.info("graphrag_llm_usage_patch_applied", classes=[cls.__name__ for cls in classes])


_llm_provider_patched = False


def patch_graphrag_llm_provider() -> None:
    """Route GraphRAG's internal LLM construction to the configured provider.

    graphrag-toolkit's ``GraphRAGConfig.to_llm`` builds a Bedrock Converse client
    from a model string and only returns an ``LLM`` INSTANCE unchanged. In
    ``openai`` mode (the default — Kilo AI Gateway, no AWS in the LLM path) we
    install an OpenAI LLM instance on ``extraction_llm``/``response_llm`` and
    patch ``to_llm`` so ANY internal construction point — including components
    that bypass the config properties — receives the OpenAI instance instead of
    a Bedrock client. ``bedrock`` mode is left untouched (legacy default).

    Idempotent and failure-tolerant: if the toolkit has moved, this logs and
    returns, leaving the legacy default in place.
    """
    global _llm_provider_patched
    if _llm_provider_patched:
        return
    try:
        from coa_common.llm import get_llama_index_llm
        from graphrag_toolkit.lexical_graph.config import GraphRAGConfig
    except ImportError:
        logger.debug("graphrag_llm_provider_patch_skipped", reason="module not importable")
        return
    llm = get_llama_index_llm()
    if llm is None:
        return  # bedrock mode — keep the toolkit's legacy default

    try:
        from llama_index.core.llms import LLM as LlamaIndexLLM

        GraphRAGConfig.extraction_llm = llm
        GraphRAGConfig.response_llm = llm
        original_to_llm = GraphRAGConfig.to_llm

        def to_llm(self, value):
            if isinstance(value, LlamaIndexLLM):
                return value
            return llm

        GraphRAGConfig.to_llm = to_llm
        GraphRAGConfig.__coa_original_to_llm__ = original_to_llm
    except Exception as exc:  # noqa: BLE001 - degrade to legacy default, never break retrieval
        logger.warning("graphrag_llm_provider_patch_failed", error=type(exc).__name__)
        return
    _llm_provider_patched = True
    logger.info("graphrag_llm_provider_patched", provider="openai", model=str(getattr(llm, "model", "")))


def graphrag_llm_provider_patched() -> bool:
    """Whether GraphRAG's LLM is routed through the configured provider (see :func:`patch_graphrag_llm_provider`)."""
    return _llm_provider_patched


def llm_usage_patch_active() -> bool:
    """Whether GraphRAG's LLM calls are being recorded (see :func:`patch_llm_usage_accounting`)."""
    return _llm_usage_patched
