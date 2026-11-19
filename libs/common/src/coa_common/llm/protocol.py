# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-agnostic LLM protocols (Phase 0 of the Kilo OpenAI shim).

The shared value types and exceptions stay defined in ``coa_common.bedrock`` —
their canonical home, and where every existing consumer imports them from — and
are re-exported here so provider implementations can depend on the protocol
module without creating an import cycle. Renaming the Bedrock-prefixed names is
out of scope for the shim; preserving them keeps the swap invisible to callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from coa_common.bedrock import (
    BedrockInvocationResult,
    BedrockTruncationError,
    GuardrailBlockedError,
    InputTooLargeError,
    PromptSegment,
    prompt_text,
)

__all__ = [
    "BedrockInvocationResult",
    "BedrockTruncationError",
    "Embedder",
    "GuardrailBlockedError",
    "InputTooLargeError",
    "LLMClient",
    "PromptSegment",
    "prompt_text",
]


@runtime_checkable
class LLMClient(Protocol):
    """Chat/completion client surface shared by all providers.

    Matches ``BedrockClient.invoke`` exactly so the two are interchangeable.
    """

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str | Sequence[PromptSegment],
        max_tokens: int = 4096,
    ) -> BedrockInvocationResult:
        """Invoke the model and return a parsed result with usage and latency."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Embedding client surface shared by all providers.

    Matches ``BedrockEmbedder`` exactly (including the optional per-job cost
    tracker) so the two are interchangeable.
    """

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...

    def embed_document(self, text: str) -> list[float]:
        """Embed indexable content."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many documents, order preserved, one vector per input."""
        ...

    def set_cost_tracker(self, cost_tracker: Any | None) -> None:
        """Attach (or clear) a per-job cost tracker after construction."""
        ...
