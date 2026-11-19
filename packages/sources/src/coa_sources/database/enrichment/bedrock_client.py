# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client — re-exported from coa_common.

The Bedrock names remain re-exported for backwards compatibility (tests and
older call sites construct them directly). New construction should go through
``get_llm_client`` so the provider is chosen by ``COA_LLM_PROVIDER`` (openai /
Kilo AI Gateway by default — no AWS in the LLM path; ``bedrock`` opts back in)
without changing call sites.
"""

from coa_common.bedrock import (
    DEFAULT_MODEL_ID,
    BedrockClient,
    BedrockInvocationResult,
    BedrockTruncationError,
    GuardrailBlockedError,
    PromptSegment,
)
from coa_common.llm import LLMClient, get_llm_client

__all__ = [
    "LLMClient",
    "BedrockClient",
    "BedrockInvocationResult",
    "BedrockTruncationError",
    "DEFAULT_MODEL_ID",
    "GuardrailBlockedError",
    "PromptSegment",
    "get_llm_client",
]
