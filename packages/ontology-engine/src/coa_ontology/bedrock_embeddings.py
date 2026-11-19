# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-embeddings client for ontology induction.

Thin adapter over the shared provider-aware embedder (:func:`coa_common.llm.get_embedder`
— Kilo OpenAI-compatible by default, Bedrock opt-in) that preserves the
``EmbeddingGeneratorClient.generate_lexical()`` interface the inducer expects.

Class labels are indexable content, so they are embedded as *documents*; the
grounding recall step compares class embeddings against other class embeddings,
so both sides use the same (document) input type and stay comparable.
"""

from __future__ import annotations

import re

from coa_common import DEFAULT_EMBED_DIMENSIONS, DEFAULT_EMBED_MODEL_ID
from coa_common.bedrock_metrics import CostTracker
from coa_common.llm import Embedder, get_embedder

_MAX_WORKERS = 10


def _split_camel(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)


class BedrockEmbeddingClient:
    """Drop-in replacement for EmbeddingGeneratorClient over the shared embedder."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = DEFAULT_EMBED_MODEL_ID,
        dimensions: int = DEFAULT_EMBED_DIMENSIONS,
    ):
        """Construct the client and its underlying embedder.

        Args:
            region: Accepted for Bedrock compatibility; unused by the OpenAI provider.
            model_id: Embedding model identifier (Bedrock id, or gateway id when
                ``COA_LLM_PROVIDER=openai`` and ``COA_EMBED_MODEL`` is unset).
            dimensions: Output embedding dimensionality.
        """
        self.model_id = model_id
        self.dimensions = dimensions
        self._embedder: Embedder = get_embedder(
            model_id=model_id, dimensions=dimensions, region=region, max_workers=_MAX_WORKERS
        )

    def set_cost_tracker(self, cost_tracker: CostTracker | None) -> None:
        """Route embed-call token usage to a per-job tracker (delegates to the embedder)."""
        self._embedder.set_cost_tracker(cost_tracker)

    def generate_lexical(
        self,
        ontology_id: str,
        classes: list[dict],
        backend: str | None = None,
        store: bool = True,
    ) -> tuple[list[dict], str]:
        """Same signature as EmbeddingGeneratorClient.generate_lexical().

        Returns (results, model_id) where each result has a 'vector' key.
        """
        texts = [
            " ".join(_split_camel(lbl) for lbl in c.get("labels", [])) + " " + " ".join(c.get("comments", []))
            for c in classes
        ]
        # Per-channel dedup: identical channel texts (e.g. the dtype
        # channel, where thousands of columns share "unknown"/"INT") are embedded
        # once, then the vector is scattered back to every original position. The
        # dedup buffer is LOCAL to this call — never a persistent, text-keyed cache
        # (that would collide search_document vs search_query; see HLD R1). Scatter
        # returns exactly len(classes) vectors in input order so the fusion loop's
        # positional fold (pipeline.py) stays aligned (HLD R2). Aliasing duplicate
        # positions to one vector object is safe: the fusion loop only reads them
        # and writes a fresh fused list (HLD R3).
        uniq = list(dict.fromkeys(texts))
        uniq_vecs = self._embedder.embed_documents(uniq)
        by_text = dict(zip(uniq, uniq_vecs, strict=True))
        vectors = [by_text[t] for t in texts]
        results = [
            {
                "entity_uri": classes[i].get("class_uri", ""),
                "entity_type": "class",
                "vector": vectors[i],
                "dimensions": len(vectors[i]),
            }
            for i in range(len(classes))
        ]
        return results, self.model_id

    def embed_text(self, text: str) -> list[float]:
        """Convenience: embed a single text string (as a document)."""
        return self._embedder.embed_document(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convenience: embed multiple texts in parallel (as documents)."""
        return self._embedder.embed_documents(texts)
