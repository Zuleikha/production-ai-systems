"""
Cross-encoder reranker.

Why rerank after retrieval?
Bi-encoder (embedding) models encode query and document independently, which
means they can't model the interaction between them. Cross-encoders see both
jointly and produce much more accurate relevance scores — but they're too slow
to run across an entire corpus (O(n) forward passes). The standard solution is
two-stage: retrieve cheaply with bi-encoders, rerank expensively with a
cross-encoder on the small candidate set.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking (700 k query-passage pairs)
- ~84 MB, CPU-friendly, ~50 ms per rerank call on a laptop CPU
- For higher quality at the cost of size: cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from src.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    logger.info("Loading cross-encoder model: %s", _MODEL_NAME)
    _cross_encoder: CrossEncoder | None = CrossEncoder(_MODEL_NAME)
    logger.info("Reranker ready")
except Exception:
    logger.warning(
        "Failed to load cross-encoder model %s; reranker will fall back to embedding scores",
        _MODEL_NAME,
    )
    _cross_encoder = None


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = _cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int = 3,
    ) -> list[RetrievedDocument]:
        if not candidates:
            return []

        if self._model is None:
            return sorted(candidates, key=lambda d: d.score, reverse=True)[:top_k]

        pairs = [(query, doc.content) for doc in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        return [
            RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),
                retrieval_method=doc.retrieval_method,
            )
            for doc, score in ranked[:top_k]
        ]
