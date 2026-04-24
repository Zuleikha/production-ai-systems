"""
Async OpenAI embedder using text-embedding-3-small.

Upgrade from text-embedding-ada-002:
- Higher retrieval quality across MTEB benchmarks
- Supports a `dimensions` parameter to reduce vector size (trades quality for
  speed/cost at serving time — left at full 1536 here)
- Cheaper per token
"""

from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import settings


class OpenAIEmbedder:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed_query(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        return resp.data[0].embedding

    async def embed_batch(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """Embed a list of texts, respecting OpenAI's per-request batch limit."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            # The API returns items in the same order as the input, but the spec
            # says to sort by index to be safe.
            ordered = sorted(resp.data, key=lambda x: x.index)
            all_embeddings.extend(item.embedding for item in ordered)
        return all_embeddings
