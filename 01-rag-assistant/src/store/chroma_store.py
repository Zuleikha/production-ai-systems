"""
ChromaDB-backed persistent vector store.

Replaces the custom JSON file store from v1. Key gains:
- HNSW approximate nearest-neighbour index — O(log n) queries instead of O(n)
- Metadata filtering (e.g. query only PDF chunks)
- Real persistence without manual JSON serialisation
- upsert semantics prevent duplicate ingestion of the same document
"""

from __future__ import annotations

import chromadb
from chromadb import Settings as ChromaSettings

from config.settings import settings


class ChromaVectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            # cosine distance so similarity = 1 - distance
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self.count)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._col.query(**kwargs)
        docs = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            docs.append({"content": doc, "metadata": meta, "score": 1.0 - dist})
        return docs

    def get_all_documents(self) -> list[dict]:
        """Return every stored document (used to build the BM25 index).

        NOTE: for very large collections (>100 k chunks) this becomes slow.
        A production system would maintain a separate BM25 index and update it
        incrementally on each upsert.
        """
        result = self._col.get(include=["documents", "metadatas"])
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    # ── Admin ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        """Drop and recreate the collection (useful in tests)."""
        self._client.delete_collection(settings.chroma_collection)
        self._col = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
