"""
Sentence-aware, token-counted document chunker.

Key improvements over the original character-based splitter:
- Never cuts mid-sentence: preserves the semantic units LLMs are trained on
- Token-based sizing: aligns with LLM context windows, not arbitrary byte counts
- Overlap is sentence-granular: adjacent chunks share complete sentences, not
  half-sentences, so the reranker can evaluate each chunk in isolation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken


@dataclass
class TextChunk:
    content: str
    metadata: dict = field(default_factory=dict)


class SentenceAwareChunker:
    # Matches end-of-sentence punctuation followed by whitespace + uppercase letter.
    # Handles common abbreviations and decimal numbers imperfectly — acceptable
    # for a first-pass chunker; a full sentence boundary detector would be heavier.
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # cl100k_base covers all current OpenAI embedding and chat models
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        if metadata is None:
            metadata = {}

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        current: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            st = self._token_count(sentence)

            if st > self.chunk_size:
                # Flush current buffer before handling the oversized sentence
                if current:
                    chunks.append(self._make_chunk(current, metadata, chunk_index))
                    chunk_index += 1
                    current, current_tokens = [], 0

                for hard_chunk in self._hard_split(sentence, metadata, chunk_index):
                    chunks.append(hard_chunk)
                    chunk_index += 1
                continue

            if current_tokens + st > self.chunk_size and current:
                chunks.append(self._make_chunk(current, metadata, chunk_index))
                chunk_index += 1
                overlap = self._overlap_tail(current)
                current = overlap
                current_tokens = sum(self._token_count(s) for s in current)

            current.append(sentence)
            current_tokens += st

        if current:
            chunks.append(self._make_chunk(current, metadata, chunk_index))

        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._SENTENCE_SPLIT.split(text) if s.strip()]

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        """Return trailing sentences that fit within chunk_overlap tokens."""
        tail: list[str] = []
        tokens = 0
        for s in reversed(sentences):
            t = self._token_count(s)
            if tokens + t > self.chunk_overlap:
                break
            tail.insert(0, s)
            tokens += t
        return tail

    def _make_chunk(self, sentences: list[str], metadata: dict, index: int) -> TextChunk:
        return TextChunk(
            content=" ".join(sentences),
            metadata={**metadata, "chunk_index": index},
        )

    def _hard_split(self, text: str, metadata: dict, start_index: int) -> list[TextChunk]:
        """Token-window fallback for single sentences exceeding chunk_size."""
        tokens = self._enc.encode(text, disallowed_special=())
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for i in range(0, len(tokens), step):
            window = tokens[i : i + self.chunk_size]
            chunks.append(TextChunk(
                content=self._enc.decode(window),
                metadata={**metadata, "chunk_index": start_index + len(chunks)},
            ))
        return chunks
