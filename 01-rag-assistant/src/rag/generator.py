"""
Grounded answer generation using gpt-4o-mini.

Upgrade from gpt-3.5-turbo:
- Better instruction following: the citation-format prompt is obeyed more reliably
- Higher knowledge cutoff
- Comparable cost as of late 2024 and into 2025

Prompt design:
- System prompt enforces grounding rules upfront
- Context is numbered so inline citations ([1], [2]) can reference specific chunks
- Conversation history is truncated to the last 3 turns to bound prompt size
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.rag.retriever import RetrievedDocument
from config.settings import settings


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]
    tokens_used: int
    latency_ms: float
    model: str


_SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Answer the user's question using ONLY the provided context blocks.

Rules:
1. Ground every claim in the context. If the context does not contain enough information to answer, say so clearly — do not guess.
2. Cite sources inline using [1], [2], etc., where the number corresponds to the numbered context block.
3. Be concise. Prefer short, clear sentences.
4. Never fabricate information that is not present in the context.
"""


class RAGGenerator:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    async def generate(
        self,
        query: str,
        context_docs: list[RetrievedDocument],
        conversation_history: list[dict] | None = None,
    ) -> GenerationResult:
        context_text = self._format_context(context_docs)
        messages = self._build_messages(query, context_text, conversation_history)

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=response.choices[0].message.content or "",
            sources=[
                {
                    "content": d.content[:300],
                    "metadata": d.metadata,
                    "score": round(d.score, 4),
                }
                for d in context_docs
            ],
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model=self._model,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_context(self, docs: list[RetrievedDocument]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{doc.content}")
        return "\n\n".join(parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[dict] | None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Last 3 turns (6 messages) keeps context focused without ballooning tokens
        if history:
            messages.extend(history[-6:])

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })
        return messages
