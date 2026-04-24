# RAG Assistant — v2

Retrieval-Augmented Generation system with hybrid retrieval and cross-encoder reranking. Accepts PDF and text documents, indexes them in ChromaDB, and answers questions with inline citations.

## Tech stack

| Layer | Technology |
|---|---|
| LLM | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` (1536-dim) |
| Vector store | ChromaDB (persistent, HNSW) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Config | `pydantic-settings` |
| Logging | `structlog` |

## Architecture

```
┌──────────────┐    HTTP    ┌───────────────────────────────────────────────┐
│  Streamlit   │ ─────────► │  FastAPI                                      │
│  frontend    │            │                                               │
└──────────────┘            │  POST /ingest ──► DocumentIngestor            │
                            │                    └─ SentenceAwareChunker    │
                            │                    └─ OpenAIEmbedder          │
                            │                    └─ ChromaDB (upsert)       │
                            │                                               │
                            │  POST /query ───► RAGPipeline                 │
                            │                    └─ HybridRetriever         │
                            │                        ├─ Dense (ChromaDB)    │
                            │                        ├─ Sparse (BM25)       │
                            │                        └─ RRF fusion          │
                            │                    └─ CrossEncoderReranker    │
                            │                    └─ RAGGenerator (gpt-4o-mini) │
                            └───────────────────────────────────────────────┘
```

## What changed from v1 — and why

| Component | v1 | v2 | Why |
|---|---|---|---|
| Embedding model | `text-embedding-ada-002` | `text-embedding-3-small` | Better MTEB scores, cheaper, supports dimensionality reduction |
| LLM | `gpt-3.5-turbo` | `gpt-4o-mini` | Better instruction following for citation format, higher knowledge cutoff |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars) | `SentenceAwareChunker` (512 tokens) | Never cuts mid-sentence; token-based sizes align with LLM context windows |
| Vector store | Custom JSON file | ChromaDB (persistent) | Real HNSW ANN indexing, metadata filtering, no manual serialisation |
| Retrieval | Dense-only, top-3 | Dense + BM25 + RRF, top-10 before rerank | Sparse leg catches exact-term matches that embeddings miss |
| Reranking | None | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Bi-encoders can't model query-doc interaction; cross-encoders produce far more accurate relevance scores |
| PDF parsing | `PyPDF2` (deprecated 2023) | `pypdf` | PyPDF2 is unmaintained; pypdf is the official successor |
| Configuration | Plain class with `os.getenv` | `pydantic-settings` `BaseSettings` | Type validation, automatic env-var binding, clear failure if required vars missing |
| Monitoring | `print()` | `structlog` + `/metrics` endpoint | Machine-readable logs, latency/token tracking exposed via API |
| Testing | Import checks | `pytest` with `pytest-asyncio` and mocks | Tests orchestration logic, not just imports |
| LangChain | Used for text splitting | Removed | Was a heavy dependency for 10 lines; native implementation is clearer |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # set OPENAI_API_KEY
python -m uvicorn src.api.main:app --reload
streamlit run src/app.py      # optional frontend
```

**Docker:**

```bash
cp .env.example .env
docker compose up --build
```

API → `localhost:8000` · Frontend → `localhost:8501` · Swagger → `localhost:8000/docs`

## Usage

**Ingest** — drop files into `data/raw/` (auto-ingested on startup) or upload at runtime:

```bash
curl -X POST localhost:8000/ingest -F "file=@report.pdf"
```

**Query:**

```bash
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the key findings?", "conversation_id": "s1"}'
```

Response: `answer`, cited `sources` with rerank scores, `tokens_used`, per-stage `latency_ms`.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status, model names, documents indexed |
| `POST` | `/query` | Ask a question, get a grounded answer |
| `POST` | `/ingest` | Upload a PDF, TXT, or Markdown file |
| `GET` | `/metrics` | Cumulative query count, latency, token usage |
| `DELETE` | `/conversation/{id}` | Clear a conversation's history |
| `GET` | `/docs` | Interactive Swagger UI |

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
├── config/
│   └── settings.py          # Pydantic BaseSettings (env-validated)
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── rag/
│   │   ├── chunker.py       # Sentence-aware token-counted chunker
│   │   ├── embedder.py      # OpenAI text-embedding-3-small
│   │   ├── retriever.py     # Hybrid dense+BM25 retrieval with RRF
│   │   ├── reranker.py      # Cross-encoder reranker
│   │   ├── generator.py     # GPT-4o-mini answer generation
│   │   └── pipeline.py      # Orchestration
│   ├── store/
│   │   └── chroma_store.py  # ChromaDB wrapper
│   ├── ingestion/
│   │   └── ingest.py        # PDF + text document loader
│   ├── monitoring/
│   │   └── metrics.py       # structlog + in-memory metrics
│   └── app.py               # Streamlit frontend
├── tests/
│   ├── test_chunker.py
│   └── test_pipeline.py
├── data/
│   ├── raw/                 # Drop source documents here
│   └── chroma/              # ChromaDB persistence (auto-created)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
