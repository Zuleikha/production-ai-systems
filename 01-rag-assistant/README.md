# RAG Assistant — v2

Retrieval-Augmented Generation system with hybrid retrieval and Cohere reranking. Accepts PDF and text documents, indexes them in Pinecone, and answers questions with inline citations.

## Tech stack

| Layer | Technology |
|---|---|
| LLM | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` (1536-dim) |
| Vector store | Pinecone (serverless) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranker | Cohere Rerank API (`rerank-v3.5`) via `httpx` |
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
                            │                    └─ Pinecone (upsert)       │
                            │                                               │
                            │  POST /query ───► RAGPipeline                 │
                            │                    └─ HybridRetriever         │
                            │                        ├─ Dense (Pinecone)    │
                            │                        ├─ Sparse (BM25)       │
                            │                        └─ RRF fusion          │
                            │                    └─ CohereReranker          │
                            │                    └─ RAGGenerator (gpt-4o-mini) │
                            └───────────────────────────────────────────────┘
```

## What changed from v1 — and why

| Component | v1 | v2 | Why |
|---|---|---|---|
| Embedding model | `text-embedding-ada-002` | `text-embedding-3-small` | Better MTEB scores, cheaper, supports dimensionality reduction |
| LLM | `gpt-3.5-turbo` | `gpt-4o-mini` | Better instruction following for citation format, higher knowledge cutoff |
| Chunking | `RecursiveCharacterTextSplitter` (1000 chars) | `SentenceAwareChunker` (512 tokens) | Never cuts mid-sentence; token-based sizes align with LLM context windows |
| Vector store | Custom JSON file → ChromaDB | Pinecone (serverless) | Managed hosting, no local storage, zero RAM overhead vs ChromaDB's HNSW in-process |
| Retrieval | Dense-only, top-3 | Dense + BM25 + RRF, top-10 before rerank | Sparse leg catches exact-term matches that embeddings miss |
| Reranker | None → CrossEncoder (local) | Cohere Rerank API via `httpx` | Local sentence-transformers + PyTorch = 400 MB RAM at import; Cohere is an HTTP call with zero local overhead |
| PDF parsing | `PyPDF2` (deprecated 2023) | `pypdf` | PyPDF2 is unmaintained; pypdf is the official successor |
| Configuration | Plain class with `os.getenv` | `pydantic-settings` `BaseSettings` | Type validation, automatic env-var binding, clear failure if required vars missing |
| Monitoring | `print()` | `structlog` + `/metrics` endpoint | Machine-readable logs, latency/token tracking exposed via API |
| Testing | Import checks | `pytest` with `pytest-asyncio` and mocks | Tests orchestration logic, not just imports |
| LangChain | Used for text splitting | Removed | Was a heavy dependency for 10 lines; native implementation is clearer |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # set OPENAI_API_KEY, PINECONE_API_KEY
                              # optionally set COHERE_API_KEY for reranking
python -m uvicorn src.api.main:app --reload
streamlit run src/app.py      # optional frontend
```

**Docker:**

```bash
cp .env.example .env
docker compose up --build
```

API → `localhost:8000` · Frontend → `localhost:8501` · Swagger → `localhost:8000/docs`

**Live deployment:** https://rag-assistant-v10v.onrender.com

**Required environment variables:**

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings + generation |
| `PINECONE_API_KEY` | Yes | Vector store |
| `COHERE_API_KEY` | No | Reranking (falls back to embedding score order if absent) |

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

129 tests, no API keys required — all external calls (OpenAI, Pinecone, Cohere) are mocked.

| Test file | Tests | What it covers |
|---|---|---|
| `test_chunker.py` | 9 | Sentence splitting, token counting, overlap, metadata |
| `test_pipeline.py` | 9 | Orchestration, conversation history, empty-store path |
| `test_retriever.py` | 14 | RRF score formula (k=60, additivity, deduplication), BM25 integration |
| `test_reranker.py` | 9 | No-key fallback, API failure fallback, rerank ordering, metadata preservation |
| `test_generator.py` | 12 | Context formatting, message building, history truncation, mocked generation |
| `test_metrics.py` | 14 | Accumulation, averages, rolling-window cap, `to_dict`, singleton |
| `test_api.py` | 33 | All routes — status codes, input validation boundaries, response shapes |
| `test_ingestor.py` | 20 | PDF extraction, file/directory ingestion, error isolation, chunk metadata |

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
│   │   ├── reranker.py      # Cohere Rerank API via httpx
│   │   ├── generator.py     # GPT-4o-mini answer generation
│   │   └── pipeline.py      # Orchestration
│   ├── store/
│   │   └── pinecone_store.py  # Pinecone wrapper
│   ├── ingestion/
│   │   └── ingest.py        # PDF + text document loader
│   ├── monitoring/
│   │   └── metrics.py       # structlog + in-memory metrics
│   └── app.py               # Streamlit frontend
├── tests/
│   ├── conftest.py          # CI-safe env setup, metrics reset fixture
│   ├── test_chunker.py      # SentenceAwareChunker unit tests
│   ├── test_pipeline.py     # RAGPipeline integration tests (mocked)
│   ├── test_retriever.py    # HybridRetriever RRF logic + integration
│   ├── test_reranker.py     # CohereReranker fallback + API paths
│   ├── test_generator.py    # RAGGenerator formatting + mocked generate()
│   ├── test_metrics.py      # RequestMetrics accumulation and to_dict
│   ├── test_api.py          # FastAPI route tests (mocked pipeline)
│   └── test_ingestor.py     # DocumentIngestor PDF + file + directory
├── data/
│   └── raw/                 # Drop source documents here
├── pytest.ini
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
