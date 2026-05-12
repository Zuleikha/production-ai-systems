# DITA RAG Pipeline

A production-ready pipeline that ingests DITA XML documentation, transforms and enriches it with metadata, indexes it into a vector store, and serves semantic search and Q&A over the content via a FastAPI backend and Streamlit UI.

## What it does

- Parses DITA XML files into structured chunks by topic and section
- Enriches each chunk with metadata: topic title, section title, source file, keyword tags
- Embeds chunks using OpenAI text-embedding-3-small
- Stores embeddings in ChromaDB with metadata for filtered retrieval
- Serves a REST API with /search and /ask endpoints
- Provides a Streamlit UI for interactive Q&A

## Architecture

```
data/                    DITA XML source files
app/
  parser.py              DITA XML parsing and chunking
  indexer.py             Embedding and ChromaDB ingestion
  retriever.py           Semantic search and answer generation
  api.py                 FastAPI endpoints
  ui.py                  Streamlit UI
Dockerfile
docker-compose.yml
requirements.txt
```

## Quick Start

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| UI | http://localhost:8501 |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /index | Re-parse and re-index all DITA files |
| POST | /search | Semantic search, returns ranked chunks |
| POST | /ask | RAG Q&A, returns answer with sources |

### Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I fix a license error?", "n_results": 4}'
```

## Adding Your Own DITA Files

Drop `.dita` files into the `data/` directory and call `POST /index` to re-index.

## Tech Stack

Python, FastAPI, ChromaDB, OpenAI, Streamlit, Docker
