# AI Engineering Portfolio

Four production-ready AI systems covering RAG, MLOps, computer vision, and NLP — each with a FastAPI serving layer, Docker deployment, and full modernisation to current tooling.

## Projects

| # | Project | Description | Tech Stack |
|---|---|---|---|
| 01 | [RAG Assistant](./01-rag-assistant) | Hybrid retrieval (dense + BM25) with cross-encoder reranking and gpt-4o-mini generation | FastAPI, Streamlit, ChromaDB, OpenAI, sentence-transformers |
| 02 | [MLOps Pipeline](./02-mlops-pipeline) | BERT fine-tuning on IMDB with asset-based orchestration and experiment tracking | Dagster, MLflow, HuggingFace Transformers, FastAPI |
| 03 | [Computer Vision](./03-computer-vision) | Object detection with Faster R-CNN ResNet-50 FPN and configurable NMS post-processing | PyTorch, Torchvision, OpenCV, FastAPI |
| 04 | [Resume Screening](./04-resume-screening) | Semantic candidate ranking with Pydantic v2 schema validation and bias monitoring | sentence-transformers, pypdf, FastAPI, Pydantic v2 |

## Quick Start

Each project runs independently. See its `README.md` for setup.

To run all services via Docker Compose:

```bash
cp .env.example .env   # add OPENAI_API_KEY
docker compose up
```

| Service | URL |
|---|---|
| RAG API | http://localhost:8000 |
| RAG UI (Streamlit) | http://localhost:8501 |
| MLOps API | http://localhost:8001 |
| MLflow UI | http://localhost:5000 |
| Vision API | http://localhost:8002 |
| Resume API | http://localhost:8003 |

## Shared Patterns

- **Config**: pydantic-settings `BaseSettings` with `.env` support across all projects
- **API**: FastAPI with async lifespan (no deprecated `@app.on_event`)
- **Linting**: `ruff` (replaces black + flake8) across all projects
- **PDF parsing**: `pypdf` (01, 04) — PyPDF2 removed
- **Docs**: Each project has `INTERVIEW_NOTES.md` (gitignored) with architecture decisions and Q&As
