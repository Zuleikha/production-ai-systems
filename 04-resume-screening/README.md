> Part of the **AI Engineering Portfolio**

# AI Resume Screening System (NLP)

## Project Overview
Production-ready AI-powered resume screening system built to understand
end-to-end AI engineering workflows using modern NLP techniques.

The focus of this project is not just model performance, but system design:
document parsing, embedding pipelines, ranking logic, API serving, and
deployment considerations.

## What the System Does
- Parses resumes and job descriptions
- Generates semantic embeddings using transformer models
- Computes similarity scores between candidates and roles
- Ranks and scores candidates based on relevance
- Exposes inference through a RESTful API

## Why I Built It
I built this project to learn how NLP systems behave in production rather than
in isolated notebooks.

Key learning goals included:
- Designing embedding-based similarity pipelines
- Handling structured and unstructured document inputs
- Implementing ranking logic over model outputs
- Building API-backed inference services
- Packaging and deploying systems in a reproducible way

## Tech Stack
- Python
- Transformer-based embeddings
- FastAPI
- Docker / Docker Compose
- REST APIs

## Project Structure
- `src/` – Core application logic
- `data/` – Resume and job description data
- `docs/` – API, model, and deployment documentation
- `deployment/` – Deployment and infrastructure configuration
- `tests/` – Automated tests

## Running the Project (Local)
```bash
pip install -r requirements.txt
python -m src.data_generation.resume_generator
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000


