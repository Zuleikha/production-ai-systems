# Resume Screening System

AI-powered resume screening using semantic embeddings. Parses PDF/DOCX resumes, generates embeddings with sentence-transformers, ranks candidates against job requirements, and validates data quality with Pydantic v2 schemas.

## Architecture

```
Resume upload (PDF/DOCX/text)
      ↓
PDFParser / python-docx → plain text
      ↓
sentence-transformers (all-MiniLM-L6-v2)
→ semantic embeddings
      ↓
Cosine similarity vs. job description embedding
→ ranked candidate list
      ↓
FastAPI /rank endpoint (JSON)
      ↓
DataQualityChecker → completeness, duplicate detection, consistency checks
BiasMonitor → demographic parity, equal opportunity analysis
```

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF parsing | pypdf (maintained successor to PyPDF2) |
| Schema validation | Pydantic v2 (`field_validator`, `model_validator`) |
| Config | pydantic-settings with `.env` support |
| API | FastAPI + Uvicorn |
| Data quality | pandas-based assertions + `DataQualityMetrics` schema |
| Bias monitoring | `BiasMonitor` (demographic parity, equal opportunity) |
| Test data | Faker-based synthetic resume generator |

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic test data
python -m src.data_generation.resume_generator

# Start API
uvicorn src.api.main:app --reload
```

## Data Schemas

`ResumeSchema` (Pydantic v2) validates:
- `ContactInfo` — email regex, optional phone/LinkedIn/GitHub
- `Skill` — name, optional level (`beginner/intermediate/advanced/expert`), years experience
- `Education` — degree, institution, graduation year, GPA
- `Experience` — title, company, ISO date range, description (20-2000 chars), skills used
- Cross-field validator: `start_date < end_date` for all experience entries

## Data Quality

`DataQualityChecker` reports:
- Field completeness percentages
- Duplicate detection (key-column and content-hash)
- Consistency checks (email format, date format)
- Overall quality score (`valid / total * 100`)

## What Changed (Modernisation)

| Before | After |
|---|---|
| `PyPDF2==3.0.1` (deprecated) | `pypdf>=4.3.0` (maintained successor) |
| `sentence-transformers==2.2.2` | `>=3.0.0` (major version with API changes) |
| Exact pins (`==`) on all deps | Minimum pins (`>=`) |
| `sqlalchemy`, `alembic`, `spacy`, `mlflow` | Removed — not used in codebase |
| `black` + `flake8` | `ruff` (covers both) |
| Pydantic v1 `BaseSettings` in config.py | `pydantic_settings.BaseSettings` (v2) |
| `class Config:` nested in settings | `model_config = SettingsConfigDict(...)` |
| Verbose comment-heavy config | Clean typed fields with `Field()` validators |
