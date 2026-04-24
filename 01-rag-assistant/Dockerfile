FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip --quiet

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the reranker model so the first query is not slow
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

RUN mkdir -p data/raw data/processed data/chroma

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
