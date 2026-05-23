# python:3.11-slim omits compilers, test tools, and most documentation that ship in the full
# python:3.11 image, cutting the final image size from ~1 GB to ~130 MB. The OpenCV runtime
# deps below (libgl1-mesa-dev etc.) are the only system libraries this service actually needs.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]