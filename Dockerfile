# ── Stage 1: Python base ──────────────────────────────────────────────────────
FROM python:3.10-slim as base

WORKDIR /app

# System dependencies for OpenCV, Tesseract, pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Dependencies ─────────────────────────────────────────────────────
FROM base as deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ──────────────────────────────────────────────────────
FROM deps as app

COPY . .

# Pre-create required directories
RUN mkdir -p data/raw_documents data/synthetic_annotations data/processed \
    models logs

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
