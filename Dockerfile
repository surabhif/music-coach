# VocalMatch — Flask + demucs (CPU). Works on Hugging Face Spaces (Docker SDK) and any
# host that runs Docker. Spaces expects the app on port 7860 by default.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    VOCALMATCH_DATA=/data \
    TORCH_HOME=/data/torch \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as uid 1000
RUN useradd -m -u 1000 user \
    && mkdir -p /data /app \
    && chown -R user:user /data /app

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
EXPOSE 7860

# Single worker: demucs + torch are memory-heavy. Long timeout for vocal isolation.
CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 2 \
    --timeout 600 \
    --graceful-timeout 60 \
    --keep-alive 5 \
    app:app
