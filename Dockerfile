# VocalMatch — Flask + demucs (CPU Docker / Render). For free HF hosting use the
# Gradio ZeroGPU Space entry (`gradio_app.py`); Docker Spaces are paid on some accounts.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    VOCALMATCH_DATA=/data \
    TORCH_HOME=/data/torch \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    VOCALMATCH_FORCE_CPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user \
    && mkdir -p /data /app \
    && chown -R user:user /data /app

WORKDIR /app

COPY --chown=user:user requirements.txt .
# CPU wheels first so the torch>=2.8 pin in requirements.txt is already satisfied.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
EXPOSE 7860

CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 2 \
    --timeout 600 \
    --graceful-timeout 60 \
    --keep-alive 5 \
    app:app
