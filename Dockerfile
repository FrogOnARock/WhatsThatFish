# WhatsThisFish serving image — CPU-only, torch-free, ONNX inference.
#
# Goal: a small image that scale-to-zero Cloud Run can pull fast. That means NO
# torch / ultralytics / ray — only onnxruntime + numpy + PIL + the FastAPI/DB
# layer. The serving code is already import-torch-free (verified), so the ONLY
# thing keeping the image slim is what we INSTALL and COPY here.
#
# Build context is this directory (whatsthatfish/). .dockerignore keeps the
# parquet/weights/venv out.

# ── base ────────────────────────────────────────────────────────────────────
# python:3.12-slim, NOT a CUDA base: no GPU at serving time, so no nvidia libs.
# 'slim' (Debian) over 'alpine' because onnxruntime/numpy ship manylinux wheels
# that need glibc — alpine's musl would force slow source builds.
FROM python:3.12-slim AS runtime

# Fail fast, no .pyc writes, unbuffered logs (so Cloud Logging sees stdout live).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false

WORKDIR /app

# ── system deps ───────────────────────────────────────────────────────────────
# onnxruntime needs libgomp (OpenMP runtime). Pillow's wheels are self-contained.
# Keep this list minimal — every apt package is image weight.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── python deps ───────────────────────────────────────────────────────────────
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --no-install-project --frozen


# ── application code + model artifacts ────────────────────────────────────────
COPY src/ ./src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src

# ── security ──────────────────────────────────────────────────────────────────
RUN useradd -ms /bin/bash appmanager
USER appmanager


# ── runtime ───────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "exec uvicorn whatsthatfish.serving.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
