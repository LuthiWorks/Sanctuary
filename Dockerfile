# The Sanctuary - Multi-stage Docker build
# Build: docker build -t sanctuary .
# Run:   docker run -it --rm sanctuary

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

WORKDIR /app
# NOTE: there is no setup.py in this repo and there never was one at this path.
# This COPY used to name it, which made the build fail on this line -- no image
# had been built since 2026-05-11. Verified 2026-08-16.
COPY pyproject.toml README.md ./
COPY sanctuary ./sanctuary

# GPU NOTE: this image is CPU-only, deliberately and explicitly.
# The project's GPU path on Windows is torch-directml, which is a DirectX API
# and does not exist inside a Linux container. Brian ruled on 2026-08-16 that
# ROCm is the path forward; a ROCm image (which needs /dev/kfd and /dev/dri
# passed through) replaces this once Docker exists on the machine to test it.
# Until then, do not let this image imply GPU capability it does not have.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies.
#
# This used to be `pip install . || pip install <hand-written package list>`.
# That fallback is exactly the failure mode CLAUDE.md forbids: the real
# resolution was failing (the spec demanded torch>=2.13.0, transformers>=5.5.0
# and requires-python>=3.11 against an environment that had none of those), and
# the fallback quietly produced an image with different dependencies that
# reported success. If resolution fails, the build must stop here.
RUN pip install --no-cache-dir .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 sanctuary && \
    useradd --uid 1000 --gid sanctuary --shell /bin/bash --create-home sanctuary

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=sanctuary:sanctuary . .

RUN mkdir -p /app/data/memories /app/data/chroma /app/data/checkpoints /app/data/logs && \
    chown -R sanctuary:sanctuary /app/data

USER sanctuary

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/sanctuary:/app \
    SANCTUARY_BASE_DIR=/app/data \
    SANCTUARY_CHROMA_DIR=/app/data/chroma \
    SANCTUARY_LOG_DIR=/app/data/logs \
    SANCTUARY_CHECKPOINT_DIR=/app/data/checkpoints \
    SANCTUARY_IDENTITY_DIR=/app/data/identity \
    SANCTUARY_HEALTH_PORT=8000 \
    SANCTUARY_RESTORE_LATEST=true

EXPOSE 8000

# Health check — hit the HTTP health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "sanctuary.run_cognitive_core"]
