# =============================================================================
# Invisible Variables Engine — Production Multi-Stage Dockerfile
# =============================================================================
#
# Build targets:
#   api        — FastAPI application server (uvicorn)
#   worker     — Celery worker (ML pipeline tasks)
#   streamlit  — Streamlit frontend dashboard
#
# Usage:
#   docker build --target api       -t ive-api:latest .
#   docker build --target worker    -t ive-worker:latest .
#   docker build --target streamlit -t ive-streamlit:latest .
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# BASE STAGE — shared foundation for all targets
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="IVE Engineering Team" \
    org.opencontainers.image.title="Invisible Variables Engine" \
    org.opencontainers.image.description="Data science platform for latent variable discovery" \
    org.opencontainers.image.version="0.1.0"

# --- Environment ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    # Poetry configuration
    POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_ANSI=0 \
    # pip
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- System dependencies ---
# Kept in a single RUN to minimise layers; cache cleaned in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Install Poetry ---
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# --- Install Python dependencies (cached layer) ---
# Copy only the dependency manifests first so this layer is rebuilt only when
# dependencies change, not when source code changes.
COPY pyproject.toml ./
# poetry.lock is optional — if it exists it will be copied; if not, Poetry
# resolves fresh (dev convenience). In CI always commit a lock file.
COPY poetry.loc[k] ./

RUN poetry install --no-root --no-interaction --no-ansi \
    && rm -rf /root/.cache/pypoetry

# --- Copy application source ---
COPY src/          ./src/
COPY streamlit_app/ ./streamlit_app/
COPY alembic/      ./alembic/
COPY alembic.ini   ./

# --- Create non-root user ---
RUN groupadd --gid 1001 ive \
    && useradd  --uid 1001 --gid ive --shell /bin/bash --create-home ive \
    && chown -R ive:ive /app

USER ive


# ─────────────────────────────────────────────────────────────────────────────
# API STAGE — FastAPI + uvicorn
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "ive.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]


# ─────────────────────────────────────────────────────────────────────────────
# WORKER STAGE — Celery ML pipeline worker
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS worker

# Workers are CPU-bound; no HTTP port needed.
# Concurrency and queues are overideable at runtime via env / compose command.

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A ive.workers.celery_app inspect ping -d "celery@$$HOSTNAME" || exit 1

CMD ["celery", "-A", "ive.workers.celery_app", "worker", \
    "--loglevel=INFO", \
    "--concurrency=4", \
    "--queues=default,analysis,high_priority"]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT STAGE — Streamlit frontend dashboard
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS streamlit

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "/app/streamlit_app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
