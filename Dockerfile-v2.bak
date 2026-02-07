ARG PYTHON_VER=3.11-slim

# ============================================================================
# Stage 1: Base Runtime Environment (minimal system libs)
# ============================================================================
FROM python:${PYTHON_VER} AS base

# Install runtime system dependencies
# Kept ffmpeg and libgl1 as requested for inspection
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    ffmpeg \
    libavcodec-extra \
    postgresql-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Dependency Builder (compiled libs and poetry)
# ============================================================================
FROM python:${PYTHON_VER} AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=2.0.0 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Build virtual environment
COPY pyproject.toml poetry.lock* ./
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-root --no-ansi --without test

# ============================================================================
# Stage 3: Final Runtime Image
# ============================================================================
FROM base AS runtime

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser && \
    chown -R appuser:appuser /app

# Environment variables (Directly use the virtualenv)
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Copy ONLY the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Use exec form for proper signal handling
CMD ["bash", "start.sh"]