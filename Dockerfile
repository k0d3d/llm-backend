ARG PYTHON_VER=3.11

# ============================================================================
# Stage 1: Base system dependencies (rarely changes)
# ============================================================================
FROM python:${PYTHON_VER}-slim AS base

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    ffmpeg \
    libavcodec-extra \
    postgresql-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Python dependencies builder (changes occasionally)
# ============================================================================
FROM python:${PYTHON_VER}-slim AS python-deps

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    build-essential \
    libgl1-mesa-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry with pinned version for reproducibility
ENV POETRY_VERSION=2.0.0 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy only dependency files (better caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies with cache mount (reuses downloads across builds)
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry lock --no-update && \
    poetry install --no-root --no-ansi

# Install additional packages that aren't in poetry (with cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry run pip install \
        "pydantic-ai[openai,anthropic,replicate]==0.2.14" \
        psycopg2-binary==2.9.10

# Install poetry plugins
RUN poetry self add poetry-plugin-dotenv@latest

# ============================================================================
# Stage 3: Final runtime image (minimal, secure)
# ============================================================================
FROM base AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser && \
    chown -R appuser:appuser /app

# Copy virtual environment and poetry from builder
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:/opt/poetry/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=python-deps --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=python-deps --chown=appuser:appuser /opt/poetry /opt/poetry

# Copy application code (do this last for best caching)
COPY --chown=appuser:appuser . /app

# Install the package itself (just creates links, very fast)
RUN --mount=type=cache,target=/tmp/poetry_cache \
    poetry install --only-root

# Switch to non-root user
USER appuser

# Use exec form for proper signal handling
CMD ["bash", "start.sh"]
