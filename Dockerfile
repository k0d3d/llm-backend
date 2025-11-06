ARG PYTHON_VER=3.11

FROM python:${PYTHON_VER}

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    build-essential \
    ffmpeg \
    libavcodec-extra \
    libgl1-mesa-dev \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
  cd /usr/local/bin && \
  ln -s /opt/poetry/bin/poetry && \
  poetry config virtualenvs.create false

# Copy only dependency files first (better cache)
COPY ./pyproject.toml ./poetry.lock* /app/

# Install all dependencies in one layer with cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-root --only main && \
    poetry run pip install pydantic-ai==0.2.14 psycopg2-binary==2.9.10 && \
    poetry self add poetry-plugin-dotenv@latest

# Copy source code (changes frequently, separate layer)
COPY . /app

# Install the package itself (fast, just links)
RUN poetry install --only-root

CMD ["bash", "start.sh"]
