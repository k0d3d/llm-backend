ARG PYTHON_VER=3.11

FROM python:${PYTHON_VER}

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
  cd /usr/local/bin && \
  ln -s /opt/poetry/bin/poetry && \
  poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /app/

# Install Python dependencies
RUN poetry install --no-root

# Install additional packages
RUN poetry run pip install pydantic-ai==0.2.14
RUN poetry run pip install psycopg2-binary==2.9.10

RUN poetry self add poetry-plugin-dotenv@latest

# Verify PostgreSQL dialect is available
RUN python -c "import sqlalchemy; from sqlalchemy.dialects import postgresql; print('PostgreSQL dialect loaded successfully')"

COPY . /app

CMD ["bash", "start.sh"]
