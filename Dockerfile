ARG PYTHON_VER=3.11

FROM python:${PYTHON_VER} AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
  cd /usr/local/bin && \
  ln -s /opt/poetry/bin/poetry && \
  poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /app/

RUN apt-get update

RUN apt-get install libpq-dev gcc build-essential wkhtmltopdf ffmpeg libavcodec-extra libgl1-mesa-glx -y

RUN poetry install --no-root

RUN poetry run pip install pydantic-ai

RUN poetry self add poetry-plugin-dotenv@latest

COPY . /app

FROM python:3.11-slim

WORKDIR /app

COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /opt/poetry /opt/poetry

CMD ["bash", "start.sh"]
