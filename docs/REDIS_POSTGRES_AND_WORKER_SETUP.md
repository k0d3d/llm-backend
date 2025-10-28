# Redis, Postgres & Worker Setup

This guide explains how to run the FastAPI web app and RQ workers locally and in production while avoiding Redis/Postgres port conflicts and worker name collisions.

## Overview

- `docker-compose.yml` defines the base services (`web`, `worker`, and optional `redis`/`postgres`).
- `docker-compose.local.yml` overrides environment variables for local development when you already have Redis and Postgres running on your machine.
- Workers now generate UUID-based names to avoid duplicate registrations (`src/llm_backend/workers/worker.py`).

## Local Development Options

### Use Existing Local Redis & Postgres Instances

If you already run Redis and Postgres on your host (e.g. via Homebrew or Docker Desktop):

```bash
# Start web + workers against host Redis & Postgres via host.docker.internal
docker compose \
  -f docker-compose.yml \
  -f docker-compose.local.yml \
  up web worker --scale worker=2
```

Notes:

- `docker-compose.local.yml` maps `host.docker.internal` to the Docker host and points `REDIS_URL` and `DATABASE_URL` there, so you can reach existing services without binding container ports.
- Because we no longer rely on host networking, `web` continues to publish `8000:8000`; access the API at `http://localhost:8000`.

### Use the Dockerized Redis & Postgres Services

```bash
# Start Redis + Postgres + web + workers using the redis profile
docker compose --profile redis --profile postgres up \
  web-with-redis worker-with-redis --scale worker-with-redis=2
```

Notes:

- The Redis container is exposed on `6380` externally (`6380:6379`).
- The Postgres container is exposed on `5433` externally (`5433:5432`).
- Dependent services use the internal hostnames `redis` and `postgres` with standard ports.
- When you switch back to the host services, simply bring everything down first: `docker compose down`.

## Worker Scaling & Naming

- Workers obtain a unique name on startup (`worker-<uuid4>`) to avoid the "There exists an active worker named 'worker-1' already" error.
- You can scale workers via `--scale worker=2` (host Redis) or `--scale worker-with-redis=2` (Docker Redis).
- Set a deterministic name by exporting `WORKER_ID` before running Compose if needed, e.g. `WORKER_ID=blue docker compose ...`.

## Troubleshooting

- **Port 6379 or 5432 already allocated**: you're likely running host Redis/Postgres instances. Either stop them or run Compose with the `docker-compose.local.yml` override, which keeps containers on the bridge network while pointing to the host services.
- **Duplicate worker name**: ensure you rebuilt the worker image after the UUID naming change (`docker compose build worker`). Existing worker registrations in Redis can be cleaned via `docker exec redis redis-cli SMEMBERS rq:workers` and `SREM` if necessary.
- **Redis keys from old runs**: `docker exec redis redis-cli KEYS 'rq:*'` to inspect leftover RQ data.
- **Database connection issues**: Verify your host Postgres is accepting connections on `localhost:5432` and the database `llm_backend` exists.

## Quick Reference Commands

```bash
# Shutdown everything
docker compose -f docker-compose.yml -f docker-compose.local.yml down

# Rebuild the worker image after code changes
docker compose -f docker-compose.yml -f docker-compose.local.yml build worker

# Tail logs for a specific service
docker compose logs -f worker

# Scale workers independently
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=3
```

## Environment Variables

The following environment variables can be overridden:

- `REDIS_URL`: Redis connection string (default: `redis://redis:6379/0`)
- `DATABASE_URL`: Postgres connection string (default: `postgresql://postgres:postgres@postgres:5432/llm_backend`)
- `WEBSOCKET_URL`: WebSocket server URL for agent communication
- `WEBSOCKET_API_KEY`: API key for WebSocket authentication
- `REPLICATE_API_TOKEN`: Replicate API token for LLM inference
