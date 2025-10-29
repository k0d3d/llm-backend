# Docker Compose Reference for llm-backend

## Quick Commands

### Option 1: With Host Services (Recommended for Local Dev)

**Use this when you have Redis and PostgreSQL running on your host machine.**

```bash
# Start web + 2 workers
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=2

# Run in background
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d web worker --scale worker=2

# Stop all services
docker compose -f docker-compose.yml -f docker-compose.local.yml down

# View logs
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f

# Rebuild after code changes
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build web worker
```

**Requirements:**
- Redis running: `redis://localhost:6379`
- PostgreSQL running: `postgresql://localhost:5432/llm_backend`
- `.env.docker` file with environment variables

---

### Option 2: With Docker Services

**Use this when you want Redis and PostgreSQL to run in Docker.**

```bash
# Start everything (web, worker, redis, postgres)
docker compose --profile redis up

# With 3 workers
docker compose --profile redis up --scale worker=3

# Run in background
docker compose --profile redis up -d

# Stop all services
docker compose --profile redis down

# Stop and remove volumes (CAUTION: Deletes data)
docker compose --profile redis down -v

# Rebuild
docker compose --profile redis up --build
```

---

## Configuration Files

### docker-compose.yml
Main configuration with:
- `web` service (port 8811)
- `worker` service
- `redis` service (port 6380, profile: redis)
- `postgres` service (port 5433, profile: redis)

### docker-compose.local.yml
Override for connecting to host services:
- Uses `host.docker.internal` to access host
- Mounts `.env.docker` file
- Same environment variables as main config

---

## Environment Variables

### Required in `.env.docker`
```env
REDIS_URL=redis://host.docker.internal:6379/0
DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/llm_backend
WEBSOCKET_URL=wss://your-websocket-url
WEBSOCKET_API_KEY=your-api-key
REPLICATE_API_TOKEN=your-replicate-token
```

---

## Accessing Services

### Web Server
- URL: `http://localhost:8811`
- Docs: `http://localhost:8811/docs`

### Redis (when using Docker)
- Host: `localhost:6380`
- Connection: `redis://localhost:6380/0`

### PostgreSQL (when using Docker)
- Host: `localhost:5433`
- Connection: `postgresql://postgres:postgres@localhost:5433/llm_backend`

---

## Troubleshooting

### Workers not starting (ImportError)
**Issue:** `ImportError: cannot import name 'Connection' from 'rq'`

**Solution:** Already fixed in worker.py (RQ 2.6.0 removed `Connection` from imports)

---

### Cannot connect to Redis/PostgreSQL
**Issue:** `Connection refused` errors

**For docker-compose.local.yml:**
- Ensure Redis/PostgreSQL are running on host
- Check they're accessible on `localhost:6379` and `localhost:5432`

**For docker compose --profile redis:**
- Services should start automatically
- Check logs: `docker compose logs redis postgres`

---

### Port conflicts
**Issue:** Port already in use

**Solutions:**
- Web (8811): Change in `docker-compose.yml` ports mapping
- Redis (6380): Already mapped to avoid conflicts with default 6379
- PostgreSQL (5433): Already mapped to avoid conflicts with default 5432

---

## Common Workflows

### Fresh Start
```bash
# Stop everything
docker compose -f docker-compose.yml -f docker-compose.local.yml down

# Rebuild
docker compose -f docker-compose.yml -f docker-compose.local.yml build

# Start fresh
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=2
```

### Check Worker Status
```bash
# View worker logs
docker compose logs worker-1 worker-2 -f

# See all containers
docker compose ps
```

### Scale Workers
```bash
# Change worker count on the fly
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d worker --scale worker=5

# Or specify in up command
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=5
```

---

## Aliases (Optional)

Add to your shell config (`.bashrc`, `.zshrc`):

```bash
# llm-backend docker compose shortcuts
alias dcl='docker compose -f docker-compose.yml -f docker-compose.local.yml'
alias dcr='docker compose --profile redis'

# Usage:
# dcl up web worker --scale worker=2
# dcr up
```

---

*Last Updated: 2025-10-28*
