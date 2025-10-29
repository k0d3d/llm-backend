# RQ Task Queue Implementation for llm-backend

## Quick Start

### Development
```bash
# Option 1: Poetry (recommended for dev)
export PYTHONPATH=$PWD/src
poetry run fastapi dev src/main.py              # Terminal 1: Web server
poetry run python -m llm_backend.workers.worker  # Terminal 2: Worker

# Option 2: Docker Compose (with host Redis/DB)
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=2

# Option 3: Docker Compose (with local Redis/DB)
docker compose --profile redis up
```

### Production
```bash
# Deploy to Dokku
git push dokku master

# Scale workers
dokku ps:scale llm-backend worker=5

# View logs
dokku logs llm-backend -t -p worker
```

---

## Current State Analysis

**llm-backend** uses `BackgroundTasks` in two endpoints:
- `/teams/run` (line 97): `background_tasks.add_task(orchestrator.execute)`
- `/teams/run-hitl` (line 163): `background_tasks.add_task(orchestrator.execute)`
- `/hitl/run` (line 150): `background_tasks.add_task(orchestrator.execute)`
- `/hitl/run/{run_id}/resume` (line 397): `background_tasks.add_task(orchestrator.execute)`

**Limitation**: Same as tohju-py-api - can only handle ~6 concurrent HITL orchestrations at once.

**Already has**:
- Redis dependency (`redis>=6.4.0`)
- Existing Dokku deployment (`.dokku/` folder)
- `start.sh` script for deployment

## Implementation Plan

### 1. Add RQ Dependency
- Add `rq = ">=1.15.0,<2.0.0"` to `pyproject.toml` dependencies

### 2. Create Worker Infrastructure

#### `src/llm_backend/workers/__init__.py`
```python
# Workers package
```

#### `src/llm_backend/workers/connection.py`
```python
"""Redis connection for RQ"""
import os
from redis import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_redis_connection():
    """Get Redis connection for RQ"""
    return Redis.from_url(REDIS_URL)
```

#### `src/llm_backend/workers/tasks.py`
```python
"""Worker tasks for RQ"""
import os
from rq import get_current_job
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus
from llm_backend.core.hitl.shared_bridge import get_shared_state_manager, get_shared_websocket_bridge
from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.core.types.common import RunInput

# Get shared components
state_manager = get_shared_state_manager()
websocket_bridge = get_shared_websocket_bridge()

def process_hitl_orchestrator(run_input_dict: dict, hitl_config_dict: dict, provider_name: str):
    """
    Process HITL orchestrator execution
    Args:
        run_input_dict: Serialized RunInput dictionary
        hitl_config_dict: Serialized HITLConfig dictionary
        provider_name: Provider name (e.g., "replicate")
    Returns:
        Final result
    """
    job = get_current_job()
    job.meta['status'] = 'processing'
    job.save_meta()

    print(f"[HITLWorker] Starting HITL orchestrator for run: {run_input_dict.get('session_id')}")

    # Reconstruct objects
    run_input = RunInput(**run_input_dict)
    hitl_config = HITLConfig(**hitl_config_dict)
    provider = ProviderRegistry.get_provider(provider_name)

    # Create orchestrator
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=hitl_config,
        run_input=run_input,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )

    # Load existing state if this is a resume
    run_id = run_input_dict.get('run_id')
    if run_id:
        import asyncio
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(state_manager.load_state(run_id))
        if state:
            orchestrator.state = state
            print(f"[HITLWorker] Loaded existing state for run {run_id}")

    # Execute orchestrator
    print("[HITLWorker] Executing HITL orchestrator...")
    import asyncio
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(orchestrator.execute())

    print(f"[HITLWorker] Orchestrator completed with status: {orchestrator.state.status}")

    job.meta['status'] = 'completed'
    job.save_meta()

    return {
        "run_id": orchestrator.state.run_id,
        "status": orchestrator.state.status,
        "result": result
    }

def process_hitl_resume(run_id: str, approval_response: dict):
    """
    Process HITL resume after approval
    Args:
        run_id: Run identifier
        approval_response: Approval response dict
    Returns:
        Resume result
    """
    job = get_current_job()
    job.meta['status'] = 'processing'
    job.save_meta()

    print(f"[HITLWorker] Resuming HITL run: {run_id}")

    # Load state
    import asyncio
    loop = asyncio.get_event_loop()
    state = loop.run_until_complete(state_manager.load_state(run_id))

    if not state:
        raise ValueError(f"Run {run_id} not found")

    # Resume the run
    loop.run_until_complete(state_manager.resume_run(run_id, approval_response))

    # Get provider and recreate orchestrator
    provider_name = state.original_input.get("provider", "replicate")
    provider = ProviderRegistry.get_provider(provider_name)

    orchestrator = HITLOrchestrator(
        provider=provider,
        config=state.config,
        run_input=state.original_input,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )

    # Load state with human edits
    orchestrator.state = state
    print(f"[HITLWorker] Loaded state with human_edits: {getattr(state, 'human_edits', 'MISSING')}")

    # Continue execution
    result = loop.run_until_complete(orchestrator.execute())

    print(f"[HITLWorker] Resume completed with status: {orchestrator.state.status}")

    job.meta['status'] = 'completed'
    job.save_meta()

    return {
        "run_id": run_id,
        "status": orchestrator.state.status,
        "result": result
    }
```

#### `src/llm_backend/workers/worker.py`
```python
"""RQ Worker script"""
import os
import sys
import logging
from dotenv import load_dotenv
from rq import Worker, Queue, Connection

# Load environment variables first
load_dotenv()

from llm_backend.workers.connection import get_redis_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def start_worker():
    """Start RQ worker"""
    redis_conn = get_redis_connection()

    with Connection(redis_conn):
        queues = ['default']
        worker = Worker(
            queues,
            connection=redis_conn,
            name=f'worker-{os.getpid()}'
        )
        logger.info(f"Starting RQ worker: {worker.name}")
        logger.info(f"Listening on queues: {queues}")
        worker.work(with_scheduler=False)

if __name__ == '__main__':
    try:
        start_worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
```

### 3. Update API Endpoints

#### `src/llm_backend/api/endpoints/teams.py`

**Changes**:
1. Import RQ components instead of BackgroundTasks
2. Initialize Redis queue at module level
3. Replace `background_tasks.add_task()` with `task_queue.enqueue()`
4. Return job_id in response

```python
from rq import Queue
from llm_backend.workers.connection import get_redis_connection
from llm_backend.workers.tasks import process_hitl_orchestrator

# Initialize Redis Queue
redis_conn = get_redis_connection()
task_queue = Queue('default', connection=redis_conn)

@router.post("/run")
async def run_replicate_team(
    run_input: RunInput,
    # Remove: background_tasks: BackgroundTasks,
    enable_hitl: Optional[bool] = Query(False),
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None)
):
    if enable_hitl:
        # ... existing provider setup code ...

        # Start HITL run
        run_id = await orchestrator.start_run(
            original_input=run_input.dict(),
            user_id=run_input.user_id,
            session_id=run_input.session_id
        )

        # Queue job instead of background task
        job = task_queue.enqueue(
            process_hitl_orchestrator,
            run_input.dict(),
            hitl_config.dict(),
            "replicate",
            job_timeout='30m'
        )

        return {
            "run_id": run_id,
            "job_id": job.id,
            "status": "queued",
            "message": "HITL run started successfully",
            "websocket_url": websocket_bridge.websocket_url,
            "hitl_enabled": True
        }
```

#### `src/llm_backend/api/endpoints/hitl.py`

**Changes**:
1. Import RQ components
2. Initialize Redis queue
3. Replace all `background_tasks.add_task()` calls
4. Add job status endpoint

```python
from rq import Queue
from llm_backend.workers.connection import get_redis_connection
from llm_backend.workers.tasks import process_hitl_orchestrator, process_hitl_resume

# Initialize Redis Queue
redis_conn = get_redis_connection()
task_queue = Queue('default', connection=redis_conn)

@router.post("/run", response_model=HITLRunResponse)
async def start_hitl_run(
    request: HITLRunRequest,
    # Remove: background_tasks: BackgroundTasks
) -> HITLRunResponse:
    # ... existing setup code ...

    # Start run
    run_id = await orchestrator.start_run(
        original_input=request.run_input.dict(),
        user_id=request.user_id,
        session_id=request.session_id
    )

    # Queue job
    job = task_queue.enqueue(
        process_hitl_orchestrator,
        request.run_input.dict(),
        hitl_config.dict(),
        provider_name,
        job_timeout='30m'
    )

    return HITLRunResponse(
        run_id=run_id,
        job_id=job.id,
        status="queued",
        message="HITL run started successfully",
        websocket_url=websocket_bridge.websocket_url if request.session_id else None
    )

@router.post("/run/{run_id}/resume")
async def resume_run(
    run_id: str,
    # Remove: background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    # ... existing validation code ...

    # Queue resume job
    job = task_queue.enqueue(
        process_hitl_resume,
        run_id,
        {"action": "resume", "timestamp": datetime.utcnow().isoformat()},
        job_timeout='30m'
    )

    return {
        "success": True,
        "message": "Run resumed successfully",
        "run_id": run_id,
        "job_id": job.id
    }

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a background job"""
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=redis_conn)
        return {
            "job_id": job.id,
            "status": job.get_status(),
            "result": job.result if job.is_finished else None,
            "error": str(job.exc_info) if job.is_failed else None,
            "meta": job.meta
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")
```

### 4. Create Deployment Files

#### `Procfile`
```
web: poetry run fastapi run src/main.py --host 0.0.0.0 --port $PORT
worker: poetry run python -m llm_backend.workers.worker
```

#### `app.json`
```json
{
  "name": "tohju-llm-backend",
  "description": "LLM Backend with HITL and RQ task queue",
  "formation": {
    "web": {
      "quantity": 1,
      "size": "basic"
    },
    "worker": {
      "quantity": 2,
      "size": "basic"
    }
  },
  "env": {
    "REDIS_URL": {
      "description": "Redis connection URL for RQ",
      "required": true
    },
    "DATABASE_URL": {
      "description": "PostgreSQL database URL",
      "required": true
    },
    "WEBSOCKET_URL": {
      "description": "WebSocket server URL",
      "required": true
    },
    "WEBSOCKET_API_KEY": {
      "description": "WebSocket API key",
      "required": true
    }
  }
}
```

#### `docker-compose.yml`
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: llm_backend
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 5

  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/llm_backend
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    command: poetry run fastapi run src/main.py --host 0.0.0.0 --port 8000

  worker:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/llm_backend
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    command: poetry run python -m llm_backend.workers.worker
    deploy:
      replicas: 2

volumes:
  redis_data:
  postgres_data:
```

### 5. Update pyproject.toml

Add to dependencies:
```toml
"rq (>=1.15.0,<2.0.0)",
```

## Local Development

### Option 1: Poetry (Multiple Terminals) - Recommended

**Best for active development with hot reload.**

```bash
# Terminal 1: Web server (with auto-reload)
poetry run fastapi dev src/main.py

# Terminal 2: Worker
poetry run python -m llm_backend.workers.worker

# Terminal 3: Second worker (optional)
poetry run python -m llm_backend.workers.worker
```

**Requirements:**
- Redis running (Upstash or `docker run -p 6379:6379 redis:7-alpine`)
- PostgreSQL running (or use remote database)
- Set environment variables in `.env`:
  - `REDIS_URL`
  - `DATABASE_URL`
  - `WEBSOCKET_URL`
  - `WEBSOCKET_API_KEY`
  - `REPLICATE_API_TOKEN`

---

### Option 2: Docker Compose with Host Services (Recommended)

**Uses Redis/PostgreSQL running on host machine (localhost).**

This is configured via `docker-compose.local.yml` which uses `host.docker.internal` to connect to services on your host machine.

```bash
# Start web and worker (connects to host services)
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker

# Scale workers
docker compose -f docker-compose.yml -f docker-compose.local.yml up web worker --scale worker=2

# Run in background
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d web worker
```

**Requirements:**
- Redis running on host: `redis://localhost:6379`
- PostgreSQL running on host: `postgresql://localhost:5432/llm_backend`
- Environment variables in `.env.docker` file

**Note:** This setup uses:
- `host.docker.internal` to access host services from Docker
- Port 8811 for web server
- Mounts `.env.docker` file for environment variables

---

### Option 3: Docker Compose with Local Services

**Full stack with local Redis + PostgreSQL.**

```bash
# Start everything including local Redis and PostgreSQL
docker-compose --profile redis up

# Scale workers
docker-compose --profile redis up --scale worker=3

# Rebuild after code changes
docker-compose --profile redis up --build
```

**File:** `docker-compose.yml` (CURRENT)
```yaml
services:
  web:
    build: .
    ports:
      - "8811:8811"
    environment:
      - PYTHONPATH=/app/src
      - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
      - DATABASE_URL=${DATABASE_URL:-postgresql://postgres:postgres@postgres:5432/llm_backend}
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
      - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}
    command: poetry run fastapi run src/main.py --host 0.0.0.0 --port 8811

  worker:
    build: .
    environment:
      - PYTHONPATH=/app/src
      - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
      - DATABASE_URL=${DATABASE_URL:-postgresql://postgres:postgres@postgres:5432/llm_backend}
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
      - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}
    command: poetry run python -m llm_backend.workers.worker

  # Optional: Local Redis (use with --profile redis)
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    profiles:
      - redis

  # Optional: Local PostgreSQL (use with --profile redis)
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: llm_backend
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 5
    profiles:
      - redis

  # Redis-dependent versions (when using Docker Redis)
  web-with-redis:
    extends: web
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/llm_backend
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
      - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}
    profiles:
      - redis

  worker-with-redis:
    extends: worker
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/llm_backend
      - WEBSOCKET_URL=${WEBSOCKET_URL}
      - WEBSOCKET_API_KEY=${WEBSOCKET_API_KEY}
      - REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}
    profiles:
      - redis

volumes:
  redis_data:
  postgres_data:
```

**Note:**
- Port 8811 is used for web (instead of 8000) to avoid conflicts
- Redis on port 6380 (mapped from 6379) to avoid conflicts
- PostgreSQL on port 5433 (mapped from 5432) to avoid conflicts
- `PYTHONPATH=/app/src` is set for proper module imports

---

### Option 4: Manual with Python

```bash
# Terminal 1: Redis (if not using Upstash)
docker run -p 6379:6379 redis:7-alpine

# Terminal 2: PostgreSQL (if not using remote)
docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=llm_backend postgres:15-alpine

# Terminal 3: Web server
export PYTHONPATH=$PWD/src
export REDIS_URL=redis://localhost:6379/0
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_backend
poetry run fastapi dev src/main.py

# Terminal 4: Worker
export PYTHONPATH=$PWD/src
export REDIS_URL=redis://localhost:6379/0
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_backend
poetry run python -m llm_backend.workers.worker
```

---

## Production Deployment (Dokku)

### Prerequisites (One-time Setup)

```bash
# On Dokku server
# 1. Create app
dokku apps:create llm-backend

# 2. Create and link Redis
dokku redis:create tohju-redis
dokku redis:link tohju-redis llm-backend

# 3. Create and link PostgreSQL
dokku postgres:create tohju-postgres
dokku postgres:link tohju-postgres llm-backend

# 4. Set environment variables
dokku config:set llm-backend \
  WEBSOCKET_URL=wss://your-websocket-url \
  WEBSOCKET_API_KEY=your-api-key \
  REPLICATE_API_TOKEN=your-replicate-token \
  SENTRY_DSN=your-sentry-dsn

# 5. Set worker scaling (optional, or use app.json)
dokku ps:scale llm-backend web=1 worker=2
```

### Deploy

```bash
# From your local machine
git remote add dokku dokku@your-server.com:llm-backend
git push dokku master  # or main

# Dokku will:
# 1. Build Docker image
# 2. Start web process (1 instance)
# 3. Start worker process (2 instances)
```

### Production Architecture

```
┌────────────────┐
│     Dokku      │
├────────────────┤
│                │
│   web (1x)     │──┐
│                │  │
│  worker (2x)   │  ├──> Redis Queue
│                │  │
└────────────────┘──┘
      ↓
  Redis Service
  PostgreSQL Service
  WebSocket Service
```

**Procfile:**
```
web: poetry run fastapi run src/main.py --host 0.0.0.0 --port $PORT
worker: poetry run python -m llm_backend.workers.worker
```

**Process Flow:**
1. HITL request → `web` process → Start run + Enqueue job → Return run_id + job_id
2. `worker` process picks up job → Executes HITL orchestrator → WebSocket notifications
3. Human approval via WebSocket → Resume job → Continue execution
4. Repeat for multiple concurrent HITL runs

### Scaling Workers

```bash
# Scale up workers for high load
dokku ps:scale llm-backend worker=5

# Scale down
dokku ps:scale llm-backend worker=2

# Check current scaling
dokku ps:report llm-backend
```

### View Logs

```bash
# All logs
dokku logs llm-backend -t

# Worker logs only
dokku logs llm-backend -t -p worker

# Web logs only
dokku logs llm-backend -t -p web
```

## Key Implementation Notes

### 1. Async Context Handling
Workers run synchronously but HITL orchestrator uses async. Use:
```python
import asyncio
loop = asyncio.get_event_loop()
result = loop.run_until_complete(orchestrator.execute())
```

### 2. State Management
- HITL state is already persisted to Redis/PostgreSQL
- Workers can load and resume state across restarts
- No additional state synchronization needed

### 3. WebSocket Bridge
- WebSocket bridge works from worker context
- No changes needed to notification system
- Workers can send approval requests via shared bridge

### 4. Error Handling
- Job failures are tracked in RQ
- HITL state machine handles orchestrator errors
- Use `job.meta` to track custom status

### 5. Timeout Configuration
- HITL runs can take 5-30 minutes
- Set `job_timeout='30m'` for long-running orchestrations
- Default timeout is 3 minutes (too short)

## Testing Locally

### With Docker Compose
```bash
# Start all services
docker-compose up

# Scale workers
docker-compose up --scale worker=3
```

### Manual (2 terminals)
```bash
# Terminal 1: Web server
poetry run fastapi dev src/main.py

# Terminal 2: Worker
poetry run python -m llm_backend.workers.worker
```

## Monitoring

### Check Queue Status
```python
from redis import Redis
from rq import Queue

redis_conn = Redis.from_url("redis://localhost:6379/0")
queue = Queue('default', connection=redis_conn)

print(f"Queued jobs: {len(queue)}")
print(f"Failed jobs: {queue.failed_job_registry.count}")
```

### Dokku Logs
```bash
# All logs
dokku logs llm-backend -t

# Worker logs only
dokku logs llm-backend -t -p worker

# Web logs only
dokku logs llm-backend -t -p web
```

## Rollback Plan

If issues occur, rollback by:
1. Scale workers to 0: `dokku ps:scale llm-backend worker=0`
2. Revert code: `git revert HEAD`
3. Push: `git push dokku master`

The API will fall back to queuing jobs that won't process until workers restart.

## Benefits

1. **Scalability**: Handle 50+ concurrent HITL runs with 10 workers
2. **Reliability**: Jobs survive server restarts via Redis persistence
3. **Monitoring**: Track job status independently from HITL state
4. **Resource Isolation**: Workers can run on separate machines
5. **Cost Efficiency**: Scale workers independently of web servers
