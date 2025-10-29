# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tohju LLM Backend - A FastAPI-based Human-in-the-Loop (HITL) system for managing AI provider executions with human oversight. The system provides a provider-agnostic architecture supporting multiple AI providers (Replicate, OpenAI, Anthropic, etc.) with configurable human checkpoints for approval workflows.

## Key Commands

### Development
```bash
# Install dependencies
poetry install

# Run development server (default port 8000)
poetry run fastapi dev src/main.py

# Run with custom port
poetry run fastapi run src/main.py --port 8080

# Production start (uses PORT env var)
bash start.sh
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_hitl_orchestrator.py

# Run with verbose output
poetry run pytest -v

# Run specific test function
poetry run pytest tests/test_hitl_orchestrator.py::test_function_name

# Run HITL integration tests
poetry run python tests/run_hitl_tests.py
```

### Database
```bash
# Run migrations (Alembic)
poetry run alembic upgrade head

# Create new migration
poetry run alembic revision --autogenerate -m "description"

# Rollback migration
poetry run alembic downgrade -1
```

### Docker
```bash
# Build image
docker build -t tohju-llm-backend .

# Run container
docker run -p 8000:8000 --env-file .env tohju-llm-backend
```

## Architecture

### Core System Components

**Provider Layer** (`src/llm_backend/core/providers/`, `src/llm_backend/providers/`)
- `AIProvider` abstract base class defines the contract for all AI providers
- Each provider implements: `get_capabilities()`, `create_payload()`, `validate_payload()`, `execute()`, `audit_response()`
- Current implementations: `ReplicateProvider`
- Extensible for OpenAI, Anthropic, HuggingFace, etc.

**HITL Orchestration Layer** (`src/llm_backend/core/hitl/`)
- `HITLOrchestrator` manages workflow state machine with human checkpoints
- `HITLState` tracks run lifecycle across async operations
- Checkpoints: `FORM_INITIALIZATION`, `INFORMATION_REVIEW`, `PAYLOAD_REVIEW`, `API_CALL`, `RESPONSE_REVIEW`, `COMPLETED`
- State persistence via Redis and/or PostgreSQL (configurable)
- WebSocket bridge for real-time approval flows

**API Layer** (`src/llm_backend/api/`)
- FastAPI endpoints in `endpoints/teams.py` and `endpoints/hitl.py`
- Main routes: `POST /hitl/runs`, `GET /hitl/runs/{id}`, `POST /hitl/runs/{id}/approve`, `POST /hitl/runs/{id}/edit`

**Form-Based Workflow** (`src/llm_backend/agents/`)
- `FormFieldClassifierAgent` uses gpt-4.1-mini-mini to classify fields in `example_input` as CONTENT/CONFIG/HYBRID
- `AttachmentMappingAgent` uses gpt-4.1-mini-mini for semantic attachment-to-field mapping (understands "image" = "input_image" = "img")
- Intelligently resets form fields (arrays → [], content fields → null, config → keep defaults)
- Pre-populates forms from user attachments using AI-powered field matching with heuristic fallback

**Legacy Multi-Agent System** (`src/dynamic_mas/`)
- Original dynamic multi-agent orchestration (CEO agent, store manager, customer service)
- Task orchestration with flow control
- WooCommerce integration tools

### Important Data Flow

1. **RunInput** arrives with `prompt`, `session_id`, `agent_tool_config`, `hitl_config`
2. **Provider Selection**: Based on `agent_tool_config` enum (e.g., `replicate-agent-tool`)
3. **Orchestrator Initialization**: Creates `HITLOrchestrator(provider, config, run_input, state_manager, websocket_bridge)`
4. **Form Initialization** (if form-based): Classify fields, reset arrays/content, pre-populate from attachments
5. **Information Review**: Show capabilities, model config; pause if thresholds not met
6. **Payload Review**: Validate payload against schema; auto-approve if 0 blocking issues (10s countdown), else require human input
7. **API Execution**: Provider calls external API
8. **Response Review**: Audit response quality; pause if needed
9. **Completion**: Return final result or send to callback URL

### Critical Patterns

**Attachment Discovery**: The system no longer expects `document_url` in `RunInput`. Instead, `_gather_attachments()` inspects recent chat history for the `session_id`/`user_id` to assemble candidate assets. If a model requires an attachment and none are found, validation surfaces this as a blocking issue.

**AI-Powered Attachment Mapping**: Uses `AttachmentMappingAgent` (gpt-4.1-mini-mini) for semantic field matching:
- Understands field name equivalence: "image" = "input_image" = "img" = "photo"
- Detects file types from URLs: .jpg → image field, .mp3 → audio field
- Handles both single string fields AND array fields (not just arrays)
- Prioritizes CONTENT fields over CONFIG fields
- Falls back to improved heuristic matching with 0.9+ confidence for exact matches
- Critical fix: Previous hardcoded logic only checked array fields, missing single "image" fields

**Auto-Approve Behavior**: When `validation_summary.blocking_issues == 0`, the frontend renders a 10-second countdown with auto-approve. Non-zero blocking issues require human action.

**State Persistence**: HITL runs are persisted to handle async operations and system restarts. Use `HITLStateStore` (Redis or PostgreSQL) for saving/loading state.

**Constructor Pattern for Orchestrator**: Always initialize with both `state_manager` and `websocket_bridge` for proper database persistence and approval flow:
```python
orchestrator = HITLOrchestrator(
    provider=provider_instance,
    config=hitl_config,
    run_input=run_input,
    state_manager=state_manager,  # REQUIRED
    websocket_bridge=websocket_bridge  # REQUIRED
)
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `DATABASE_URL`: Supports `redis://`, `postgresql://`, or `hybrid://` (see `.env.example` for details)
- `WEBSOCKET_URL`: WebSocket server for real-time HITL communication
- `WEBSOCKET_API_KEY`: API key for WebSocket REST endpoints
- `REPLICATE_API_TOKEN`: Replicate API token
- `SENTRY_DSN`: Optional error tracking
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR

## Testing Conventions

- Tests use `pytest` with `pytest-asyncio` for async tests
- Mock WebSocket connections and provider API calls
- Test fixtures defined in `tests/conftest.py`
- HITL tests cover: core functionality, database integrity, session resumability, edge cases, form workflows

## Key Files to Reference

- `docs/ARCHITECTURE.md` - Provider-agnostic HITL architecture overview
- `docs/HITL_ORCHESTRATOR.md` - Detailed orchestrator design with code examples
- `docs/FORM_BASED_HITL.md` - Form workflow with field classification
- `docs/API_REFERENCE.md` - API endpoint specifications
- `docs/FRONTEND_INTEGRATION.md` - Frontend integration guide
- `src/llm_backend/core/hitl/orchestrator.py` - Main orchestrator implementation
- `src/llm_backend/core/providers/base.py` - AIProvider abstract interface
- `src/llm_backend/providers/replicate_provider.py` - Replicate implementation
- `src/llm_backend/agents/attachment_mapper.py` - AI-powered attachment-to-field mapping
- `src/llm_backend/agents/form_field_classifier.py` - Field classification agent
- `src/llm_backend/agents/field_analyzer.py` - Field analysis and replaceable field detection

## Development Notes

- Python 3.11+ required
- Uses Poetry for dependency management
- FastAPI with uvicorn for async HTTP server
- Pydantic for data validation
- SQLAlchemy + Alembic for database migrations
- Both Redis and PostgreSQL supported (hybrid mode recommended for production)
