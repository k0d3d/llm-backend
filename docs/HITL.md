# Human-in-the-Loop (HITL) Design - Provider-Agnostic Architecture

This document outlines the provider-agnostic Human-in-the-Loop (HITL) system that enables controlled human review at key steps of AI model execution across multiple providers (Replicate, OpenAI, Anthropic, etc.). The architecture separates HITL workflow management from specific provider implementations.

## Goals

- Enable controlled human review at key steps of the model run.
- Improve safety, accuracy, and user intent alignment while preserving automation when safe.
- Maintain a clear audit trail of automated vs human decisions, edits, and outcomes.
- Introduce minimal disruption to current `POST /teams/run` while paving the way for a more robust orchestration.

## Current Architecture

### Endpoint Flow
- **Primary Router**: `hitl` in `llm_backend/api/endpoints/hitl.py` (registered in `llm_backend/api/api.py`)
- **Key Endpoints** (all prefixed by your API prefix, typically `/api`):
  - `POST /api/hitl/run` — Start a HITL-enabled workflow
  - `GET /api/hitl/run/{run_id}/status` — Get run status and progress
  - `POST /api/hitl/run/{run_id}/approve` — Submit human approval/edit/reject
  - `POST /api/hitl/run/{run_id}/pause` — Pause a running workflow
  - `POST /api/hitl/run/{run_id}/resume` — Resume a paused or awaiting-human workflow
  - `DELETE /api/hitl/run/{run_id}` — Cancel a workflow
  - `GET /api/hitl/runs` — List runs
  - `GET /api/hitl/runs/active` — List active/resumable runs (filter by `user_id`, `session_id`)
  - `GET /api/hitl/run/{run_id}/state` — Retrieve full state for session resume
  - `GET /api/hitl/sessions/{session_id}/active` — Session-scoped active runs
  - `GET /api/hitl/approvals/pending` — Pending approvals for a user
  - `GET /api/hitl/providers` — Available providers and mapped tools

- **Orchestration**: `HITLOrchestrator` coordinates provider execution and checkpoints.

## Critical Persistence Requirements

For HITL runs to appear in the database, the orchestrator must follow this flow:

1. **Initial State Creation**: Call `orchestrator.start_run()` to save the initial QUEUED state to the database
2. **Checkpoint Persistence**: Use `websocket_bridge.request_human_approval()` instead of direct WebSocket sends
3. **State Transitions**: All status changes (RUNNING → AWAITING_HUMAN → COMPLETED) are persisted via the bridge

**⚠️ Important**: Direct WebSocket messaging (`_send_websocket_message`) bypasses database persistence. Always use the bridge methods (`request_human_approval`) for approval requests to ensure runs are saved to the `hitl_runs` table.
- **Persistence**: State is managed by `create_state_manager()` and used throughout the endpoints.
- **Session Overrides**: `HITLOrchestrator.start_run()` now merges any caller-provided `original_input` with the latest `RunInput.model_dump()` so explicit overrides like `session_id` or `user_id` persist in the initial database snapshot.

### Current Data Flow (Legacy)
```
RunInput → ReplicateTeam → information_agent → replicate_agent → api_interaction_agent → response_audit_agent → Result
```

### New Provider-Agnostic Data Flow
```
RunInput → HITLOrchestrator → AIProvider → HITL Checkpoints → Result
```

The orchestrator automatically gathers attachments by probing recent session history (see `_gather_attachments()` in `llm_backend/core/hitl/orchestrator.py`). If a required asset is missing, the payload validation step surfaces a blocking issue and pauses the run for human intervention.

**Attachment Resolution Fallback**: When AI agents fail or produce incomplete payloads, a schema-aware fallback cascade ensures user attachments are mapped to the correct field by checking the model's `example_input` schema. This prevents attachments from being dropped when fields are filtered by `_filter_payload_to_schema()`. See `docs/FORM_BASED_HITL.md` for detailed fallback behavior.

## Provider-Agnostic Architecture

The new architecture separates concerns into three layers:

### 1. Provider Layer
- **AIProvider Interface**: Abstract base class defining common operations
- **Provider Implementations**: ReplicateProvider, OpenAIProvider, AnthropicProvider, etc.
- **Provider Registry**: Maps tools to providers and manages instantiation

### 2. HITL Orchestration Layer  
- **HITLOrchestrator**: Manages workflow and checkpoints
- **State Management**: Persists run state across async operations
- **Event Streaming**: Real-time updates via Server-Sent Events

### 3. API Layer
- **Unified Endpoints**: Same API contract across all providers
- **Backward Compatibility**: Legacy endpoints maintained

## Testing & Verification

The HITL stack includes a dedicated regression suite to ensure resumability, persistence, and edge-case handling remain stable:

- `tests/test_hitl_session_resumability.py`
- `tests/test_hitl_database_integrity.py`
- `tests/test_hitl_edge_cases.py`

Execute them with Poetry:

```bash
poetry run pytest tests/test_hitl_session_resumability.py -v
poetry run pytest tests/test_hitl_database_integrity.py -v
poetry run pytest tests/test_hitl_edge_cases.py -v

# Aggregated report & database verification
poetry run python tests/run_hitl_tests.py
```

The orchestrated runner executes all suites and performs an additional resumability validation to confirm paused runs retain session/user metadata and checkpoint context.

## Proposed HITL Checkpoints

### 1. Information Review (Pre-run)
- **Trigger**: When `information_agent` detects ambiguity, safety concerns, or low confidence
- **Action**: Pause for human review of model selection and capability alignment
- **Human Options**: Approve, modify prompt, change model/tool selection

### 2. Payload Review (Pre-API call)
- **Trigger**: After `replicate_agent` produces `AgentPayload`
- **Action**: Present payload for human approval/editing
- **Features**: 
  - Highlight diffs from `example_input`
  - Show validation results from tool checks (e.g., `ModelRetry` for missing prompt/image)
  - Provide suggested fixes

### 3. Response Review (Post-run)
- **Trigger**: After `api_interaction_agent` completes
- **Action**: Show raw output vs `response_audit_agent` cleaned text side-by-side
- **Human Options**: Accept, edit, or request retry with adjustments

### 4. Error/Retry HITL
- **Trigger**: `ModelRetry` exceptions (e.g., missing prompt/image)
- **Action**: Convert exceptions into actionable tasks
- **Human Options**: Fix payload, override validation, or cancel

## REST API Reference (Current)

This section documents the currently implemented REST API.

### Start HITL Run
POST `/api/hitl/run`

Request body:
```json
{
  "run_input": { "prompt": "...", "agent_tool_config": { "REPLICATETOOL": { "data": { "model_name": "..." } } } },
  "hitl_config": null,
  "user_id": "user-123",
  "session_id": "sess-456"
}
```

Response:
```json
{
  "run_id": "<uuid>",
  "status": "queued",
  "message": "HITL run started successfully",
  "websocket_url": "wss://..." 
}
```

### Get Run Status
GET `/api/hitl/run/{run_id}/status`

Returns progress, current step, pending actions, and timestamps.

### Approve/Reject/Edit Checkpoint
POST `/api/hitl/run/{run_id}/approve`

Body:
```json
{
  "approval_id": "<token>",
  "action": "approve" | "edit" | "reject",
  "edits": {"prompt": "..."},
  "reason": "optional",
  "approved_by": "user-123"
}
```

### Pause / Resume / Cancel
POST `/api/hitl/run/{run_id}/pause`

POST `/api/hitl/run/{run_id}/resume`

DELETE `/api/hitl/run/{run_id}`

### List Runs
GET `/api/hitl/runs?user_id=&status=&limit=50`

### Session Resume Endpoints
- GET `/api/hitl/runs/active?user_id=&session_id=&status=` — resumable runs
- GET `/api/hitl/run/{run_id}/state` — full state for resume
- GET `/api/hitl/sessions/{session_id}/active?user_id=` — session-specific runs

### Approvals & Providers
- GET `/api/hitl/approvals/pending?user_id=`
- GET `/api/hitl/providers`

## State Machine

### States and Transitions
```
created → information_review? → payload_review → api_call → response_review? → completed
                    ↓                ↓                           ↓
                awaiting_human   awaiting_human           awaiting_human
                    ↓                ↓                           ↓
                 approve/edit     approve/edit             approve/edit
```

### State Behaviors
- **Auto-advance**: When policy allows and thresholds are met
- **Pause**: In `awaiting_human` when required or thresholds trigger
- **Resume**: On `approve`/`edit` actions
- **Failure**: Any failure leads to `failed` with reason; user may retry

## Integration Points in Code

### `llm_backend/api/endpoints/teams.py`
**Current**: `run_replicate_team()` directly calls `replicate_team.run()`

**Proposed Changes**:
- Orchestrate long-running or paused runs
- Return `202 Accepted` with `run_id` and `status="awaiting_human"` when gates are hit
- Support both synchronous (auto-mode) and asynchronous (HITL) execution

### `llm_backend/agents/replicate_team.py`
**Current**: Sequential agent execution in `ReplicateTeam.run()`

**Proposed Changes**:
- **After `information_agent`**: Compute `confidence`, `reasons`, `safety_flags`; pause if policy requires
- **After `replicate_agent`**: Expose `AgentPayload` to HITL gate for approval/edit
- **After `api_interaction_agent`**: Allow response review/edit selection
- **Standardize**: Internal `RunState` object to carry `current_step`, artifacts, and pause/continue logic

### Enhanced Agent Outputs

#### Information Agent
Add fields to `InformationInputResponse`:
```python
confidence: float  # 0-1 confidence score
reasons: List[str]  # Why continue or pause
safety_flags: List[str]  # NSFW, PII risk, copyright risk
recommended_model_settings: Dict  # Suggested overrides
```

#### Replicate Agent
Enhance `ModelRetry` handling:
- Convert exceptions to structured, user-facing tasks
- Provide suggested fixes (e.g., "Map image_file to input_image")
- Track validation results for HITL presentation

## Data Model and Persistence

### Hybrid PostgreSQL + JSONB Architecture

The HITL system now uses a hybrid approach combining relational columns with JSONB document storage for maximum flexibility:

#### HITLRun Model
```python
class HITLRun:
    # Relational columns for indexing and queries
    run_id: UUID (primary key)
    status: str (indexed)
    current_step: str
    provider_name: str
    session_id: str (indexed)
    user_id: str (indexed)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    
    # JSON columns for structured data
    original_input: JSON
    hitl_config: JSON
    capabilities: JSON
    suggested_payload: JSON
    validation_issues: JSON
    raw_response: JSON
    processed_response: Text
    final_result: Text
    pending_actions: JSON
    approval_token: str
    checkpoint_context: JSON
    last_approval: JSON
    
    # Metrics
    total_execution_time_ms: int
    human_review_time_ms: int
    provider_execution_time_ms: int
    
    # JSONB document snapshot (NEW)
    state_snapshot: JSONB  # Complete HITLState as document
```

#### JSONB Indexes for Performance
```sql
-- Status queries
CREATE INDEX hitl_runs_state_status_idx ON hitl_runs USING GIN ((state_snapshot->>'status'));

-- Checkpoint type queries  
CREATE INDEX hitl_runs_checkpoint_type_idx ON hitl_runs USING GIN ((state_snapshot->'checkpoint_context'->>'type'));

-- Pending actions queries
CREATE INDEX hitl_runs_pending_actions_idx ON hitl_runs USING GIN ((state_snapshot->'pending_actions'));

-- Validation issues queries
CREATE INDEX hitl_runs_validation_issues_idx ON hitl_runs USING GIN ((state_snapshot->'validation_issues'));

-- General JSONB queries
CREATE INDEX hitl_runs_state_snapshot_gin_idx ON hitl_runs USING GIN (state_snapshot);
```

#### Document-Style Query Methods
```python
# Query runs by checkpoint type
runs = await db_store.query_runs_by_checkpoint_type("payload_review", user_id="user123")

# Query runs with validation issues
runs = await db_store.query_runs_with_validation_issues(severity="error")

# Query runs with specific pending actions
runs = await db_store.query_runs_by_pending_action("approve")

# Get analytics using JSONB aggregations
analytics = await db_store.get_run_analytics()
```

### Benefits of Hybrid JSONB Approach

#### Advantages
- **Backward Compatibility**: Existing relational queries continue to work
- **Document Flexibility**: Complex nested data stored naturally in JSONB
- **Query Performance**: GIN indexes enable fast JSONB path queries
- **Analytics**: Native PostgreSQL aggregations on JSON data
- **Migration Safety**: Gradual transition without breaking existing code

#### Use Cases
- **Complex Validation Rules**: Store validation checkpoints as nested JSON
- **Dynamic Metadata**: Checkpoint context varies by provider/model
- **Audit Trails**: Complete state snapshots for debugging
- **Analytics**: Aggregate metrics across different checkpoint types

### Migration Strategy
1. **Phase 1**: Add `state_snapshot` JSONB column (✅ Completed)
2. **Phase 2**: Populate JSONB alongside existing columns (✅ Completed)  
3. **Phase 3**: Create GIN indexes for performance (✅ Completed)
4. **Phase 4**: Add document-style query methods (✅ Completed)
5. **Phase 5**: Gradually migrate queries to use JSONB where beneficial

### Versioning Strategy
- Complete `HITLState` snapshots stored in `state_snapshot` JSONB column
- Relational columns maintained for backward compatibility and indexing
- Version all human edits with timestamps and actor information in JSON

## Validation and Guidance

### Structured Error Handling
- Convert `ModelRetry` exceptions to actionable HITL tasks:
  - Missing prompt → "Add prompt to payload"
  - Missing image_file → "Attach image or map to correct field"
  - Malformed field → "Fix field format: expected X, got Y"

### Auto-suggestions
- Based on `example_input` schema analysis:
  - Suggest field mappings (e.g., `image_file` → `input_image`)
  - Provide default values for optional parameters
  - Highlight required vs optional fields

## User Experience

### Eventing and Streaming
- **Server-Sent Events** for real-time step updates
- **Event types**: `created`, `information_review`, `payload_review`, `api_call`, `response_review`, `completed`
- **Human-friendly messages** to guide reviewers at each step

### UI Workflow Recommendations

#### Information Review Panel
- Model summary and capability overview
- Confidence score and reasoning
- Safety flags and recommendations
- Actions: Proceed, Modify Prompt, Change Model

#### Payload Editor
- Side-by-side view: `example_input` vs proposed payload
- Inline validation with hints
- Diff highlighting for changed fields
- Actions: Approve, Edit Fields, Reset to Example

#### Response Review
- Raw API response vs audited/cleaned text
- Quality indicators and safety checks
- Actions: Accept, Edit Text, Retry with Changes

#### HITL Validation Card (Frontend)
- Renders in `HITLValidationCard.tsx` inside the chat thread when the orchestrator pauses a run
- Shows checkpoint metadata, required actions, and contextual validation issues
- **Auto-approve flow**: When `validation_summary.blocking_issues === 0`, the UI starts a 10-second countdown, displays "No feedback required" messaging, and auto-submits approval when the timer reaches zero.
- **Human override**: Reviewers can cancel the countdown, trigger immediate approval, or reject/cancel while providing a reason. Canceling stops the timer and surfaces a reminder to review manually.
- **Persistent acknowledgement**: After any action (approve, cancel, reject) the card remains visible and swaps to a green confirmation banner such as "Thank you for your feedback" instead of disappearing, preserving conversational context for the thread.

### Observability and Audit

### Logging Strategy
Log every step with:
- **Actor**: `system | human | agent_name`
- **Input/Output snapshots** (with sensitive value redaction)
- **Decision type**: `auto_approved | human_approved | human_edited`
- **Latency and diagnostics** for external calls

### Metrics and Monitoring
- **HITL rates** per model and user
- **Auto-pass rates** and threshold effectiveness
- **Average human latency** per review type
- **Common payload corrections** for model improvement
- **Alerts** for queue backups or threshold anomalies

## Safety, Privacy, and Governance

### Access Control
- **RBAC/Permissions**: Only authorized roles can approve runs, edit payloads, or switch models
- **Audit trail**: Complete history of who made what changes when

### Data Protection
- **Redaction**: Avoid storing secrets in payload snapshots
- **Retention**: Configurable data retention policies for runs and artifacts
- **Privacy**: PII detection and handling in prompts and responses

### Content Policies
- **Safety flags**: Surface content policy violations
- **Required approval**: Human review for flagged or gray-area content
- **Escalation**: Route sensitive content to appropriate reviewers

## Rollout Strategy

### Phase 1: Foundation
- Add run state management and single HITL checkpoint (payload review)
- Optional auto-bypass for testing
- Basic UI for payload approval/editing

### Phase 2: Full Gates
- Add information review (pre-run) and response review (post-run) gates
- Confidence thresholds and automated triggering
- Enhanced UI with diff views and validation

### Phase 3: Advanced Features
- SLAs and queue management
- Full event streaming and real-time updates
- Advanced analytics and reporting

### Phase 4: Intelligence
- Feedback collection and reinforcement learning
- Auto-adjustment of prompts and tool configs based on human edits
- Predictive confidence scoring

## Configuration Examples

### Auto Mode (Current Behavior)
```json
{
  "run_policy": "auto",
  "review_thresholds": null,
  "allowed_actions": []
}
```

### Conservative HITL
```json
{
  "run_policy": "require_human",
  "review_thresholds": null,
  "allowed_actions": ["information_review", "payload_review", "response_review"]
}
```

### Threshold-Based HITL
```json
{
  "run_policy": "auto_with_thresholds",
  "review_thresholds": {
    "confidence_min": 0.8,
    "safety_flags": ["nsfw", "pii"],
    "payload_changes_max": 3
  },
  "allowed_actions": ["payload_review", "response_review"]
}
```

## Implementation Checklist

- [ ] Define `RunState` and `RunStatus` enums
- [ ] Create `Run` entity and persistence layer
- [ ] Add HITL configuration to `RunInput`
- [ ] Implement async orchestration in `run_replicate_team()`
- [ ] Add pause/resume logic to `ReplicateTeam.run()`
- [ ] Create new HITL management endpoints
- [ ] Enhance agent outputs with confidence and safety data
- [ ] Build UI components for each review type
- [ ] Add SSE/WebSocket support for real-time updates
- [ ] Implement audit logging and metrics
- [ ] Create comprehensive tests for HITL flows
- [ ] Document API changes and migration guide