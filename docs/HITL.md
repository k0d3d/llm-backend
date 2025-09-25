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
- **Persistence**: State is managed by `create_state_manager()` and used throughout the endpoints.

### Current Data Flow (Legacy)
```
RunInput → ReplicateTeam → information_agent → replicate_agent → api_interaction_agent → response_audit_agent → Result
```

### New Provider-Agnostic Data Flow
```
RunInput → HITLOrchestrator → AIProvider → HITL Checkpoints → Result
```

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
- **Backward Compatibility**: Legacy endpoints maintained during migration

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

### Run Entity
```python
class Run:
    run_id: str
    status: RunStatus
    current_step: RunStep
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    
    # Input data
    original_input: RunInput
    tool_config: Dict
    
    # Step artifacts
    information_result: Optional[InformationInputResponse]
    suggested_payload: Optional[AgentPayload]
    api_response: Optional[str]
    final_result: Optional[str]
    
    # Human interactions
    approvals: List[Approval]
    edits: List[Edit]
    feedback: Optional[Feedback]
    
    # Audit trail
    step_history: List[StepEvent]
```

### Versioning Strategy
- Keep snapshots of `example_input`, `tool_config`, and `AgentPayload` per gate
- Store diffs to enable "undo" and provenance tracking
- Version all human edits with timestamps and actor information

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

## Observability and Audit

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