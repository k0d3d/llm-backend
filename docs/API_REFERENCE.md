# HITL API Reference

This document provides a comprehensive reference for the Human-in-the-Loop (HITL) API endpoints and data structures.

## Base URL

All endpoints are relative to your API base URL: `https://your-api.com/api`

## Authentication

All endpoints require authentication via Bearer token:
```
Authorization: Bearer <your-token>
```

## Core Endpoints

### Create Run

Create a new HITL run with a specific provider.

**Endpoint:** `POST /hitl/runs`

**Request Body:**
```json
{
  "prompt": "string",
  "user_email": "string",
  "user_id": "string", 
  "agent_email": "string",
  "session_id": "string",
  "message_type": "string",
  "log_id": "string?",
  "agent_tool_config": {
    "replicate-agent-tool": {
      "data": {
        "name": "string",
        "description": "string",
        "example_input": {},
        "latest_version": "string"
      }
    }
  },
  "hitl_config": {
    "policy": "auto | require_human | auto_with_thresholds",
    "review_thresholds": {
      "confidence_min": 0.8,
      "safety_flags": ["nsfw", "pii", "copyright"],
      "payload_changes_max": 3
    },
    "allowed_steps": ["information_review", "payload_review", "response_review"],
    "timeout_seconds": 3600
  }
}
```

Note: File/image attachments are not provided directly in this request. If the selected model requires assets, the backend will attempt to auto-discover recent attachments from the user's chat history for the given `session_id`. If none are found, the HITL flow will pause with a validation checkpoint requesting the asset.

**Response:**
```json
{
  "run_id": "uuid",
  "status": "completed | awaiting_human | running | failed",
  "current_step": "information_review | payload_review | response_review | completed",
  "result": "string?",
  "message": "string?",
  "actions_required": ["approve", "edit", "reject"]?,
  "suggested_payload": {}?,
  "validation_issues": []?,
  "events_url": "/hitl/runs/{run_id}/events",
  "expires_at": "ISO timestamp?"
}
```

### Get Run Status

Get the current status and state of a run.

**Endpoint:** `GET /hitl/runs/{run_id}`

**Response:**
```json
{
  "run_id": "uuid",
  "status": "queued | awaiting_human | running | completed | failed | cancelled",
  "current_step": "created | information_review | payload_review | api_call | response_review | completed",
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp", 
  "expires_at": "ISO timestamp?",
  "pending_actions": ["approve", "edit", "reject"],
  "step_history": [
    {
      "step": "information_review",
      "status": "completed",
      "timestamp": "ISO timestamp",
      "actor": "system | human | agent_name",
      "message": "string?"
    }
  ],
  "metadata": {
    "total_execution_time_ms": 1500,
    "human_review_time_ms": 30000,
    "provider_execution_time_ms": 800
  }
}
```

### Approve Current Step

Approve the current step and continue execution.

**Endpoint:** `POST /hitl/runs/{run_id}/approve`

**Request Body:**
```json
{
  "actor": "user@example.com",
  "message": "Looks good to proceed"
}
```

**Response:**
```json
{
  "run_id": "uuid",
  "status": "completed | awaiting_human | running",
  "current_step": "string",
  "result": "string?",
  "message": "string"
}
```

### Edit Current Step

Edit the current step data and continue execution.

**Endpoint:** `POST /hitl/runs/{run_id}/edit`

**Request Body:**
```json
{
  "actor": "user@example.com",
  "message": "Updated prompt for clarity",
  "edits": {
    "prompt": "Modified prompt text",
    "payload": {
      "width": 1024,
      "height": 1024
    },
    "response": "Edited response text"
  }
}
```

**Response:**
```json
{
  "run_id": "uuid", 
  "status": "completed | awaiting_human | running",
  "current_step": "string",
  "result": "string?",
  "message": "string"
}
```

### Reject Current Step

Reject the current step and cancel the run.

**Endpoint:** `POST /hitl/runs/{run_id}/reject`

**Request Body:**
```json
{
  "actor": "user@example.com",
  "reason": "Content violates policy"
}
```

**Response:**
```json
{
  "run_id": "uuid",
  "status": "cancelled",
  "reason": "string",
  "cancelled_at_step": "information_review | payload_review | response_review"
}
```

### Stream Events

Stream real-time events for a run via Server-Sent Events.

**Endpoint:** `GET /hitl/runs/{run_id}/events`

**Response:** `text/event-stream`

**Event Types:**
```
data: {"type": "step_started", "step": "payload_review", "timestamp": "..."}

data: {"type": "awaiting_human", "step": "payload_review", "actions_required": ["approve", "edit"]}

data: {"type": "step_completed", "step": "payload_review", "actor": "user@example.com"}

data: {"type": "run_completed", "result": "Generated image URL", "metadata": {...}}

data: {"type": "heartbeat"}
```

## Legacy Endpoints (Backward Compatibility)

### Teams Run (Legacy)

**Endpoint:** `POST /teams/run`

**Query Parameters:**
- `use_hitl=true` - Enable HITL orchestrator (optional, default: false)

**Request/Response:** Same as original ReplicateTeam format

### Teams Run HITL

**Endpoint:** `POST /teams/run-hitl`

Explicit HITL version of the teams endpoint.

## Data Structures

### HITLConfig

```json
{
  "policy": "auto | require_human | auto_with_thresholds",
  "review_thresholds": {
    "confidence_min": 0.8,
    "safety_flags": ["nsfw", "pii", "copyright"],
    "payload_changes_max": 3,
    "response_quality_min": 0.7
  },
  "allowed_steps": ["information_review", "payload_review", "response_review"],
  "timeout_seconds": 3600,
  "enable_streaming": false
}
```

### ValidationIssue

```json
{
  "field": "input.prompt",
  "issue": "Prompt not found in payload",
  "severity": "error | warning | info",
  "suggested_fix": "Add prompt to input.prompt field",
  "auto_fixable": true
}
```

### ProviderCapabilities

```json
{
  "name": "SDXL Image Generator",
  "description": "Generate high-quality images from text prompts",
  "version": "v1.0",
  "input_schema": {
    "prompt": "string",
    "width": "integer",
    "height": "integer"
  },
  "supported_operations": ["image_generation"],
  "safety_features": ["content_filter", "nsfw_detection"],
  "rate_limits": {
    "requests_per_minute": 60
  },
  "cost_per_request": 0.01
}
```

### StepEvent

```json
{
  "step": "payload_review",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "actor": "system | user@example.com | agent_name",
  "message": "Auto-approved based on thresholds",
  "metadata": {
    "execution_time_ms": 150,
    "confidence_score": 0.95
  }
}
```

## Error Responses

All endpoints return standard HTTP status codes with error details:

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "Invalid request data",
  "details": {
    "field": "hitl_config.policy",
    "issue": "Must be one of: auto, require_human, auto_with_thresholds"
  }
}
```

### 404 Not Found
```json
{
  "error": "run_not_found", 
  "message": "Run with ID 'abc123' not found"
}
```

### 409 Conflict
```json
{
  "error": "invalid_state",
  "message": "Cannot approve run in current state",
  "current_state": "completed"
}
```

### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded for provider",
  "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
  "error": "provider_error",
  "message": "Provider execution failed",
  "details": "Connection timeout to Replicate API"
}
```

## Rate Limits

- **Run Creation**: 100 requests per minute per user
- **Status Checks**: 1000 requests per minute per user  
- **Approvals/Edits**: 200 requests per minute per user
- **Event Streaming**: 10 concurrent connections per user

## Webhooks

Configure webhooks to receive run status updates:

**Webhook Payload:**
```json
{
  "run_id": "uuid",
  "event_type": "run_completed | run_failed | awaiting_human",
  "timestamp": "ISO timestamp",
  "data": {
    "status": "completed",
    "result": "Generated content",
    "metadata": {}
  }
}
```

**Webhook Headers:**
```
X-HITL-Signature: sha256=<signature>
X-HITL-Event: run_completed
Content-Type: application/json
```

## SDKs and Examples

### Python SDK

```python
from hitl_client import HITLClient

client = HITLClient(api_key="your-api-key")

# Create run
run = client.create_run(
    prompt="Generate a sunset image",
    provider="replicate",
    hitl_config={
        "policy": "auto_with_thresholds",
        "review_thresholds": {"confidence_min": 0.8}
    }
)

# Check status
status = client.get_run_status(run.run_id)

# Approve if needed
if status.status == "awaiting_human":
    result = client.approve_run(run.run_id, "user@example.com", "Approved")
```

### JavaScript SDK

```javascript
import { HITLClient } from '@your-org/hitl-client';

const client = new HITLClient({ apiKey: 'your-api-key' });

// Create run
const run = await client.createRun({
  prompt: 'Generate a sunset image',
  provider: 'replicate',
  hitlConfig: {
    policy: 'auto_with_thresholds',
    reviewThresholds: { confidenceMin: 0.8 }
  }
});

// Stream events
const eventStream = client.streamEvents(run.runId);
eventStream.on('awaiting_human', (event) => {
  console.log('Human review required:', event);
});
```

### cURL Examples

```bash
# Create run
curl -X POST https://api.example.com/hitl/runs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a sunset image",
    "user_email": "user@example.com",
    "user_id": "user123",
    "agent_tool_config": {...}
  }'

# Get status
curl https://api.example.com/hitl/runs/run-id-123 \
  -H "Authorization: Bearer $API_KEY"

# Approve step
curl -X POST https://api.example.com/hitl/runs/run-id-123/approve \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"actor": "user@example.com", "message": "Approved"}'
```
