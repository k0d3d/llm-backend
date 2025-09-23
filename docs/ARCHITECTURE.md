# Provider-Agnostic HITL Architecture

This document outlines the new provider-agnostic Human-in-the-Loop (HITL) architecture that decouples HITL workflow management from specific AI providers like Replicate.

## Overview

The architecture separates concerns into three main layers:

1. **Provider Layer**: AI provider implementations (Replicate, OpenAI, Anthropic, etc.)
2. **HITL Orchestration Layer**: Generic workflow management with human checkpoints
3. **API Layer**: HTTP endpoints and request/response handling

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
├─────────────────────────────────────────────────────────────┤
│  POST /teams/run                                           │
│  POST /teams/runs                                          │
│  GET /teams/runs/{id}                                      │
│  POST /teams/runs/{id}/approve                             │
│  POST /teams/runs/{id}/edit                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                 HITL Orchestration Layer                   │
├─────────────────────────────────────────────────────────────┤
│  HITLOrchestrator                                          │
│  ├─ Information Review Checkpoint                          │
│  ├─ Payload Review Checkpoint                              │
│  ├─ Response Review Checkpoint                             │
│  └─ Error/Retry Handling                                   │
│                                                            │
│  RunState Management                                       │
│  ├─ State Persistence                                      │
│  ├─ Step Transitions                                       │
│  └─ Human Approval Workflows                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                   Provider Layer                           │
├─────────────────────────────────────────────────────────────┤
│  AIProvider (Abstract Base)                                │
│  ├─ get_capabilities()                                     │
│  ├─ create_payload()                                       │
│  ├─ validate_payload()                                     │
│  ├─ execute()                                              │
│  └─ audit_response()                                       │
│                                                            │
│  Concrete Implementations:                                 │
│  ├─ ReplicateProvider                                      │
│  ├─ OpenAIProvider                                         │
│  ├─ AnthropicProvider                                      │
│  └─ HuggingFaceProvider                                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### AIProvider Interface

The `AIProvider` abstract base class defines the contract that all AI providers must implement:

- **Capabilities Discovery**: `get_capabilities()` returns provider metadata
- **Payload Creation**: `create_payload()` converts generic inputs to provider-specific format
- **Validation**: `validate_payload()` checks payload correctness and returns issues
- **Execution**: `execute()` calls the provider API
- **Response Processing**: `audit_response()` cleans/formats the response

### HITLOrchestrator

The orchestrator manages the workflow and human checkpoints:

- **Step Management**: Tracks current step and state transitions
- **Checkpoint Logic**: Determines when to pause for human review
- **State Persistence**: Maintains run state across async operations
- **Human Interaction**: Handles approvals, edits, and rejections

### Provider Implementations

Each provider implements the `AIProvider` interface:

- **ReplicateProvider**: Wraps existing Replicate functionality
- **OpenAIProvider**: For GPT models and DALL-E
- **AnthropicProvider**: For Claude models
- **HuggingFaceProvider**: For open-source models

## Data Flow

### Synchronous Flow (Auto Mode)
```
RunInput → HITLOrchestrator → Provider → Result
```

### Asynchronous Flow (HITL Mode)
```
RunInput → HITLOrchestrator → Checkpoint → Human Review → Continue → Provider → Result
```

### Detailed Step Flow
1. **Request Received**: API endpoint receives `RunInput` and `HITLConfig`
2. **Provider Selection**: Based on `AgentTools` enum, appropriate provider is instantiated
3. **Orchestrator Creation**: `HITLOrchestrator` is created with provider and config
4. **Information Review**: Provider capabilities are analyzed, human review if needed
5. **Payload Creation**: Provider creates payload from generic inputs
6. **Payload Review**: Validation results presented, human review if needed
7. **Execution**: Provider executes the request
8. **Response Review**: Raw and processed responses shown, human review if needed
9. **Completion**: Final result returned or sent to callback URL

## Configuration

### HITL Policies
- **AUTO**: No human intervention, fully automated
- **REQUIRE_HUMAN**: All steps require human approval
- **AUTO_WITH_THRESHOLDS**: Conditional pausing based on confidence/safety thresholds

### Review Thresholds
```json
{
  "confidence_min": 0.8,
  "safety_flags": ["nsfw", "pii", "copyright"],
  "payload_changes_max": 3,
  "response_quality_min": 0.7
}
```

### Allowed Steps
Configure which checkpoints are enabled:
- `information_review`
- `payload_review` 
- `response_review`

## State Management

### Run States
- `created`: Initial state
- `information_review`: Waiting for capability/model approval
- `payload_review`: Waiting for payload approval
- `api_call`: Provider execution in progress
- `response_review`: Waiting for response approval
- `completed`: Finished successfully
- `failed`: Error occurred
- `cancelled`: Human cancelled the run

### State Persistence
- **Run Entity**: Core run information and current state
- **Step History**: Audit trail of all state transitions
- **Artifacts**: Intermediate results (payloads, responses, etc.)
- **Human Actions**: Record of all approvals, edits, rejections

## Benefits

### Flexibility
- Easy to add new AI providers
- Consistent HITL experience across all providers
- Provider-specific optimizations without affecting workflow

### Maintainability
- Clear separation of concerns
- Reusable HITL logic
- Standardized provider interface

### Scalability
- Async execution with state persistence
- Horizontal scaling of provider implementations
- Queue management for human review tasks

### Testing
- Mock providers for testing HITL flows
- Unit testing of individual components
- Integration testing with real providers

## Migration Path

1. **Phase 1**: Create base interfaces and orchestrator
2. **Phase 2**: Implement ReplicateProvider wrapping existing logic
3. **Phase 3**: Update API endpoints to use orchestrator
4. **Phase 4**: Add additional providers (OpenAI, Anthropic, etc.)
5. **Phase 5**: Deprecate old ReplicateTeam direct usage

## Extension Points

### Custom Providers
Implement `AIProvider` interface for new services:
- Custom model endpoints
- Internal AI services
- Specialized processing pipelines

### Custom Checkpoints
Extend orchestrator with additional review points:
- Security scanning
- Cost approval
- Content moderation
- Compliance checks

### Custom Workflows
Create specialized orchestrators for different use cases:
- Batch processing
- Multi-step pipelines
- A/B testing workflows
