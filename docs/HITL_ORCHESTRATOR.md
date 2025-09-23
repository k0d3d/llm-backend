# HITL Orchestrator Design

This document details the design and implementation of the `HITLOrchestrator` class, which manages the human-in-the-loop workflow for AI provider executions.

## Core Concepts

The orchestrator acts as a state machine that coordinates between AI providers and human reviewers, ensuring appropriate checkpoints are enforced based on configuration and runtime conditions.

## Class Structure

### HITLOrchestrator

```python
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid

class HITLPolicy(str, Enum):
    AUTO = "auto"
    REQUIRE_HUMAN = "require_human"
    AUTO_WITH_THRESHOLDS = "auto_with_thresholds"

class HITLStep(str, Enum):
    CREATED = "created"
    INFORMATION_REVIEW = "information_review"
    PAYLOAD_REVIEW = "payload_review"
    API_CALL = "api_call"
    RESPONSE_REVIEW = "response_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class HITLStatus(str, Enum):
    QUEUED = "queued"
    AWAITING_HUMAN = "awaiting_human"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class HITLConfig(BaseModel):
    policy: HITLPolicy
    review_thresholds: Optional[Dict[str, Any]] = None
    allowed_steps: List[HITLStep] = []
    timeout_seconds: int = 3600
    auto_approve_confidence_threshold: float = 0.9
    require_approval_safety_flags: List[str] = ["nsfw", "pii", "copyright"]
    max_payload_changes: int = 5
    enable_streaming: bool = False

class StepEvent(BaseModel):
    step: HITLStep
    status: HITLStatus
    timestamp: datetime
    actor: str  # "system", "human", "agent_name"
    message: Optional[str] = None
    metadata: Dict[str, Any] = {}

class HITLState(BaseModel):
    run_id: str
    current_step: HITLStep
    status: HITLStatus
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    
    # Configuration
    config: HITLConfig
    original_input: Dict[str, Any]
    
    # Step artifacts
    capabilities: Optional[Dict[str, Any]] = None
    suggested_payload: Optional[Dict[str, Any]] = None
    validation_issues: List[Dict[str, Any]] = []
    raw_response: Optional[Any] = None
    processed_response: Optional[str] = None
    final_result: Optional[str] = None
    
    # Human interactions
    pending_actions: List[str] = []
    approval_token: Optional[str] = None
    
    # Audit trail
    step_history: List[StepEvent] = []
    
    # Metrics
    total_execution_time_ms: int = 0
    human_review_time_ms: int = 0
    provider_execution_time_ms: int = 0

class HITLOrchestrator:
    def __init__(self, provider: AIProvider, config: HITLConfig, run_input: RunInput):
        self.provider = provider
        self.config = config
        self.run_input = run_input
        self.run_id = str(uuid.uuid4())
        
        # Initialize state
        self.state = HITLState(
            run_id=self.run_id,
            current_step=HITLStep.CREATED,
            status=HITLStatus.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            config=config,
            original_input=run_input.dict()
        )
        
        # Set expiration if configured
        if config.timeout_seconds > 0:
            self.state.expires_at = datetime.utcnow() + timedelta(seconds=config.timeout_seconds)
        
        self._add_step_event(HITLStep.CREATED, HITLStatus.QUEUED, "system", "Run created")
    
    async def execute(self) -> Dict[str, Any]:
        """Main execution flow with HITL checkpoints"""
        start_time = time.time()
        
        try:
            # Step 1: Information Review
            result = await self._step_information_review()
            if self._is_paused(result):
                return result
            
            # Step 2: Payload Review
            result = await self._step_payload_review()
            if self._is_paused(result):
                return result
            
            # Step 3: API Execution
            result = await self._step_api_execution()
            if self._is_paused(result):
                return result
            
            # Step 4: Response Review
            result = await self._step_response_review()
            if self._is_paused(result):
                return result
            
            # Step 5: Completion
            return await self._step_completion()
            
        except Exception as e:
            return await self._handle_error(e)
        finally:
            self.state.total_execution_time_ms = int((time.time() - start_time) * 1000)
    
    async def _step_information_review(self) -> Dict[str, Any]:
        """Information review checkpoint"""
        self._transition_to_step(HITLStep.INFORMATION_REVIEW)
        
        # Get provider capabilities
        capabilities = self.provider.get_capabilities()
        self.state.capabilities = capabilities.dict()
        
        # Check if human review is required
        if self._should_pause_at_information_review(capabilities):
            return self._create_pause_response(
                step=HITLStep.INFORMATION_REVIEW,
                message="Model capabilities require human review",
                actions_required=["approve", "edit_prompt", "change_model"],
                data={
                    "capabilities": capabilities.dict(),
                    "confidence_score": self._calculate_information_confidence(capabilities)
                }
            )
        
        self._add_step_event(HITLStep.INFORMATION_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved based on thresholds")
        return {"continue": True}
    
    async def _step_payload_review(self) -> Dict[str, Any]:
        """Payload review checkpoint"""
        self._transition_to_step(HITLStep.PAYLOAD_REVIEW)
        
        # Create payload
        operation_type = self._infer_operation_type()
        payload = self.provider.create_payload(
            prompt=self.run_input.prompt,
            attachments=[self.run_input.document_url] if self.run_input.document_url else [],
            operation_type=operation_type,
            config=self.run_input.agent_tool_config or {}
        )
        
        # Validate payload
        validation_issues = self.provider.validate_payload(
            payload,
            self.run_input.prompt,
            [self.run_input.document_url] if self.run_input.document_url else []
        )
        
        self.state.suggested_payload = payload.dict()
        self.state.validation_issues = [issue.dict() for issue in validation_issues]
        
        # Check if human review is required
        if self._should_pause_at_payload_review(payload, validation_issues):
            return self._create_pause_response(
                step=HITLStep.PAYLOAD_REVIEW,
                message="Payload requires human review",
                actions_required=["approve", "edit_payload", "fix_validation_issues"],
                data={
                    "suggested_payload": payload.dict(),
                    "validation_issues": [issue.dict() for issue in validation_issues],
                    "diff_from_example": self._calculate_payload_diff(payload),
                    "estimated_cost": self.provider.estimate_cost(payload)
                }
            )
        
        # Auto-fix validation issues if possible
        if validation_issues:
            payload = self._auto_fix_payload(payload, validation_issues)
            self.state.suggested_payload = payload.dict()
        
        self._add_step_event(HITLStep.PAYLOAD_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved payload")
        return {"continue": True, "payload": payload}
    
    async def _step_api_execution(self) -> Dict[str, Any]:
        """API execution step"""
        self._transition_to_step(HITLStep.API_CALL, HITLStatus.RUNNING)
        
        # Get payload from previous step or state
        payload_dict = self.state.suggested_payload
        payload = self.provider.__class__.payload_class(**payload_dict)
        
        # Execute with provider
        start_time = time.time()
        response = self.provider.execute(payload)
        execution_time = int((time.time() - start_time) * 1000)
        
        self.state.provider_execution_time_ms = execution_time
        self.state.raw_response = response.raw_response
        self.state.processed_response = response.processed_response
        
        if response.error:
            raise Exception(f"Provider execution failed: {response.error}")
        
        self._add_step_event(HITLStep.API_CALL, HITLStatus.COMPLETED, "system", f"Executed in {execution_time}ms")
        return {"continue": True, "response": response}
    
    async def _step_response_review(self) -> Dict[str, Any]:
        """Response review checkpoint"""
        self._transition_to_step(HITLStep.RESPONSE_REVIEW)
        
        # Audit response
        response_dict = {"raw_response": self.state.raw_response, "processed_response": self.state.processed_response}
        response = ProviderResponse(**response_dict)
        audited_response = self.provider.audit_response(response)
        
        # Check if human review is required
        if self._should_pause_at_response_review(response, audited_response):
            return self._create_pause_response(
                step=HITLStep.RESPONSE_REVIEW,
                message="Response requires human review",
                actions_required=["approve", "edit_response", "retry"],
                data={
                    "raw_response": response.raw_response,
                    "processed_response": response.processed_response,
                    "audited_response": audited_response,
                    "quality_score": self._calculate_response_quality(response)
                }
            )
        
        self.state.final_result = audited_response
        self._add_step_event(HITLStep.RESPONSE_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved response")
        return {"continue": True, "final_result": audited_response}
    
    async def _step_completion(self) -> Dict[str, Any]:
        """Completion step"""
        self._transition_to_step(HITLStep.COMPLETED, HITLStatus.COMPLETED)
        
        return {
            "run_id": self.run_id,
            "status": "completed",
            "result": self.state.final_result,
            "metadata": {
                "total_execution_time_ms": self.state.total_execution_time_ms,
                "provider_execution_time_ms": self.state.provider_execution_time_ms,
                "human_review_time_ms": self.state.human_review_time_ms,
                "step_count": len(self.state.step_history)
            }
        }
    
    async def approve_current_step(self, actor: str, message: Optional[str] = None) -> Dict[str, Any]:
        """Approve the current step and continue execution"""
        if self.state.status != HITLStatus.AWAITING_HUMAN:
            raise ValueError("No pending human action")
        
        self._add_step_event(self.state.current_step, HITLStatus.COMPLETED, actor, message or "Approved")
        
        # Continue execution from current step
        return await self.execute()
    
    async def edit_current_step(self, actor: str, edits: Dict[str, Any], message: Optional[str] = None) -> Dict[str, Any]:
        """Apply edits to current step and continue"""
        if self.state.status != HITLStatus.AWAITING_HUMAN:
            raise ValueError("No pending human action")
        
        # Apply edits based on current step
        if self.state.current_step == HITLStep.INFORMATION_REVIEW:
            if "prompt" in edits:
                self.run_input.prompt = edits["prompt"]
            if "model_config" in edits:
                self.run_input.agent_tool_config.update(edits["model_config"])
        
        elif self.state.current_step == HITLStep.PAYLOAD_REVIEW:
            if "payload" in edits:
                self.state.suggested_payload.update(edits["payload"])
        
        elif self.state.current_step == HITLStep.RESPONSE_REVIEW:
            if "response" in edits:
                self.state.final_result = edits["response"]
        
        self._add_step_event(self.state.current_step, HITLStatus.COMPLETED, actor, f"Edited: {message}")
        
        # Continue execution
        return await self.execute()
    
    async def reject_current_step(self, actor: str, reason: str) -> Dict[str, Any]:
        """Reject current step and cancel run"""
        self._transition_to_step(self.state.current_step, HITLStatus.CANCELLED)
        self._add_step_event(self.state.current_step, HITLStatus.CANCELLED, actor, f"Rejected: {reason}")
        
        return {
            "run_id": self.run_id,
            "status": "cancelled",
            "reason": reason,
            "cancelled_at_step": self.state.current_step
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current run state"""
        return {
            "run_id": self.run_id,
            "status": self.state.status,
            "current_step": self.state.current_step,
            "pending_actions": self.state.pending_actions,
            "created_at": self.state.created_at.isoformat(),
            "updated_at": self.state.updated_at.isoformat(),
            "expires_at": self.state.expires_at.isoformat() if self.state.expires_at else None,
            "step_history": [event.dict() for event in self.state.step_history],
            "metadata": {
                "total_execution_time_ms": self.state.total_execution_time_ms,
                "human_review_time_ms": self.state.human_review_time_ms
            }
        }
    
    # Helper methods
    def _should_pause_at_information_review(self, capabilities: ProviderCapabilities) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.INFORMATION_REVIEW in self.config.allowed_steps
        
        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            confidence = self._calculate_information_confidence(capabilities)
            return confidence < self.config.auto_approve_confidence_threshold
        
        return False
    
    def _should_pause_at_payload_review(self, payload: ProviderPayload, issues: List[ValidationIssue]) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.PAYLOAD_REVIEW in self.config.allowed_steps
        
        # Always pause for errors
        if any(issue.severity == "error" for issue in issues):
            return True
        
        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            # Check payload changes threshold
            changes = self._count_payload_changes(payload)
            if changes > self.config.max_payload_changes:
                return True
        
        return False
    
    def _should_pause_at_response_review(self, response: ProviderResponse, audited: str) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.RESPONSE_REVIEW in self.config.allowed_steps
        
        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            quality = self._calculate_response_quality(response)
            threshold = self.config.review_thresholds.get("response_quality_min", 0.7)
            return quality < threshold
        
        return False
    
    def _create_pause_response(self, step: HITLStep, message: str, actions_required: List[str], data: Dict[str, Any] = None) -> Dict[str, Any]:
        self.state.status = HITLStatus.AWAITING_HUMAN
        self.state.pending_actions = actions_required
        self.state.approval_token = str(uuid.uuid4())
        self.state.updated_at = datetime.utcnow()
        
        response = {
            "run_id": self.run_id,
            "status": "awaiting_human",
            "current_step": step,
            "message": message,
            "actions_required": actions_required,
            "approval_token": self.state.approval_token,
            "expires_at": self.state.expires_at.isoformat() if self.state.expires_at else None,
            "events_url": f"/teams/runs/{self.run_id}/events"
        }
        
        if data:
            response.update(data)
        
        return response
    
    def _transition_to_step(self, step: HITLStep, status: HITLStatus = HITLStatus.RUNNING):
        self.state.current_step = step
        self.state.status = status
        self.state.updated_at = datetime.utcnow()
    
    def _add_step_event(self, step: HITLStep, status: HITLStatus, actor: str, message: str):
        event = StepEvent(
            step=step,
            status=status,
            timestamp=datetime.utcnow(),
            actor=actor,
            message=message
        )
        self.state.step_history.append(event)
    
    def _is_paused(self, result: Dict[str, Any]) -> bool:
        return result.get("status") == "awaiting_human"
    
    async def _handle_error(self, error: Exception) -> Dict[str, Any]:
        self._transition_to_step(self.state.current_step, HITLStatus.FAILED)
        self._add_step_event(self.state.current_step, HITLStatus.FAILED, "system", str(error))
        
        return {
            "run_id": self.run_id,
            "status": "failed",
            "error": str(error),
            "failed_at_step": self.state.current_step
        }
```

## State Persistence

The orchestrator state should be persisted to handle async operations and system restarts:

```python
# llm_backend/core/hitl/persistence.py
from abc import ABC, abstractmethod
from typing import Optional

class HITLStateStore(ABC):
    @abstractmethod
    async def save_state(self, state: HITLState) -> None:
        pass
    
    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        pass
    
    @abstractmethod
    async def delete_state(self, run_id: str) -> None:
        pass

class RedisStateStore(HITLStateStore):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def save_state(self, state: HITLState) -> None:
        key = f"hitl:state:{state.run_id}"
        await self.redis.setex(key, 3600, state.json())
    
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        key = f"hitl:state:{run_id}"
        data = await self.redis.get(key)
        return HITLState.parse_raw(data) if data else None
    
    async def delete_state(self, run_id: str) -> None:
        key = f"hitl:state:{run_id}"
        await self.redis.delete(key)
```

## Event Streaming

Support real-time updates via Server-Sent Events:

```python
# llm_backend/core/hitl/events.py
from fastapi import Request
from fastapi.responses import StreamingResponse
import asyncio
import json

class HITLEventStream:
    def __init__(self):
        self.subscribers = {}
    
    async def subscribe(self, run_id: str, request: Request):
        queue = asyncio.Queue()
        self.subscribers[run_id] = self.subscribers.get(run_id, [])
        self.subscribers[run_id].append(queue)
        
        async def event_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield "data: {\"type\": \"heartbeat\"}\n\n"
                        
            finally:
                self.subscribers[run_id].remove(queue)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    async def publish_event(self, run_id: str, event: Dict[str, Any]):
        if run_id in self.subscribers:
            for queue in self.subscribers[run_id]:
                await queue.put(event)
```

## Usage Examples

### Basic Usage
```python
orchestrator = HITLOrchestrator(provider, config, run_input)
result = await orchestrator.execute()
```

### With Human Interaction
```python
# Initial execution
result = await orchestrator.execute()

if result["status"] == "awaiting_human":
    # Human reviews and approves
    result = await orchestrator.approve_current_step("user@example.com", "Looks good")
```

### With Edits
```python
# Human makes edits
result = await orchestrator.edit_current_step(
    "user@example.com",
    {"prompt": "Modified prompt"},
    "Updated prompt for clarity"
)
```
