# Migration Guide: From ReplicateTeam to Provider-Agnostic HITL

This guide outlines the step-by-step migration from the current `ReplicateTeam` implementation to the new provider-agnostic HITL architecture.

## Migration Overview

The migration involves:
1. Creating the new provider interface and orchestrator
2. Implementing `ReplicateProvider` wrapper
3. Updating API endpoints to use the orchestrator
4. Maintaining backward compatibility during transition
5. Deprecating old implementation

## Phase 1: Foundation Setup

### 1.1 Create Base Provider Interface

Create the abstract base classes:

```bash
mkdir -p src/llm_backend/core/providers
mkdir -p src/llm_backend/core/hitl
mkdir -p src/llm_backend/providers
```

Files to create:
- `src/llm_backend/core/providers/__init__.py`
- `src/llm_backend/core/providers/base.py` (from PROVIDER_INTERFACE.md)
- `src/llm_backend/core/hitl/__init__.py`
- `src/llm_backend/core/hitl/orchestrator.py` (from HITL_ORCHESTRATOR.md)
- `src/llm_backend/core/hitl/persistence.py`
- `src/llm_backend/core/hitl/events.py`

### 1.2 Update Type Definitions

Extend `src/llm_backend/core/types/common.py`:

```python
# Add to existing file
class HITLConfig(BaseModel):
    policy: str = "auto"  # auto, require_human, auto_with_thresholds
    review_thresholds: Optional[Dict[str, Any]] = None
    allowed_steps: List[str] = []
    timeout_seconds: int = 3600

# Update RunInput to include HITL config
class RunInput(BaseModel):
    # ... existing fields ...
    hitl_config: Optional[HITLConfig] = None
```

### 1.3 Provider Registry

Create `src/llm_backend/core/providers/registry.py`:

```python
from typing import Dict, Type, Optional
from llm_backend.core.providers.base import AIProvider
from llm_backend.core.types.common import AgentTools

class ProviderRegistry:
    _providers: Dict[str, Type[AIProvider]] = {}
    _tool_mapping: Dict[AgentTools, str] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[AIProvider], tool_enum: Optional[AgentTools] = None):
        cls._providers[name] = provider_class
        if tool_enum:
            cls._tool_mapping[tool_enum] = name
    
    @classmethod
    def get_provider_for_tool(cls, tool: AgentTools, config: Dict) -> AIProvider:
        provider_name = cls._tool_mapping.get(tool)
        if not provider_name:
            raise ValueError(f"No provider registered for tool: {tool}")
        return cls.get_provider(provider_name, config)
    
    @classmethod
    def get_provider(cls, name: str, config: Dict) -> AIProvider:
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name](config)
```

## Phase 2: Replicate Provider Implementation

### 2.1 Create ReplicateProvider

Create `src/llm_backend/providers/replicate_provider.py`:

```python
import time
import re
from typing import Dict, Any, List
from llm_backend.core.providers.base import (
    AIProvider, ProviderPayload, ProviderResponse, 
    ProviderCapabilities, ValidationIssue, OperationType
)
from llm_backend.core.types.replicate import AgentPayload
from llm_backend.tools.replicate_tool import run_replicate

class ReplicatePayload(ProviderPayload):
    input: Dict[str, Any]
    operation_type: OperationType
    model_version: str

class ReplicateProvider(AIProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.example_input = config.get("example_input", {})
        self.description = config.get("description", "")
        self.latest_version = config.get("latest_version", "")
        self.model_name = config.get("name", "")
        self.run_input = None  # Will be set by orchestrator
    
    def set_run_input(self, run_input):
        """Set run input for provider execution"""
        self.run_input = run_input
    
    def get_capabilities(self) -> ProviderCapabilities:
        operation_type = self._infer_operation_type()
        
        return ProviderCapabilities(
            name=self.model_name,
            description=self.description,
            version=self.latest_version,
            input_schema=self.example_input,
            supported_operations=[operation_type],
            safety_features=["content_filter"],
            rate_limits={"requests_per_minute": 60}
        )
    
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict) -> ReplicatePayload:
        input_data = self.example_input.copy()
        
        # Map prompt to appropriate field
        prompt_fields = ["prompt", "text", "input", "query"]
        for field in prompt_fields:
            if field in input_data:
                input_data[field] = prompt
                break
        
        # Map attachments
        if attachments:
            image_fields = ["image", "image_url", "input_image", "first_frame_image"]
            for field in image_fields:
                if field in input_data and attachments:
                    input_data[field] = attachments[0]
                    break
        
        return ReplicatePayload(
            provider_name="replicate",
            input=input_data,
            operation_type=operation_type,
            model_version=self.latest_version
        )
    
    def validate_payload(self, payload: ReplicatePayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        issues = []
        
        # Check prompt presence
        if not any(prompt in str(value) for value in payload.input.values()):
            issues.append(ValidationIssue(
                field="input",
                issue="Prompt not found in payload",
                severity="error",
                suggested_fix=f"Add prompt to appropriate field",
                auto_fixable=True
            ))
        
        # Check image presence if required
        if attachments:
            image_found = any(att in str(payload.input) for att in attachments)
            if not image_found:
                issues.append(ValidationIssue(
                    field="input",
                    issue="Required image not found",
                    severity="error",
                    suggested_fix="Map image to appropriate field",
                    auto_fixable=True
                ))
        
        return issues
    
    def execute(self, payload: ReplicatePayload) -> ProviderResponse:
        start_time = time.time()
        
        try:
            run, status_code = run_replicate(
                run_input=self.run_input,
                model_params={
                    "example_input": self.example_input,
                    "latest_version": self.latest_version,
                },
                input=payload.input,
                operation_type=payload.operation_type.value,
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ProviderResponse(
                raw_response=run,
                processed_response=str(run),
                metadata={"model_version": payload.model_version},
                execution_time_ms=execution_time,
                status_code=status_code
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ProviderResponse(
                raw_response=None,
                processed_response="",
                metadata={},
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def audit_response(self, response: ProviderResponse) -> str:
        if response.error:
            return f"Error: {response.error}"
        
        cleaned = str(response.processed_response)
        cleaned = re.sub(r'https?://replicate\.com[^\s]*', '', cleaned)
        return cleaned.strip()
    
    def estimate_cost(self, payload: ReplicatePayload) -> float:
        return 0.01  # Basic estimate
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        return {"requests_remaining": 100, "reset_time": time.time() + 3600}
    
    def _infer_operation_type(self) -> OperationType:
        description_lower = self.description.lower()
        
        if "image" in description_lower:
            return OperationType.IMAGE_GENERATION
        elif "video" in description_lower:
            return OperationType.VIDEO_GENERATION
        elif "audio" in description_lower:
            return OperationType.AUDIO_GENERATION
        else:
            return OperationType.TEXT_GENERATION
```

### 2.2 Register Replicate Provider

Update `src/llm_backend/providers/__init__.py`:

```python
from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.providers.replicate_provider import ReplicateProvider
from llm_backend.core.types.common import AgentTools

# Register providers
ProviderRegistry.register("replicate", ReplicateProvider, AgentTools.REPLICATETOOL)
```

## Phase 3: API Endpoint Updates

### 3.1 Create New HITL Endpoints

Create `src/llm_backend/api/endpoints/hitl.py`:

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any

from llm_backend.core.hitl.orchestrator import HITLOrchestrator, HITLConfig
from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.core.types.common import RunInput, AgentTools

router = APIRouter()

# In-memory store for demo (replace with persistent storage)
active_runs: Dict[str, HITLOrchestrator] = {}

@router.post("/runs")
async def create_run(run_input: RunInput):
    """Create a new HITL run"""
    # Get provider
    agent_tool_config = run_input.agent_tool_config
    tool_type = None
    
    for tool in AgentTools:
        if tool.value in agent_tool_config:
            tool_type = tool
            break
    
    if not tool_type:
        raise HTTPException(400, "No supported tool found in agent_tool_config")
    
    provider = ProviderRegistry.get_provider_for_tool(
        tool_type, 
        agent_tool_config[tool_type.value].get("data", {})
    )
    provider.set_run_input(run_input)
    
    # Create orchestrator
    hitl_config = run_input.hitl_config or HITLConfig()
    orchestrator = HITLOrchestrator(provider, hitl_config, run_input)
    
    # Store for later access
    active_runs[orchestrator.run_id] = orchestrator
    
    # Start execution
    result = await orchestrator.execute()
    
    return result

@router.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    """Get current run status"""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    
    orchestrator = active_runs[run_id]
    return orchestrator.get_current_state()

@router.post("/runs/{run_id}/approve")
async def approve_run(run_id: str, approval: Dict[str, Any]):
    """Approve current step"""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    
    orchestrator = active_runs[run_id]
    result = await orchestrator.approve_current_step(
        approval.get("actor", "unknown"),
        approval.get("message")
    )
    
    return result

@router.post("/runs/{run_id}/edit")
async def edit_run(run_id: str, edit_request: Dict[str, Any]):
    """Edit current step and continue"""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    
    orchestrator = active_runs[run_id]
    result = await orchestrator.edit_current_step(
        edit_request.get("actor", "unknown"),
        edit_request.get("edits", {}),
        edit_request.get("message")
    )
    
    return result

@router.post("/runs/{run_id}/reject")
async def reject_run(run_id: str, rejection: Dict[str, Any]):
    """Reject current step and cancel run"""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    
    orchestrator = active_runs[run_id]
    result = await orchestrator.reject_current_step(
        rejection.get("actor", "unknown"),
        rejection.get("reason", "No reason provided")
    )
    
    return result

@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request):
    """Stream run events via SSE"""
    if run_id not in active_runs:
        raise HTTPException(404, "Run not found")
    
    # Implementation would use HITLEventStream
    # For now, return basic SSE
    async def event_generator():
        yield "data: {\"type\": \"connected\"}\n\n"
        # Would stream real events here
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### 3.2 Update Teams Endpoint

Modify `src/llm_backend/api/endpoints/teams.py`:

```python
from fastapi import APIRouter
from typing import Optional

from llm_backend.agents.replicate_team import ReplicateTeam  # Keep for backward compatibility
from llm_backend.core.types.common import AgentTools, RunInput, HITLConfig
from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.providers.registry import ProviderRegistry

router = APIRouter()

@router.post("/run")
async def run_replicate_team(run_input: RunInput, use_hitl: bool = False):
    """
    Run replicate team with optional HITL support
    
    Args:
        run_input: The run input data
        use_hitl: Whether to use new HITL orchestrator (default: False for backward compatibility)
    """
    
    if use_hitl:
        # New HITL path
        agent_tool_config = run_input.agent_tool_config
        replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)
        
        # Get provider
        provider = ProviderRegistry.get_provider(
            "replicate", 
            replicate_agent_tool_config.get("data", {})
        )
        provider.set_run_input(run_input)
        
        # Create orchestrator with default auto config
        hitl_config = run_input.hitl_config or HITLConfig(policy="auto")
        orchestrator = HITLOrchestrator(provider, hitl_config, run_input)
        
        # Execute
        result = await orchestrator.execute()
        return result
    
    else:
        # Legacy path - keep existing behavior
        agent_tool_config = run_input.agent_tool_config
        replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)

        replicate_team = ReplicateTeam(
            prompt=run_input.prompt,
            tool_config=replicate_agent_tool_config.get("data", {}),
            run_input=run_input,
        )

        return replicate_team.run()

# New endpoint for explicit HITL usage
@router.post("/run-hitl")
async def run_replicate_team_hitl(run_input: RunInput):
    """Run replicate team with HITL orchestrator"""
    return await run_replicate_team(run_input, use_hitl=True)
```

### 3.3 Update API Router

Modify `src/llm_backend/api/api.py`:

```python
from fastapi import APIRouter

from llm_backend.api.endpoints import teams, hitl

api_router = APIRouter()

api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(hitl.router, prefix="/hitl", tags=["hitl"])
```

## Phase 4: Testing and Validation

### 4.1 Create Tests

Create `tests/test_hitl_migration.py`:

```python
import pytest
from llm_backend.core.types.common import RunInput, HITLConfig, AgentTools
from llm_backend.providers.replicate_provider import ReplicateProvider
from llm_backend.core.hitl.orchestrator import HITLOrchestrator

@pytest.fixture
def sample_run_input():
    return RunInput(
        prompt="Test prompt",
        user_email="test@example.com",
        user_id="test_user",
        agent_email="agent@example.com",
        session_id="test_session",
        message_type="test",
        agent_tool_config={
            AgentTools.REPLICATETOOL: {
                "data": {
                    "name": "test-model",
                    "description": "Test model for image generation",
                    "example_input": {"prompt": "test", "width": 512, "height": 512},
                    "latest_version": "v1.0"
                }
            }
        }
    )

def test_replicate_provider_creation(sample_run_input):
    config = sample_run_input.agent_tool_config[AgentTools.REPLICATETOOL]["data"]
    provider = ReplicateProvider(config)
    
    capabilities = provider.get_capabilities()
    assert capabilities.name == "test-model"
    assert "image" in capabilities.description.lower()

def test_orchestrator_auto_mode(sample_run_input):
    config = sample_run_input.agent_tool_config[AgentTools.REPLICATETOOL]["data"]
    provider = ReplicateProvider(config)
    provider.set_run_input(sample_run_input)
    
    hitl_config = HITLConfig(policy="auto")
    orchestrator = HITLOrchestrator(provider, hitl_config, sample_run_input)
    
    assert orchestrator.run_id is not None
    assert orchestrator.state.current_step == "created"

# Add more tests for different scenarios
```

### 4.2 Integration Testing

Test both old and new paths:

```python
# Test legacy endpoint
response = client.post("/teams/run", json=run_input.dict())
assert response.status_code == 200

# Test new HITL endpoint
response = client.post("/teams/run-hitl", json=run_input.dict())
assert response.status_code == 200

# Test explicit HITL endpoints
response = client.post("/hitl/runs", json=run_input.dict())
assert response.status_code == 200
```

## Phase 5: Deployment and Rollout

### 5.1 Feature Flags

Add feature flag support:

```python
# In environment or config
ENABLE_HITL_ORCHESTRATOR = os.getenv("ENABLE_HITL_ORCHESTRATOR", "false").lower() == "true"

# In endpoint
if ENABLE_HITL_ORCHESTRATOR:
    # Use new orchestrator
else:
    # Use legacy ReplicateTeam
```

### 5.2 Gradual Migration

1. **Week 1**: Deploy with feature flag disabled, test in staging
2. **Week 2**: Enable for internal testing with `use_hitl=True` parameter
3. **Week 3**: Enable for subset of users via feature flag
4. **Week 4**: Enable for all users, monitor performance
5. **Week 5**: Deprecate legacy endpoint, update documentation

### 5.3 Monitoring

Add metrics for:
- HITL orchestrator usage vs legacy
- Performance comparison
- Error rates
- Human intervention rates

## Phase 6: Cleanup

### 6.1 Deprecation Notices

Add deprecation warnings to old ReplicateTeam:

```python
import warnings

class ReplicateTeam:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ReplicateTeam is deprecated. Use HITLOrchestrator with ReplicateProvider instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing code
```

### 6.2 Remove Legacy Code

After successful migration:
1. Remove `ReplicateTeam` class
2. Remove legacy endpoint paths
3. Update all documentation
4. Remove feature flags

## Rollback Plan

If issues arise:
1. Disable feature flag immediately
2. Route all traffic to legacy endpoint
3. Investigate and fix issues
4. Re-enable gradually

## Benefits After Migration

1. **Provider Flexibility**: Easy to add OpenAI, Anthropic, etc.
2. **Consistent HITL**: Same workflow across all providers
3. **Better Testing**: Mock providers for unit tests
4. **Cleaner Code**: Separation of concerns
5. **Scalability**: Async execution with state management
