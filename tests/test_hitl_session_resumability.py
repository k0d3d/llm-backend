"""
Comprehensive tests for HITL session resumability and data persistence

This test suite focuses on:
1. Session resumability across different scenarios
2. Database state persistence and integrity
3. Image/file attachment handling during resume
4. Edge cases and error scenarios
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep, HITLState, StepEvent
from llm_backend.core.hitl.persistence import DatabaseStateStore
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge
from llm_backend.core.providers.base import (
    AIProvider,
    ProviderResponse,
    ProviderCapabilities,
    ValidationIssue,
    ProviderPayload,
    OperationType,
)
from llm_backend.core.types.common import RunInput


class MockImagePayload(ProviderPayload):
    """Simple payload structure for mock provider"""

    input_image: Optional[str] = None
    prompt: str = ""
    operation: str = ""


class MockImageProvider(AIProvider):
    """Mock provider that requires image input for testing attachment scenarios"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="mock-image-provider",
            description="Mock provider for image background removal",
            version="1.0",
            input_schema={
                "input_image": {
                    "type": "string",
                    "format": "uri",
                    "description": "Image URL or base64 data",
                }
            },
            supported_operations=[OperationType.IMAGE_EDITING],
            safety_features=["nsfw_filter"],
            rate_limits={"rpm": 60},
            cost_per_request=0.01,
            max_input_size=10_485_760,
            max_output_size=10_485_760,
        )

    def validate_payload(
        self,
        payload: ProviderPayload,
        prompt: str,
        attachments: List[str],
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        image: Optional[str] = None
        if isinstance(payload, MockImagePayload):
            image = payload.input_image
        else:
            # fallback for payloads represented as dictionaries
            image = getattr(payload, "input_image", None) or getattr(payload, "input", {}).get("input_image")

        if not image and not attachments:
            issues.append(
                ValidationIssue(
                    field="input_image",
                    issue="Required image attachment is missing",
                    severity="error",
                    suggested_fix="Upload an image file or provide an attachment",
                )
            )

        if prompt and len(prompt) < 5:
            issues.append(
                ValidationIssue(
                    field="prompt",
                    issue="Prompt is too short for meaningful processing",
                    severity="warning",
                    suggested_fix="Provide a more descriptive prompt",
                    auto_fixable=False,
                )
            )

        return issues

    def create_payload(
        self,
        prompt: str,
        attachments: List[str],
        operation_type: OperationType,
        config: Dict,
    ) -> ProviderPayload:
        image = attachments[0] if attachments else config.get("input_image")

        return MockImagePayload(
            provider_name="mock-image-provider",
            input_image=image,
            prompt=prompt,
            operation=operation_type.value if isinstance(operation_type, OperationType) else str(operation_type),
            metadata={
                "attachments": attachments,
                "config": config,
            },
        )

    def execute(self, payload: ProviderPayload, **kwargs) -> ProviderResponse:
        image = None
        if isinstance(payload, MockImagePayload):
            image = payload.input_image
        else:
            image = getattr(payload, "input_image", None)

        if not image:
            return ProviderResponse(
                raw_response={},
                processed_response="",
                metadata={"provider": "mock-image-provider"},
                execution_time_ms=5,
                error="Missing required input_image parameter",
            )

        return ProviderResponse(
            raw_response={
                "output_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "message": "Background removed successfully",
            },
            processed_response="Background removed successfully",
            metadata={"provider": "mock-image-provider"},
            execution_time_ms=12,
        )

    def audit_response(self, response: ProviderResponse, **kwargs) -> str:
        """Mock audit implementation"""
        return response.processed_response or ""

    def estimate_cost(self, payload: ProviderPayload, **kwargs) -> float:
        """Mock cost estimation"""
        return 0.01

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Mock rate limit status"""
        return {"remaining": 100, "reset_time": 3600}


@pytest.fixture
def test_database_url():
    """Test database URL - uses in-memory SQLite for testing"""
    return "sqlite:///:memory:"


@pytest.fixture
def database_state_store():
    """Database state store for testing"""
    store = DatabaseStateStore("sqlite:///:memory:")
    return store
    # Cleanup is handled by in-memory database


@pytest.fixture
def mock_websocket_bridge():
    """Mock WebSocket bridge with realistic behavior"""
    bridge = Mock(spec=WebSocketHITLBridge)
    bridge.request_human_approval = AsyncMock()
    bridge.send_status_update = AsyncMock()
    bridge.send_step_completion = AsyncMock()
    bridge.send_error_notification = AsyncMock()
    bridge.handle_approval_response = AsyncMock(return_value=True)
    return bridge


@pytest.fixture
def sample_run_input():
    """Sample run input for testing"""
    return RunInput(
        prompt="Remove the background from this image",
        user_email="test@example.com",
        user_id="test-user-" + str(uuid.uuid4()),
        agent_email="agent@example.com",
        session_id="test-session-" + str(uuid.uuid4()),
        message_type="USER_MESSAGE",
        agent_tool_config={
            "provider": "replicate",
            "model": "background-removal"
        }
    )


@pytest.fixture
def hitl_config_with_checkpoints():
    """HITL config that enables all checkpoints"""
    return HITLConfig(
        policy="require_human",
        timeout_seconds=3600,
        auto_approve_confidence_threshold=0.9,
        max_payload_changes=5
    )


class TestSessionResumability:
    """Test session pause and resume functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_session_resume_after_pause(self, database_state_store, sample_run_input, hitl_config_with_checkpoints, mock_websocket_bridge):
        """Test basic session resume after manual pause"""
        # Create orchestrator
        provider = MockImageProvider()
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=hitl_config_with_checkpoints,
            run_input=sample_run_input,
            state_manager=database_state_store,
            websocket_bridge=mock_websocket_bridge
        )
        
        # Start run with session context
        original_input = sample_run_input.model_dump()
        original_input.update({
            "session_id": "test-session-123",
            "user_id": "user-456"
        })
        
        run_id = await orchestrator.start_run(
            original_input=original_input,
            user_id="user-456",
            session_id="test-session-123"
        )
        
        # Verify initial state is saved
        initial_state = await database_state_store.load_state(run_id)
        assert initial_state is not None
        assert initial_state.run_id == run_id
        assert initial_state.status == HITLStatus.QUEUED
        assert initial_state.original_input["session_id"] == "test-session-123"
        assert initial_state.original_input["user_id"] == "user-456"
        
        # Simulate pause during validation checkpoint
        await database_state_store.pause_run(
            run_id=run_id,
            checkpoint_type="information_review",
            context={
                "message": "Image validation required",
                "validation_issues": [{"field": "input_image", "severity": "error"}],
                "confidence_score": 0.3
            }
        )
        
        # Verify paused state
        paused_state = await database_state_store.load_state(run_id)
        assert paused_state.status == HITLStatus.AWAITING_HUMAN
        assert paused_state.checkpoint_context is not None
        assert paused_state.checkpoint_context["type"] == "information_review"
        
        # Test session-based active runs query
        active_runs = await database_state_store.list_active_runs(session_id="test-session-123")
        assert len(active_runs) == 1
        assert active_runs[0]["run_id"] == run_id
        assert active_runs[0]["status"] == "awaiting_human"
        assert active_runs[0]["session_id"] == "test-session-123"
        
        # Resume with approval
        approval_response = {
            "action": "approve",
            "approved_by": "user-456",
            "timestamp": datetime.utcnow().isoformat()
        }
        resumed_state = await database_state_store.resume_run(run_id, approval_response)
        
        # Verify resumed state
        assert resumed_state.status == HITLStatus.RUNNING
        assert resumed_state.last_approval == approval_response
        
        print("✅ Basic session resume test passed")
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, database_state_store, mock_websocket_bridge, sample_run_input, hitl_config_with_checkpoints):
        """Test that multiple sessions are properly isolated"""
        provider = MockImageProvider()
        
        # Create runs for different sessions
        sessions = [
            {"session_id": "session-1", "user_id": "user-1"},
            {"session_id": "session-2", "user_id": "user-2"},
            {"session_id": "session-3", "user_id": "user-1"}  # Same user, different session
        ]
        
        run_ids = []
        for session in sessions:
            orchestrator = HITLOrchestrator(
                provider=provider,
                config=hitl_config_with_checkpoints,
                run_input=sample_run_input,
                state_manager=database_state_store,
                websocket_bridge=mock_websocket_bridge
            )

            original_input = sample_run_input.model_dump()
            original_input.update(session)
            
            run_id = await orchestrator.start_run(
                original_input=original_input,
                user_id=session["user_id"],
                session_id=session["session_id"]
            )
            run_ids.append(run_id)
            
            # Pause each run
            await database_state_store.pause_run(
                run_id=run_id,
                checkpoint_type="information_review",
                context={"session": session["session_id"]}
            )
        
        # Test session isolation
        session1_runs = await database_state_store.list_active_runs(session_id="session-1")
        session2_runs = await database_state_store.list_active_runs(session_id="session-2")
        session3_runs = await database_state_store.list_active_runs(session_id="session-3")
        
        assert len(session1_runs) == 1
        assert len(session2_runs) == 1
        assert len(session3_runs) == 1
        
        assert session1_runs[0]["session_id"] == "session-1"
        assert session2_runs[0]["session_id"] == "session-2"
        assert session3_runs[0]["session_id"] == "session-3"
        
        # Test user-based filtering (user-1 has 2 sessions)
        user1_runs = await database_state_store.list_active_runs(user_id="user-1")
        user2_runs = await database_state_store.list_active_runs(user_id="user-2")
        
        assert len(user1_runs) == 2  # session-1 and session-3
        assert len(user2_runs) == 1  # session-2
        
        print("✅ Multiple sessions isolation test passed")


class TestDatabaseStatePersistence:
    """Test database state persistence and integrity"""
    
    @pytest.mark.asyncio
    async def test_complete_state_persistence_cycle(self, database_state_store, sample_run_input, hitl_config_with_checkpoints):
        """Test complete state persistence through all workflow steps"""
        # Create initial state
        run_id = str(uuid.uuid4())
        initial_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=hitl_config_with_checkpoints,
            original_input={
                "prompt": "Test prompt",
                "session_id": "persistence-test-session",
                "user_id": "persistence-test-user"
            },
            capabilities={"model": "remove-bg", "requires_image": True},
            suggested_payload={"model": "remove-bg", "input": {"image": None}},
            validation_issues=[
                {"field": "input_image", "severity": "error", "issue": "Missing required image"}
            ],
            pending_actions=["upload_image", "approve"],
            approval_token="test-token-123"
        )
        
        # Add step events
        initial_state.step_history = [
            StepEvent(
                step=HITLStep.CREATED,
                status=HITLStatus.QUEUED,
                timestamp=datetime.utcnow(),
                actor="system",
                message="Run created"
            ),
            StepEvent(
                step=HITLStep.INFORMATION_REVIEW,
                status=HITLStatus.AWAITING_HUMAN,
                timestamp=datetime.utcnow(),
                actor="system",
                message="Validation checkpoint reached",
                metadata={"validation_issues": 1}
            )
        ]
        
        # Save state
        await database_state_store.save_state(initial_state)
        
        # Load and verify complete state
        loaded_state = await database_state_store.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state.run_id == run_id
        assert loaded_state.current_step == HITLStep.INFORMATION_REVIEW
        assert loaded_state.status == HITLStatus.AWAITING_HUMAN
        assert loaded_state.original_input["session_id"] == "persistence-test-session"
        assert loaded_state.capabilities["requires_image"] is True
        assert len(loaded_state.validation_issues) == 1
        assert loaded_state.approval_token == "test-token-123"
        assert len(loaded_state.step_history) == 2
        
        print("✅ Complete state persistence cycle test passed")


class TestAttachmentHandling:
    """Test image/file attachment handling in resume scenarios"""
    
    @pytest.mark.asyncio
    async def test_attachment_requirement_detection(self, database_state_store, mock_websocket_bridge):
        """Test detection of attachment requirements during validation"""
        provider = MockImageProvider()
        
        # Create run input without image
        run_input = RunInput(
            prompt="Remove background from image",
            user_email="user@example.com",
            user_id="attachment-user",
            agent_email="agent@example.com",
            session_id="attachment-session",
            message_type="USER_MESSAGE",
            agent_tool_config={
                "REPLICATETOOL": {
                    "data": {"model": "remove-bg"}
                }
            }
        )
        
        # Test payload creation and validation
        payload = provider.create_payload(
            prompt=run_input.prompt,
            attachments=[],
            operation_type=OperationType.IMAGE_EDITING,
            config=run_input.agent_tool_config
        )

        validation_issues = provider.validate_payload(payload, run_input.prompt, [])

        # Verify attachment requirement is detected
        assert len(validation_issues) > 0
        image_issue = next((issue for issue in validation_issues if "image" in issue.field.lower()), None)
        assert image_issue is not None
        assert image_issue.severity == "error"
        assert "missing" in image_issue.issue.lower()
        
        print("✅ Attachment requirement detection test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
