"""
Tests for HITL orchestrator functionality
"""

import pytest
from unittest.mock import Mock, AsyncMock

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep
from llm_backend.core.hitl.persistence import HybridStateManager
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge
from llm_backend.core.providers.base import AIProvider, ProviderResponse


class MockProvider(AIProvider):
    """Mock provider for testing"""

    def __init__(self, config: dict = None):
        super().__init__(config or {})

    async def get_capabilities(self, input_data: dict) -> dict:
        return {
            "supported_models": ["test-model"],
            "max_tokens": 1000,
            "supports_streaming": True
        }

    async def create_payload(self, prompt: str, attachments: list, operation_type, config: dict, hitl_edits: dict = None) -> dict:
        return {
            "model": "test-model",
            "prompt": prompt,
            "max_tokens": 100
        }

    async def validate_payload(self, payload: dict) -> dict:
        issues = []
        if not payload.get("model"):
            issues.append({"field": "model", "issue": "Model is required", "severity": "error"})
        if not payload.get("prompt"):
            issues.append({"field": "prompt", "issue": "Prompt is required", "severity": "error"})
        return {"blocking_issues": len(issues), "issues": issues}

    async def execute(self, payload: dict) -> ProviderResponse:
        return ProviderResponse(
            raw_response={"output": "Test response"},
            processed_response="Test response",
            metadata={"execution_time": 1.5},
            execution_time_ms=1500
        )

    async def audit_response(self, response: ProviderResponse) -> dict:
        return {"quality_score": 0.95, "safety_flags": []}


@pytest.fixture
def mock_state_manager():
    """Mock state manager"""
    manager = Mock(spec=HybridStateManager)
    manager.save_state = AsyncMock()
    manager.load_state = AsyncMock()
    manager.delete_state = AsyncMock()
    return manager


@pytest.fixture
def mock_websocket_bridge():
    """Mock WebSocket bridge"""
    bridge = Mock(spec=WebSocketHITLBridge)
    bridge.request_human_approval = AsyncMock()
    bridge.send_status_update = AsyncMock()
    bridge.send_step_completion = AsyncMock()
    bridge.send_error_notification = AsyncMock()
    return bridge


@pytest.fixture
def hitl_config():
    """Default HITL configuration"""
    return HITLConfig(
        timeout_seconds=300
    )


@pytest.fixture
def orchestrator(mock_state_manager, mock_websocket_bridge, hitl_config):
    """HITL orchestrator with mocked dependencies"""
    mock_provider = MockProvider(config={})

    # Create minimal run_input with all required fields
    from llm_backend.core.types.common import RunInput
    run_input = RunInput(
        prompt="test prompt",
        user_email="test@example.com",
        user_id="test-user",
        agent_email="agent@example.com",
        session_id="test-session",
        message_type="user_message",
        agent_tool_config={"tool": "test"}
    )

    return HITLOrchestrator(
        provider=mock_provider,
        config=hitl_config,
        run_input=run_input,
        state_manager=mock_state_manager,
        websocket_bridge=mock_websocket_bridge
    )


class TestHITLOrchestrator:
    """Test cases for HITL orchestrator"""

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.run_id is not None
        assert orchestrator.state is not None
        assert orchestrator.state.status == HITLStatus.QUEUED
        assert orchestrator.state.current_step == HITLStep.CREATED
        assert orchestrator.provider is not None
        assert orchestrator.config is not None

    def test_state_manager_attached(self, orchestrator, mock_state_manager):
        """Test that state manager is properly attached"""
        assert orchestrator.state_manager is mock_state_manager

    def test_websocket_bridge_attached(self, orchestrator, mock_websocket_bridge):
        """Test that websocket bridge is properly attached"""
        assert orchestrator.websocket_bridge is mock_websocket_bridge

    def test_provider_attached(self, orchestrator):
        """Test that provider is properly attached"""
        assert isinstance(orchestrator.provider, MockProvider)
        assert orchestrator.provider.run_input is not None
