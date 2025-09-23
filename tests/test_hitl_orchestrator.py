"""
Tests for HITL orchestrator functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep
from llm_backend.core.hitl.persistence import HybridStateManager
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge
from llm_backend.core.providers.base import AIProvider, ProviderResponse


class MockProvider(AIProvider):
    """Mock provider for testing"""
    
    async def get_capabilities(self, input_data: dict) -> dict:
        return {
            "supported_models": ["test-model"],
            "max_tokens": 1000,
            "supports_streaming": True
        }
    
    async def suggest_payload(self, input_data: dict, capabilities: dict) -> dict:
        return {
            "model": "test-model",
            "prompt": input_data.get("prompt", "test prompt"),
            "max_tokens": 100
        }
    
    async def validate_payload(self, payload: dict) -> list:
        issues = []
        if not payload.get("model"):
            issues.append("Model is required")
        if not payload.get("prompt"):
            issues.append("Prompt is required")
        return issues
    
    async def execute(self, payload: dict) -> ProviderResponse:
        return ProviderResponse(
            success=True,
            raw_response={"output": "Test response"},
            processed_response="Test response",
            metadata={"execution_time": 1.5}
        )


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
        require_human_approval=True,
        checkpoint_payload_suggestion=True,
        checkpoint_validation_review=True,
        checkpoint_response_review=True,
        approval_timeout_seconds=300
    )


@pytest.fixture
def orchestrator(mock_state_manager, mock_websocket_bridge, hitl_config):
    """HITL orchestrator with mocked dependencies"""
    with patch('llm_backend.core.providers.registry.ProviderRegistry.get_provider') as mock_get_provider:
        mock_get_provider.return_value = MockProvider()
        
        return HITLOrchestrator(
            provider_name="test",
            config=hitl_config,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )


class TestHITLOrchestrator:
    """Test cases for HITL orchestrator"""
    
    @pytest.mark.asyncio
    async def test_start_run(self, orchestrator, mock_state_manager):
        """Test starting a new HITL run"""
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        
        run_id = await orchestrator.start_run(input_data)
        
        assert run_id is not None
        assert len(run_id) > 0
        mock_state_manager.save_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_run_with_approvals(self, orchestrator, mock_state_manager, mock_websocket_bridge):
        """Test executing a run with human approvals"""
        # Setup
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Mock approval responses
        mock_websocket_bridge.request_human_approval.return_value = {
            "action": "approve",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Execute run
        result = await orchestrator.execute_run(run_id)
        
        # Verify
        assert result is not None
        assert mock_websocket_bridge.request_human_approval.call_count >= 2  # At least payload and validation checkpoints
        mock_state_manager.save_state.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_run_with_edits(self, orchestrator, mock_state_manager, mock_websocket_bridge):
        """Test executing a run with human edits"""
        # Setup
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Mock approval with edits
        mock_websocket_bridge.request_human_approval.return_value = {
            "action": "edit",
            "edits": {"max_tokens": 200},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Execute run
        result = await orchestrator.execute_run(run_id)
        
        # Verify
        assert result is not None
        mock_websocket_bridge.request_human_approval.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_run_with_rejection(self, orchestrator, mock_state_manager, mock_websocket_bridge):
        """Test executing a run with human rejection"""
        # Setup
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Mock rejection
        mock_websocket_bridge.request_human_approval.return_value = {
            "action": "reject",
            "reason": "Invalid request",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Execute run
        with pytest.raises(Exception, match="rejected"):
            await orchestrator.execute_run(run_id)
    
    @pytest.mark.asyncio
    async def test_capability_discovery_step(self, orchestrator):
        """Test capability discovery step"""
        input_data = {"prompt": "test prompt"}
        
        capabilities = await orchestrator._step_capability_discovery(input_data)
        
        assert capabilities is not None
        assert "supported_models" in capabilities
        assert "max_tokens" in capabilities
    
    @pytest.mark.asyncio
    async def test_payload_suggestion_step(self, orchestrator):
        """Test payload suggestion step"""
        input_data = {"prompt": "test prompt"}
        capabilities = {"supported_models": ["test-model"]}
        
        payload = await orchestrator._step_payload_suggestion(input_data, capabilities)
        
        assert payload is not None
        assert "model" in payload
        assert "prompt" in payload
    
    @pytest.mark.asyncio
    async def test_payload_validation_step(self, orchestrator):
        """Test payload validation step"""
        valid_payload = {"model": "test-model", "prompt": "test prompt"}
        invalid_payload = {"model": "", "prompt": ""}
        
        # Test valid payload
        issues = await orchestrator._step_payload_validation(valid_payload)
        assert len(issues) == 0
        
        # Test invalid payload
        issues = await orchestrator._step_payload_validation(invalid_payload)
        assert len(issues) > 0
    
    @pytest.mark.asyncio
    async def test_provider_execution_step(self, orchestrator):
        """Test provider execution step"""
        payload = {"model": "test-model", "prompt": "test prompt"}
        
        response = await orchestrator._step_provider_execution(payload)
        
        assert response is not None
        assert response.success is True
        assert response.processed_response == "Test response"
    
    @pytest.mark.asyncio
    async def test_response_processing_step(self, orchestrator):
        """Test response processing step"""
        raw_response = {"output": "Test response"}
        
        processed = await orchestrator._step_response_processing(raw_response)
        
        assert processed is not None
        assert isinstance(processed, str)
    
    @pytest.mark.asyncio
    async def test_pause_and_resume(self, orchestrator, mock_state_manager):
        """Test pausing and resuming a run"""
        # Start run
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Pause run
        await orchestrator.pause_run(run_id, "manual_pause", {"reason": "user request"})
        
        # Verify state was saved
        mock_state_manager.save_state.assert_called()
        
        # Resume run
        await orchestrator.resume_run(run_id, {"action": "resume"})
        
        # Verify state was updated
        assert mock_state_manager.save_state.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator, mock_state_manager, mock_websocket_bridge):
        """Test error handling during execution"""
        # Setup
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Mock provider error
        with patch.object(orchestrator.provider, 'execute', side_effect=Exception("Provider error")):
            with pytest.raises(Exception, match="Provider error"):
                await orchestrator.execute_run(run_id)
        
        # Verify error notification was sent
        mock_websocket_bridge.send_error_notification.assert_called()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator, mock_websocket_bridge):
        """Test timeout handling for approvals"""
        # Setup
        input_data = {"prompt": "test prompt", "user_id": "test_user"}
        run_id = await orchestrator.start_run(input_data)
        
        # Mock timeout
        mock_websocket_bridge.request_human_approval.side_effect = asyncio.TimeoutError("Approval timeout")
        
        # Execute run
        with pytest.raises(Exception, match="timeout"):
            await orchestrator.execute_run(run_id)
    
    @pytest.mark.asyncio
    async def test_no_approval_required(self, mock_state_manager, mock_websocket_bridge):
        """Test execution without human approval"""
        # Create config without approval requirements
        config = HITLConfig(require_human_approval=False)
        
        with patch('llm_backend.core.providers.registry.ProviderRegistry.get_provider') as mock_get_provider:
            mock_get_provider.return_value = MockProvider()
            
            orchestrator = HITLOrchestrator(
                provider_name="test",
                config=config,
                state_manager=mock_state_manager,
                websocket_bridge=mock_websocket_bridge
            )
            
            # Execute run
            input_data = {"prompt": "test prompt"}
            run_id = await orchestrator.start_run(input_data)
            result = await orchestrator.execute_run(run_id)
            
            # Verify no approval was requested
            mock_websocket_bridge.request_human_approval.assert_not_called()
            assert result is not None


@pytest.mark.asyncio
async def test_orchestrator_integration():
    """Integration test for the full orchestrator workflow"""
    
    # This test would require actual database and Redis connections
    # For now, we'll skip it in unit tests
    pytest.skip("Integration test requires actual database and Redis connections")


if __name__ == "__main__":
    pytest.main([__file__])
