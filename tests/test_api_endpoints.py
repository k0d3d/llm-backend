"""
Tests for HITL API endpoints
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

from llm_backend.api.endpoints.hitl import router
from llm_backend.core.hitl.types import HITLConfig, HITLStatus
from llm_backend.core.types.common import RunInput, AgentTools


@pytest.fixture
def mock_dependencies():
    """Mock all dependencies for API tests"""
    with patch('llm_backend.api.endpoints.hitl.state_manager') as mock_state_manager, \
         patch('llm_backend.api.endpoints.hitl.websocket_bridge') as mock_websocket_bridge, \
         patch('llm_backend.core.providers.registry.ProviderRegistry') as mock_registry:
        
        # Setup mocks
        mock_state_manager.save_state = AsyncMock()
        mock_state_manager.load_state = AsyncMock()
        mock_state_manager.list_active_runs = AsyncMock()
        
        mock_websocket_bridge.handle_approval_response = AsyncMock()
        mock_websocket_bridge.list_pending_approvals = AsyncMock()
        mock_websocket_bridge.cancel_approval = AsyncMock()
        
        mock_registry._tool_mappings = {AgentTools.REPLICATETOOL: "replicate"}
        mock_registry._providers = {"replicate": Mock()}
        
        yield {
            "state_manager": mock_state_manager,
            "websocket_bridge": mock_websocket_bridge,
            "registry": mock_registry
        }


@pytest.fixture
def client():
    """Test client for API endpoints"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestHITLAPIEndpoints:
    """Test cases for HITL API endpoints"""
    
    def test_start_hitl_run_success(self, client, mock_dependencies):
        """Test successful HITL run start"""
        with patch('llm_backend.api.endpoints.hitl.HITLOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator.start_run = AsyncMock(return_value="test_run_id")
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Prepare request data
            run_input = {
                "prompt": "test prompt",
                "agent_tool_config": {
                    AgentTools.REPLICATETOOL: {"data": {"model": "test-model"}}
                },
                "document_url": None
            }
            
            request_data = {
                "run_input": run_input,
                "hitl_config": {
                    "require_human_approval": True,
                    "checkpoint_payload_suggestion": True
                },
                "user_id": "test_user",
                "session_id": "test_session"
            }
            
            # Make request
            response = client.post("/hitl/run", json=request_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == "test_run_id"
            assert data["status"] == "queued"
            assert "websocket_url" in data
    
    def test_start_hitl_run_no_provider(self, client, mock_dependencies):
        """Test HITL run start with no supported provider"""
        # Prepare request with unsupported tool
        run_input = {
            "prompt": "test prompt",
            "agent_tool_config": {
                "UNSUPPORTED_TOOL": {"data": {}}
            },
            "document_url": None
        }
        
        request_data = {
            "run_input": run_input
        }
        
        # Make request
        response = client.post("/hitl/run", json=request_data)
        
        # Verify error response
        assert response.status_code == 400
        assert "No supported provider found" in response.json()["detail"]
    
    def test_get_run_status_success(self, client, mock_dependencies):
        """Test successful run status retrieval"""
        # Mock state data
        mock_state = Mock()
        mock_state.run_id = "test_run_id"
        mock_state.status = HITLStatus.RUNNING
        mock_state.current_step = "payload_suggestion"
        mock_state.created_at = datetime.utcnow()
        mock_state.updated_at = datetime.utcnow()
        mock_state.expires_at = None
        mock_state.pending_actions = []
        mock_state.step_history = [
            Mock(step="capability_discovery", status="completed", timestamp=datetime.utcnow(), message="Done")
        ]
        
        mock_dependencies["state_manager"].load_state.return_value = mock_state
        
        # Make request
        response = client.get("/hitl/run/test_run_id/status")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test_run_id"
        assert data["status"] == HITLStatus.RUNNING
        assert data["current_step"] == "payload_suggestion"
        assert "progress" in data
    
    def test_get_run_status_not_found(self, client, mock_dependencies):
        """Test run status retrieval for non-existent run"""
        mock_dependencies["state_manager"].load_state.return_value = None
        
        # Make request
        response = client.get("/hitl/run/nonexistent/status")
        
        # Verify error response
        assert response.status_code == 404
        assert "Run not found" in response.json()["detail"]
    
    def test_approve_checkpoint_success(self, client, mock_dependencies):
        """Test successful checkpoint approval"""
        mock_dependencies["websocket_bridge"].handle_approval_response.return_value = True
        
        # Prepare approval data
        approval_data = {
            "approval_id": "test_approval",
            "action": "approve",
            "approved_by": "test_user"
        }
        
        # Make request
        response = client.post("/hitl/run/test_run_id/approve", json=approval_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] == "test_run_id"
        assert data["approval_id"] == "test_approval"
    
    def test_approve_checkpoint_invalid(self, client, mock_dependencies):
        """Test checkpoint approval with invalid approval ID"""
        mock_dependencies["websocket_bridge"].handle_approval_response.return_value = False
        
        # Prepare approval data
        approval_data = {
            "approval_id": "invalid_approval",
            "action": "approve"
        }
        
        # Make request
        response = client.post("/hitl/run/test_run_id/approve", json=approval_data)
        
        # Verify error response
        assert response.status_code == 400
        assert "Invalid approval ID" in response.json()["detail"]
    
    def test_pause_run_success(self, client, mock_dependencies):
        """Test successful run pause"""
        # Mock state data
        mock_state = Mock()
        mock_state.status = HITLStatus.RUNNING
        mock_dependencies["state_manager"].load_state.return_value = mock_state
        
        # Make request
        response = client.post("/hitl/run/test_run_id/pause")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] == "test_run_id"
    
    def test_pause_run_invalid_status(self, client, mock_dependencies):
        """Test pause run with invalid status"""
        # Mock state data
        mock_state = Mock()
        mock_state.status = HITLStatus.COMPLETED
        mock_dependencies["state_manager"].load_state.return_value = mock_state
        
        # Make request
        response = client.post("/hitl/run/test_run_id/pause")
        
        # Verify error response
        assert response.status_code == 400
        assert "Cannot pause run in status" in response.json()["detail"]
    
    def test_resume_run_success(self, client, mock_dependencies):
        """Test successful run resume"""
        # Mock state data
        mock_state = Mock()
        mock_state.status = HITLStatus.AWAITING_HUMAN
        mock_state.config = HITLConfig()
        mock_state.original_input = {"provider": "replicate"}
        mock_dependencies["state_manager"].load_state.return_value = mock_state
        
        with patch('llm_backend.api.endpoints.hitl.HITLOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Make request
            response = client.post("/hitl/run/test_run_id/resume")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["run_id"] == "test_run_id"
    
    def test_cancel_run_success(self, client, mock_dependencies):
        """Test successful run cancellation"""
        # Mock state data
        mock_state = Mock()
        mock_state.status = HITLStatus.RUNNING
        mock_dependencies["state_manager"].load_state.return_value = mock_state
        mock_dependencies["websocket_bridge"].list_pending_approvals.return_value = []
        
        # Make request
        response = client.delete("/hitl/run/test_run_id")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] == "test_run_id"
    
    def test_list_runs_success(self, client, mock_dependencies):
        """Test successful runs listing"""
        # Mock runs data
        mock_runs = [
            {
                "run_id": "run1",
                "status": "running",
                "current_step": "payload_suggestion",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "run_id": "run2",
                "status": "awaiting_human",
                "current_step": "validation_review",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        mock_dependencies["state_manager"].list_active_runs.return_value = mock_runs
        
        # Make request
        response = client.get("/hitl/runs")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 2
        assert data["total"] == 2
    
    def test_list_runs_with_filters(self, client, mock_dependencies):
        """Test runs listing with filters"""
        mock_runs = [
            {
                "run_id": "run1",
                "status": "running",
                "current_step": "payload_suggestion"
            }
        ]
        mock_dependencies["state_manager"].list_active_runs.return_value = mock_runs
        
        # Make request with filters
        response = client.get("/hitl/runs?user_id=test_user&status=running&limit=10")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 1
    
    def test_list_pending_approvals_success(self, client, mock_dependencies):
        """Test successful pending approvals listing"""
        # Mock approvals data
        mock_approvals = [
            {
                "approval_id": "approval1",
                "run_id": "run1",
                "checkpoint_type": "payload_approval",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        mock_dependencies["websocket_bridge"].list_pending_approvals.return_value = mock_approvals
        
        # Make request
        response = client.get("/hitl/approvals/pending")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["approvals"]) == 1
        assert data["total"] == 1
    
    def test_list_providers_success(self, client, mock_dependencies):
        """Test successful providers listing"""
        # Make request
        response = client.get("/hitl/providers")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "total" in data
    
    def test_handle_websocket_message_success(self, client, mock_dependencies):
        """Test successful WebSocket message handling"""
        with patch('llm_backend.api.endpoints.hitl.HITLWebSocketHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.handle_message = AsyncMock(return_value={"type": "test_response"})
            mock_handler_class.return_value = mock_handler
            
            # Prepare message data
            message_data = {
                "type": "hitl_approval_response",
                "data": {
                    "approval_id": "test_approval",
                    "action": "approve"
                }
            }
            
            # Make request
            response = client.post("/hitl/websocket/message", json=message_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "test_response"


if __name__ == "__main__":
    pytest.main([__file__])
