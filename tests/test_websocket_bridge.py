"""
Tests for WebSocket HITL bridge functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import aiohttp

from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge, HITLWebSocketHandler, BrowserAgentHITLIntegration
from llm_backend.core.hitl.types import HITLStatus
from llm_backend.core.hitl.persistence import HybridStateManager


@pytest.fixture
def mock_state_manager():
    """Mock state manager"""
    manager = Mock(spec=HybridStateManager)
    manager.pause_run = AsyncMock()
    manager.resume_run = AsyncMock()
    return manager


@pytest.fixture
def websocket_bridge(mock_state_manager):
    """WebSocket bridge with mocked dependencies"""
    return WebSocketHITLBridge(
        websocket_url="wss://test.example.com",
        websocket_api_key="test_api_key",
        state_manager=mock_state_manager,
        approval_timeout=60
    )


class TestWebSocketHITLBridge:
    """Test cases for WebSocket HITL bridge"""
    
    @pytest.mark.asyncio
    async def test_request_human_approval_success(self, websocket_bridge, mock_state_manager):
        """Test successful human approval request"""
        # Mock WebSocket API call
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Mock approval response
            approval_response = {
                "approval_id": "test_approval",
                "action": "approve",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Simulate approval response after short delay
            async def mock_approval():
                await asyncio.sleep(0.1)
                await websocket_bridge.handle_approval_response(approval_response)
            
            # Start approval task
            approval_task = asyncio.create_task(mock_approval())
            
            # Request approval
            response = await websocket_bridge.request_human_approval(
                run_id="test_run",
                checkpoint_type="payload_approval",
                context={"test": "data"},
                user_id="test_user"
            )
            
            # Wait for mock approval
            await approval_task
            
            # Verify response
            assert response["action"] == "approve"
            mock_state_manager.pause_run.assert_called_once()
            mock_state_manager.resume_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_human_approval_timeout(self, websocket_bridge):
        """Test approval request timeout"""
        # Mock WebSocket API call
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Set very short timeout
            websocket_bridge.approval_timeout = 0.1
            
            # Request approval (should timeout)
            with pytest.raises(Exception, match="timed out"):
                await websocket_bridge.request_human_approval(
                    run_id="test_run",
                    checkpoint_type="payload_approval",
                    context={"test": "data"}
                )
    
    @pytest.mark.asyncio
    async def test_handle_approval_response_valid(self, websocket_bridge):
        """Test handling valid approval response"""
        # Create pending approval
        approval_id = "test_approval"
        websocket_bridge.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "run_id": "test_run",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data"},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Handle response
        response = {
            "approval_id": approval_id,
            "action": "approve",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await websocket_bridge.handle_approval_response(response)
        
        assert success is True
        assert approval_id not in websocket_bridge.pending_approvals
    
    @pytest.mark.asyncio
    async def test_handle_approval_response_invalid(self, websocket_bridge):
        """Test handling invalid approval response"""
        # Handle response for non-existent approval
        response = {
            "approval_id": "nonexistent",
            "action": "approve",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await websocket_bridge.handle_approval_response(response)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_send_status_update(self, websocket_bridge):
        """Test sending status update"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await websocket_bridge.send_status_update(
                run_id="test_run",
                status=HITLStatus.RUNNING,
                current_step="payload_suggestion",
                message="Processing...",
                user_id="test_user"
            )
            
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_step_completion(self, websocket_bridge):
        """Test sending step completion notification"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await websocket_bridge.send_step_completion(
                run_id="test_run",
                step="payload_suggestion",
                result={"model": "test-model"},
                user_id="test_user"
            )
            
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_error_notification(self, websocket_bridge):
        """Test sending error notification"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await websocket_bridge.send_error_notification(
                run_id="test_run",
                error="Test error",
                step="payload_validation",
                user_id="test_user"
            )
            
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_pending_approvals(self, websocket_bridge):
        """Test listing pending approvals"""
        # Add test approvals
        approval1 = {
            "approval_id": "approval1",
            "run_id": "run1",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data1"},
            "user_id": "user1",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        approval2 = {
            "approval_id": "approval2",
            "run_id": "run2",
            "checkpoint_type": "validation_review",
            "context": {"test": "data2"},
            "user_id": "user2",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        websocket_bridge.pending_approvals["approval1"] = approval1
        websocket_bridge.pending_approvals["approval2"] = approval2
        
        # List all approvals
        all_approvals = await websocket_bridge.list_pending_approvals()
        assert len(all_approvals) == 2
        
        # List approvals for specific user
        user1_approvals = await websocket_bridge.list_pending_approvals("user1")
        assert len(user1_approvals) == 1
        assert user1_approvals[0]["approval_id"] == "approval1"
    
    @pytest.mark.asyncio
    async def test_cancel_approval(self, websocket_bridge):
        """Test cancelling a pending approval"""
        # Add test approval
        approval_id = "test_approval"
        websocket_bridge.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "run_id": "test_run",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data"},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Cancel approval
        success = await websocket_bridge.cancel_approval(approval_id, "Test cancellation")
        
        assert success is True
        assert approval_id not in websocket_bridge.pending_approvals
    
    @pytest.mark.asyncio
    async def test_websocket_api_error_handling(self, websocket_bridge):
        """Test WebSocket API error handling"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock API error
            mock_response = Mock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Should not raise exception, just log error
            await websocket_bridge.send_status_update(
                run_id="test_run",
                status=HITLStatus.RUNNING,
                current_step="test_step"
            )
            
            mock_post.assert_called_once()


class TestHITLWebSocketHandler:
    """Test cases for HITL WebSocket message handler"""
    
    @pytest.fixture
    def handler(self, websocket_bridge):
        """WebSocket handler with mocked bridge"""
        return HITLWebSocketHandler(websocket_bridge)
    
    @pytest.mark.asyncio
    async def test_handle_approval_response_message(self, handler, websocket_bridge):
        """Test handling approval response message"""
        # Setup pending approval
        approval_id = "test_approval"
        websocket_bridge.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "run_id": "test_run",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data"},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Handle message
        message = {
            "type": "hitl_approval_response",
            "data": {
                "approval_id": approval_id,
                "action": "approve",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        response = await handler.handle_message(message)
        
        assert response["type"] == "hitl_approval_response_ack"
        assert response["success"] is True
        assert response["approval_id"] == approval_id
    
    @pytest.mark.asyncio
    async def test_handle_list_pending_message(self, handler, websocket_bridge):
        """Test handling list pending approvals message"""
        # Add test approval
        approval_id = "test_approval"
        websocket_bridge.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "run_id": "test_run",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data"},
            "user_id": "test_user",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        # Handle message
        message = {
            "type": "hitl_list_pending",
            "data": {"user_id": "test_user"}
        }
        
        response = await handler.handle_message(message)
        
        assert response["type"] == "hitl_pending_approvals"
        assert len(response["data"]) == 1
        assert response["data"][0]["approval_id"] == approval_id
    
    @pytest.mark.asyncio
    async def test_handle_cancel_approval_message(self, handler, websocket_bridge):
        """Test handling cancel approval message"""
        # Add test approval
        approval_id = "test_approval"
        websocket_bridge.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "run_id": "test_run",
            "checkpoint_type": "payload_approval",
            "context": {"test": "data"},
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Handle message
        message = {
            "type": "hitl_cancel_approval",
            "data": {
                "approval_id": approval_id,
                "reason": "User cancelled"
            }
        }
        
        response = await handler.handle_message(message)
        
        assert response["type"] == "hitl_cancel_approval_ack"
        assert response["success"] is True
        assert response["approval_id"] == approval_id
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message(self, handler):
        """Test handling unknown message type"""
        message = {
            "type": "unknown_message_type",
            "data": {}
        }
        
        response = await handler.handle_message(message)
        
        assert response["type"] == "error"
        assert "Unknown message type" in response["message"]


class TestBrowserAgentHITLIntegration:
    """Test cases for browser agent HITL integration"""
    
    @pytest.fixture
    def integration(self, websocket_bridge):
        """Browser agent integration with mocked bridge"""
        return BrowserAgentHITLIntegration(websocket_bridge)
    
    @pytest.mark.asyncio
    async def test_request_browser_approval(self, integration, websocket_bridge):
        """Test requesting browser action approval"""
        # Mock approval response
        websocket_bridge.request_human_approval = AsyncMock(return_value={
            "action": "approve",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Request approval
        response = await integration.request_browser_approval(
            run_id="test_run",
            action_description="Click login button",
            browser_context={"url": "https://example.com", "element": "button#login"},
            user_id="test_user",
            session_id="test_session"
        )
        
        # Verify
        assert response["action"] == "approve"
        websocket_bridge.request_human_approval.assert_called_once()
        
        # Check call arguments
        call_args = websocket_bridge.request_human_approval.call_args
        assert call_args[1]["checkpoint_type"] == "browser_action_approval"
        assert call_args[1]["run_id"] == "test_run"
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["session_id"] == "test_session"
    
    @pytest.mark.asyncio
    async def test_notify_browser_completion(self, integration, websocket_bridge):
        """Test notifying browser of action completion"""
        # Mock bridge method
        websocket_bridge.send_step_completion = AsyncMock()
        
        # Notify completion
        await integration.notify_browser_completion(
            run_id="test_run",
            action_result={"success": True, "screenshot": "base64data"},
            user_id="test_user",
            session_id="test_session"
        )
        
        # Verify
        websocket_bridge.send_step_completion.assert_called_once()
        
        # Check call arguments
        call_args = websocket_bridge.send_step_completion.call_args
        assert call_args[1]["run_id"] == "test_run"
        assert call_args[1]["step"] == "browser_action"
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["session_id"] == "test_session"


if __name__ == "__main__":
    pytest.main([__file__])
