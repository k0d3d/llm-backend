"""
WebSocket HITL bridge for browser communication
Integrates HITL checkpoints with browser feedback via WebSocket messages
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiohttp
import logging

from llm_backend.core.hitl.types import HITLStatus

try:  # Avoid circular imports at runtime
    from llm_backend.core.hitl.persistence import HybridStateManager
except ImportError:  # pragma: no cover - fallback for type checking
    HybridStateManager = None  # type: ignore

logger = logging.getLogger(__name__)


class WebSocketHITLBridge:
    """Bridge between HITL orchestrator and WebSocket server"""
    
    _CANCEL_ACTIONS = {"reject", "rejected", "cancel", "cancelled", "timeout"}
    
    def __init__(
        self,
        websocket_url: str,
        websocket_api_key: Optional[str] = None,
        state_manager: Optional["HybridStateManager"] = None,
        approval_timeout: int = 300
    ):
        self.websocket_url = websocket_url
        self.websocket_api_key = websocket_api_key
        self.state_manager = state_manager
        self.approval_timeout = approval_timeout

        if state_manager is None:
            logger.warning(
                "WebSocketHITLBridge initialized without a state_manager; pending approvals "
                "cannot be persisted. Configure persistent storage to avoid data loss."
            )
    
    async def request_human_approval(
        self,
        run_id: str,
        checkpoint_type: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Request human approval through WebSocket and wait for response
        
        Args:
            run_id: HITL run identifier
            checkpoint_type: Type of checkpoint (payload_approval, validation_review, etc.)
            context: Context data for the approval request
            user_id: User to send approval request to
            session_id: Specific browser session to target
            schema: Optional field schema metadata for editable fields
            
        Returns:
            Approval response from human reviewer
        """
        approval_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=self.approval_timeout)
        
        # Enhance context with schema metadata if provided
        enhanced_context = context.copy()
        if schema:
            enhanced_context["schema"] = schema
        
        # Create approval request
        approval_request = {
            "approval_id": approval_id,
            "run_id": run_id,
            "checkpoint_type": checkpoint_type,
            "context": enhanced_context,
            "user_id": user_id,
            "session_id": session_id,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        if not self.state_manager:
            raise RuntimeError(
                "State manager not configured; cannot persist pending approvals."
            )

        # Store pending approval in database
        await self.state_manager.save_pending_approval(approval_request)
        
        # Pause the run in state manager
        if self.state_manager:
            await self.state_manager.pause_run(run_id, checkpoint_type, context)
        
        try:
            # Send approval request via WebSocket
            await self._send_websocket_message({
                "type": "hitl_approval_request",
                "data": approval_request
            }, user_id, session_id)

            return approval_request

        except Exception as e:
            logger.error(f"Failed to dispatch approval request {approval_id}: {e}")
            raise
    
    async def handle_approval_response(self, approval_response: Dict[str, Any]) -> bool:
        """
        Handle approval response from human reviewer
        
        Args:
            approval_response: Dict containing approval decision and metadata
            
        Returns:
            True if response was processed successfully
        """
        approval_id = approval_response.get("approval_id")
        
        if not self.state_manager:
            logger.error("State manager is required to process approval responses")
            return False

        approval_data = None
        for attempt in range(5):
            logger.info(
                "Attempting to load approval %s from state manager (attempt %d)",
                approval_id,
                attempt + 1,
            )
            approval_data = await self.state_manager.load_pending_approval(approval_id)
            if approval_data:
                logger.info(
                    "Successfully loaded approval %s from state manager", approval_id
                )
                break
            await asyncio.sleep(0.5)

        if not approval_data:
            logger.warning(
                "Received response for unknown approval: %s. Attempting fallback to run_id.", approval_id
            )
            # Fallback: Try to find the run by run_id from the approval response
            run_id = approval_response.get("run_id") or approval_response.get("runId")
            if run_id:
                try:
                    # Load the run state directly
                    state = await self.state_manager.load_state(run_id)
                    if state:
                        logger.info(f"Found run {run_id} via fallback, processing approval without approval entry")
                        # Create synthetic approval_data for processing
                        approval_data = {
                            "approval_id": approval_id,
                            "run_id": run_id,
                            "checkpoint_type": "unknown",
                            "context": {},
                            "user_id": approval_response.get("approved_by"),
                            "session_id": state.original_input.get("session_id"),
                            "created_at": datetime.utcnow().isoformat(),
                            "expires_at": datetime.utcnow().isoformat()
                        }
                    else:
                        logger.error(f"Run {run_id} not found in fallback")
                        return False
                except Exception as e:
                    logger.error(f"Fallback failed: {e}")
                    return False
            else:
                logger.error("No run_id provided in approval response for fallback")
                return False
        
        # Validate response
        required_fields = ["approval_id", "action", "timestamp"]
        if not all(field in approval_response for field in required_fields):
            logger.error(f"Invalid approval response format: {approval_response}")
            return False
        
        # Store response and trigger callback
        approval_data["response"] = approval_response
        approval_data["responded_at"] = datetime.utcnow().isoformat()
        
        # Update run status based on action before cleanup
        run_id = approval_data.get("run_id")
        action = approval_response.get("action")

        should_resume = run_id and action not in self._CANCEL_ACTIONS

        if run_id:
            try:
                await self.state_manager.resume_run(run_id, {
                    **approval_response,
                    "action": action,
                    "timestamp": approval_response.get("timestamp", datetime.utcnow().isoformat())
                })
            except Exception as e:
                logger.error(f"Failed to update run {run_id} with approval response: {e}")

        # Clean up from both database and memory
        await self.state_manager.remove_pending_approval(approval_id)

        logger.info(f"Processed approval response for {approval_id}: {approval_response['action']}")

        # Kick off orchestrator resume workflow asynchronously
        if should_resume:
            try:
                from llm_backend.core.hitl.shared_bridge import get_shared_state_manager, get_shared_websocket_bridge
                from llm_backend.core.hitl.orchestrator import HITLOrchestrator
                from llm_backend.core.providers.registry import ProviderRegistry

                shared_state_manager = get_shared_state_manager()
                shared_websocket_bridge = get_shared_websocket_bridge()

                async def _resume_run() -> None:
                    state = await shared_state_manager.load_state(run_id)
                    if not state:
                        logger.warning(f"Cannot resume run {run_id}; state not found")
                        return

                    agent_config = state.original_input.get("agent_tool_config")
                    provider_name, provider_config = ProviderRegistry.resolve_provider(agent_config)

                    if not provider_name:
                        provider_name = state.original_input.get("provider", "replicate")
                        provider_config = provider_config or state.original_input.get("provider_config", {})

                    provider = ProviderRegistry.get_provider(provider_name, provider_config)

                    orchestrator = HITLOrchestrator(
                        provider=provider,
                        config=state.config,
                        run_input=state.original_input,
                        state_manager=shared_state_manager,
                        websocket_bridge=shared_websocket_bridge,
                    )

                    orchestrator.state = state
                    try:
                        await orchestrator.resume_from_state()
                    except Exception as exec_error:
                        logger.error(f"Failed to resume run {run_id}: {exec_error}")

                asyncio.create_task(_resume_run())

            except Exception as bootstrap_error:
                logger.error(f"Failed to bootstrap orchestrator resume for run {run_id}: {bootstrap_error}")

        return True
    
    def create_field_schema(
        self,
        required_fields: List[str],
        optional_fields: Dict[str, Dict[str, Any]] = None,
        editable_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create field schema metadata for approval requests
        
        Args:
            required_fields: List of field names that must be present
            optional_fields: Dict mapping field names to their metadata
                           e.g., {"temperature": {"default": 0.7, "type": "float", "range": [0, 2]}}
            editable_fields: List of fields that can be modified by humans
            
        Returns:
            Schema dictionary for inclusion in approval context
        """
        schema = {
            "required_fields": required_fields or [],
            "optional_fields": optional_fields or {},
            "editable_fields": editable_fields or []
        }
        
        # If no editable fields specified, assume all fields are editable
        if not editable_fields:
            all_fields = set(required_fields or [])
            all_fields.update((optional_fields or {}).keys())
            schema["editable_fields"] = list(all_fields)
        
        return schema
    
    async def send_status_update(
        self,
        run_id: str,
        status: HITLStatus,
        current_step: str,
        message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send HITL status update to browser"""
        
        status_update = {
            "type": "hitl_status_update",
            "data": {
                "run_id": run_id,
                "status": status.value,
                "current_step": current_step,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._send_websocket_message(status_update, user_id, session_id)
    
    async def send_step_completion(
        self,
        run_id: str,
        step: str,
        result: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send step completion notification to browser"""
        
        completion_message = {
            "type": "hitl_step_completion",
            "data": {
                "run_id": run_id,
                "step": step,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._send_websocket_message(completion_message, user_id, session_id)
    
    async def send_error_notification(
        self,
        run_id: str,
        error: str,
        step: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send error notification to browser"""
        
        error_message = {
            "type": "hitl_error",
            "data": {
                "run_id": run_id,
                "error": error,
                "step": step,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        await self._send_websocket_message(error_message, user_id, session_id)
    
    async def send_thinking_status(
        self,
        message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send thinking status to indicate processing"""
        payload = {
            "status": "thinking",
            "action": "thinking",
            "message": message
        }
        await self._send_websocket_message(payload, user_id, session_id)

    async def send_done_thinking(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send done_thinking status to indicate processing complete"""
        payload = {
            "status": "done_thinking",
            "action": "done_thinking",
            "message": None
        }
        await self._send_websocket_message(payload, user_id, session_id)

    async def _send_websocket_message(
        self,
        message: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Send message via WebSocket server REST API"""
        
        if not self.websocket_api_key:
            logger.warning("WebSocket API key not configured, skipping message send")
            return
        
        # Prepare WebSocket server API request
        api_url = f"{self.websocket_url.replace('wss://', 'https://').replace('ws://', 'http://')}/api/update-status"
        
        # Extract status and action from message if available, otherwise use defaults
        status = message.get("status", "hitl_request")
        action = message.get("action", "approval_required")
        
        payload = {
            "sessionId": session_id,
            "userId": user_id,
            "status": status,
            "action": action,
            "message": message.get("message"),
            "data": message if message.get("type") else None
        }
        
        headers = {
            "X-API-Key": self.websocket_api_key,
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send WebSocket message: {response.status} - {await response.text()}")
                    else:
                        msg_type = message.get('type', message.get('action', 'unknown'))
                        logger.debug(f"WebSocket message sent successfully: {msg_type}")
                        
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pending approvals for a user"""
        
        if not self.state_manager:
            raise RuntimeError(
                "State manager not configured; cannot list pending approvals."
            )

        return await self.state_manager.list_pending_approvals(user_id)
    
    async def cancel_approval(self, approval_id: str, reason: str = "Cancelled", run_id: Optional[str] = None) -> bool:
        """Cancel a pending approval"""
        
        if not self.state_manager:
            raise RuntimeError(
                "State manager not configured; cannot cancel pending approvals."
            )

        approval_data = await self.state_manager.load_pending_approval(approval_id)
        
        # Fallback: If approval not found but run_id provided, try to cancel the run directly
        if not approval_data and run_id:
            logger.warning(f"Approval {approval_id} not found, attempting to cancel run {run_id} directly")
            try:
                state = await self.state_manager.load_state(run_id)
                if state:
                    await self.state_manager.resume_run(run_id, {
                        "action": "cancelled",
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.info(f"Cancelled run {run_id} via fallback: {reason}")
                    return True
                else:
                    logger.error(f"Run {run_id} not found for cancellation fallback")
                    return False
            except Exception as e:
                logger.error(f"Fallback cancellation failed: {e}")
                return False
        elif not approval_data:
            logger.warning(f"Approval {approval_id} not found and no run_id provided for fallback")
            return False
        
        # Get run_id to update run status
        run_id = approval_data.get("run_id")
        
        # Trigger callback with cancellation
        if approval_id in self.approval_callbacks:
            callback = self.approval_callbacks.pop(approval_id)
            callback({
                "approval_id": approval_id,
                "action": "cancelled",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Clean up from database
        await self.state_manager.remove_pending_approval(approval_id)
        
        # Mark associated run as cancelled to remove from /active list
        if run_id:
            try:
                await self.state_manager.resume_run(run_id, {
                    "action": "cancelled",
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to update run status on cancellation: {e}")

        logger.info(f"Cancelled approval {approval_id}: {reason}")
        return True


class HITLWebSocketHandler:
    """Handler for incoming WebSocket messages related to HITL"""
    
    def __init__(self, bridge: WebSocketHITLBridge):
        self.bridge = bridge
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket message"""
        
        message_type = message.get("type")
        data = message.get("data", {})
        
        if message_type == "hitl_approval_response":
            success = await self.bridge.handle_approval_response(data)
            return {
                "type": "hitl_approval_response_ack",
                "success": success,
                "approval_id": data.get("approval_id")
            }
        
        elif message_type == "hitl_list_pending":
            user_id = data.get("user_id")
            approvals = await self.bridge.list_pending_approvals(user_id)
            return {
                "type": "hitl_pending_approvals",
                "data": approvals
            }
        
        elif message_type == "hitl_cancel_approval":
            approval_id = data.get("approval_id")
            reason = data.get("reason", "Cancelled by user")
            success = await self.bridge.cancel_approval(approval_id, reason)
            return {
                "type": "hitl_cancel_approval_ack",
                "success": success,
                "approval_id": approval_id
            }
        
        else:
            logger.warning(f"Unknown HITL message type: {message_type}")
            return {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }


# Integration helper for browser agents
class BrowserAgentHITLIntegration:
    """Helper class for integrating HITL with browser agents"""
    
    def __init__(self, websocket_bridge: WebSocketHITLBridge):
        self.bridge = websocket_bridge
    
    async def request_browser_approval(
        self,
        run_id: str,
        action_description: str,
        browser_context: Dict[str, Any],
        user_id: str,
        session_id: Optional[str] = None,
        editable_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Request approval for browser action with optional field schema"""
        
        context = {
            "action_description": action_description,
            "browser_context": browser_context,
            "requires_human_review": True
        }
        
        # Create schema for common browser action fields
        schema = None
        if editable_fields or browser_context:
            common_optional_fields = {
                "selector": {"type": "string", "description": "CSS selector for element"},
                "text": {"type": "string", "description": "Text to input or click"},
                "url": {"type": "string", "description": "URL to navigate to"},
                "wait_time": {"default": 1.0, "type": "float", "description": "Wait time in seconds"},
                "screenshot": {"default": True, "type": "boolean", "description": "Take screenshot after action"}
            }
            
            schema = self.bridge.create_field_schema(
                required_fields=["action_description"],
                optional_fields=common_optional_fields,
                editable_fields=editable_fields
            )
        
        return await self.bridge.request_human_approval(
            run_id=run_id,
            checkpoint_type="browser_action_approval",
            context=context,
            user_id=user_id,
            session_id=session_id,
            schema=schema
        )
    
    async def notify_browser_completion(
        self,
        run_id: str,
        action_result: Dict[str, Any],
        user_id: str,
        session_id: Optional[str] = None
    ) -> None:
        """Notify browser of action completion"""
        
        await self.bridge.send_step_completion(
            run_id=run_id,
            step="browser_action",
            result=action_result,
            user_id=user_id,
            session_id=session_id
        )
