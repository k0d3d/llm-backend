"""
HITL API endpoints for human-in-the-loop workflows
"""

import os
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus
from llm_backend.core.hitl.persistence import create_state_manager
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge
from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.core.types.common import RunInput

router = APIRouter(prefix="/hitl", tags=["HITL"])

# Initialize state management
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/tohju")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "wss://ws.tohju.com")
WEBSOCKET_API_KEY = os.getenv("WEBSOCKET_API_KEY")

state_manager = create_state_manager(DATABASE_URL)
websocket_bridge = WebSocketHITLBridge(WEBSOCKET_URL, WEBSOCKET_API_KEY, state_manager)


class HITLRunRequest(BaseModel):
    """Request to start a HITL run"""
    run_input: RunInput
    hitl_config: Optional[HITLConfig] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class HITLRunResponse(BaseModel):
    """Response from starting a HITL run"""
    run_id: str
    status: str
    message: str
    websocket_url: Optional[str] = None


class HITLApprovalRequest(BaseModel):
    """Human approval response"""
    approval_id: Optional[str] = None
    action: Optional[str] = Field(None, description="approve, edit, or reject")
    edits: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    approved_by: Optional[str] = None
    
    # Frontend compatibility fields
    runId: Optional[str] = None
    sessionId: Optional[str] = None
    approved: Optional[bool] = None
    
    def model_post_init(self, __context) -> None:
        """Auto-map frontend fields to backend fields"""
        if self.runId and not self.approval_id:
            self.approval_id = self.runId
        if self.approved is not None and not self.action:
            self.action = "approve" if self.approved else "reject"


class HITLStatusResponse(BaseModel):
    """HITL run status response"""
    run_id: str
    status: str
    current_step: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    progress: Dict[str, Any]
    pending_actions: List[Dict[str, Any]]


@router.post("/run", response_model=HITLRunResponse)
async def start_hitl_run(
    request: HITLRunRequest,
    background_tasks: BackgroundTasks
) -> HITLRunResponse:
    """
    Start a new Human-in-the-Loop (HITL) workflow
    
    Creates a new HITL run that will execute with human oversight checkpoints.
    The run executes asynchronously in the background and communicates via WebSocket
    when human approval is required.
    
    - **run_input**: Complete input configuration including prompt and tool settings
    - **hitl_config**: Optional HITL configuration (defaults to standard checkpoints)
    - **user_id**: User identifier for approval routing
    - **session_id**: Session identifier for WebSocket communication
    
    Returns run_id and WebSocket URL for real-time communication.
    """
    
    try:
        # Get provider from tool configuration
        agent_tool_config = request.run_input.agent_tool_config
        
        # Import provider registry
        from llm_backend.core.providers.registry import ProviderRegistry
        
        # Determine provider based on tool
        provider_name = None
        
        for tool, config in agent_tool_config.items():
            if tool in ProviderRegistry._tool_mappings:
                provider_name = ProviderRegistry._tool_mappings[tool]
                break
        
        if not provider_name:
            raise HTTPException(
                status_code=400,
                detail="No supported provider found in agent tool configuration"
            )
        
        # Create HITL config
        hitl_config = request.hitl_config or HITLConfig()
        
        # Get provider instance
        provider = ProviderRegistry.get_provider(provider_name)
        
        # Initialize orchestrator
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=hitl_config,
            run_input=request.run_input,
            state_manager=state_manager,
            websocket_bridge=websocket_bridge
        )
        
        # Start run in background
        run_id = await orchestrator.start_run(
            original_input=request.run_input.dict(),
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Execute run asynchronously
        background_tasks.add_task(
            orchestrator.execute_run,
            run_id
        )
        
        return HITLRunResponse(
            run_id=run_id,
            status="queued",
            message="HITL run started successfully",
            websocket_url=WEBSOCKET_URL if request.session_id else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run/{run_id}/status", response_model=HITLStatusResponse)
async def get_run_status(run_id: str) -> HITLStatusResponse:
    """
    Get detailed status of a HITL run
    
    Returns comprehensive status information including progress, step history,
    and pending actions. Use this endpoint to monitor run progress and
    determine if human intervention is required.
    
    - **run_id**: Unique identifier for the HITL run
    
    Returns detailed status with progress metrics and pending actions.
    """
    
    try:
        state = await state_manager.load_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Calculate progress
        total_steps = len([
            "capability_discovery",
            "payload_suggestion", 
            "payload_validation",
            "provider_execution",
            "response_processing"
        ])
        
        completed_steps = len([
            event for event in state.step_history 
            if event.status == "completed"
        ])
        
        progress = {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "percentage": int((completed_steps / total_steps) * 100) if total_steps > 0 else 0,
            "current_step": state.current_step,
            "step_history": [
                {
                    "step": event.step,
                    "status": event.status,
                    "timestamp": event.timestamp.isoformat(),
                    "message": event.message
                }
                for event in state.step_history
            ]
        }
        
        return HITLStatusResponse(
            run_id=state.run_id,
            status=state.status,
            current_step=state.current_step,
            created_at=state.created_at,
            updated_at=state.updated_at,
            expires_at=state.expires_at,
            progress=progress,
            pending_actions=state.pending_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{run_id}/approve")
async def approve_checkpoint(
    run_id: str,
    approval: HITLApprovalRequest
) -> Dict[str, Any]:
    """
    Submit human approval or rejection for a HITL checkpoint
    
    Processes human decisions for pending approval requests. Supports three actions:
    - **approve**: Continue execution with current parameters
    - **edit**: Continue with modified parameters (provide edits object)
    - **reject**: Stop execution with reason
    
    - **approval_id**: Unique identifier for the pending approval
    - **action**: One of 'approve', 'edit', or 'reject'
    - **edits**: Optional modifications for 'edit' action
    - **reason**: Required for 'reject' action, optional for others
    - **approved_by**: User identifier submitting the approval
    
    Returns success confirmation and processes the approval immediately.
    """
    
    try:
        # Process approval through WebSocket bridge
        approval_response = {
            "approval_id": approval.approval_id,
            "action": approval.action,
            "edits": approval.edits,
            "reason": approval.reason,
            "approved_by": approval.approved_by,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await websocket_bridge.handle_approval_response(approval_response)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Invalid approval ID or approval already processed"
            )
        
        return {
            "success": True,
            "message": f"Approval {approval.action} processed successfully",
            "run_id": run_id,
            "approval_id": approval.approval_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{run_id}/pause")
async def pause_run(run_id: str) -> Dict[str, Any]:
    """Pause a running HITL run"""
    
    try:
        state = await state_manager.load_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="Run not found")
        
        if state.status != HITLStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause run in status: {state.status}"
            )
        
        # Pause the run
        await state_manager.pause_run(
            run_id=run_id,
            checkpoint_type="manual_pause",
            context={"paused_by": "api_request"}
        )
        
        return {
            "success": True,
            "message": "Run paused successfully",
            "run_id": run_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{run_id}/resume")
async def resume_run(
    run_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Resume a paused HITL run"""
    
    try:
        state = await state_manager.load_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="Run not found")
        
        if state.status != HITLStatus.AWAITING_HUMAN:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume run in status: {state.status}"
            )
        
        # Resume the run
        await state_manager.resume_run(
            run_id=run_id,
            approval_response={"action": "resume", "timestamp": datetime.utcnow().isoformat()}
        )
        
        # Continue execution in background
        provider_name = state.original_input.get("provider", "replicate")
        provider = ProviderRegistry.get_provider(provider_name)
        
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=state.config,
            run_input=state.original_input,
            state_manager=state_manager,
            websocket_bridge=websocket_bridge
        )
        
        background_tasks.add_task(
            orchestrator.execute
        )
        
        return {
            "success": True,
            "message": "Run resumed successfully",
            "run_id": run_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/run/{run_id}")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    """Cancel a HITL run"""
    
    try:
        state = await state_manager.load_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Update state to cancelled
        state.status = HITLStatus.FAILED
        state.updated_at = datetime.utcnow()
        await state_manager.save_state(state)
        
        # Cancel any pending approvals
        pending_approvals = await websocket_bridge.list_pending_approvals()
        for approval in pending_approvals:
            if approval["run_id"] == run_id:
                await websocket_bridge.cancel_approval(
                    approval["approval_id"],
                    "Run cancelled"
                )
        
        return {
            "success": True,
            "message": "Run cancelled successfully",
            "run_id": run_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def list_runs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    List HITL runs with optional filtering
    
    - **user_id**: Filter runs by user ID
    - **status**: Filter by run status (queued, running, awaiting_human, completed, failed, cancelled)
    - **limit**: Maximum number of runs to return (default: 50)
    
    Returns a list of runs with basic information for overview purposes.
    """
    
    try:
        runs = await state_manager.list_active_runs(user_id)
        
        # Filter by status if provided
        if status:
            runs = [run for run in runs if run["status"] == status]
        
        # Limit results
        runs = runs[:limit]
        
        return {
            "runs": runs,
            "total": len(runs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/active")
async def get_active_runs(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get active HITL runs for session resume functionality
    
    - **user_id**: Filter runs by user ID
    - **session_id**: Filter runs by session ID
    - **status**: Filter by specific status (awaiting_human, running, etc.)
    
    Returns detailed information about active runs that can be resumed.
    """
    
    try:
        runs = await state_manager.list_active_runs(user_id)
        
        # Filter by session_id if provided
        if session_id:
            runs = [run for run in runs if run.get("session_id") == session_id]
        
        # Filter by status if provided (default to awaiting_human for resume)
        if status:
            runs = [run for run in runs if run["status"] == status]
        else:
            # Default to runs that can be resumed
            runs = [run for run in runs if run["status"] in ["awaiting_human", "paused"]]
        
        # Enhance with resume-specific information
        enhanced_runs = []
        for run in runs:
            state = await state_manager.load_state(run["run_id"])
            if state:
                enhanced_runs.append({
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "current_step": state.current_step,
                    "created_at": state.created_at.isoformat(),
                    "expires_at": state.expires_at.isoformat() if state.expires_at else None,
                    "pending_actions": state.pending_actions,
                    "context_summary": f"Step: {state.current_step}, Actions: {len(state.pending_actions)}",
                    "user_id": run.get("user_id"),
                    "session_id": run.get("session_id")
                })
        
        return {
            "runs": enhanced_runs,
            "total": len(enhanced_runs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/state")
async def get_run_state(run_id: str) -> Dict[str, Any]:
    """
    Get complete run state for session resume
    
    Returns all artifacts, step history, and context needed to resume a HITL workflow.
    This endpoint provides comprehensive state information for frontend resume functionality.
    """
    
    try:
        state = await state_manager.load_state(run_id)
        if not state:
            raise HTTPException(status_code=404, detail="Run not found")
        
        return {
            "run_id": state.run_id,
            "status": state.status,
            "current_step": state.current_step,
            "original_input": state.original_input,
            "hitl_config": state.config.dict() if state.config else {},
            
            # Current state artifacts
            "capabilities": state.capabilities,
            "suggested_payload": state.suggested_payload,
            "validation_issues": state.validation_issues,
            "raw_response": state.raw_response,
            "processed_response": state.processed_response,
            
            # Pending human actions
            "pending_actions": state.pending_actions,
            "approval_token": getattr(state, 'approval_token', None),
            "expires_at": state.expires_at.isoformat() if state.expires_at else None,
            
            # Step history for context
            "step_history": [
                {
                    "step": event.step,
                    "status": event.status,
                    "timestamp": event.timestamp.isoformat(),
                    "actor": event.actor,
                    "message": event.message,
                    "metadata": event.metadata
                }
                for event in state.step_history
            ],
            
            # Validation summary for current step
            "validation_summary": getattr(state, 'validation_summary', None),
            
            # Timestamps
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            
            # Metrics
            "total_execution_time_ms": getattr(state, 'total_execution_time_ms', 0),
            "human_review_time_ms": getattr(state, 'human_review_time_ms', 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/active")
async def get_session_active_runs(
    session_id: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all active HITL runs for a specific session
    
    This endpoint is specifically designed for session initialization,
    allowing the frontend to load all pending HITL requests when a user
    opens or resumes a chat session.
    """
    
    try:
        runs = await state_manager.list_active_runs(user_id, session_id)
        
        # Filter by active statuses (session_id filtering now done in database)
        session_runs = [
            run for run in runs 
            if run["status"] in ["awaiting_human", "paused", "running"]
        ]
        
        # Get detailed state for each run
        detailed_runs = []
        for run in session_runs:
            state = await state_manager.load_state(run["run_id"])
            if state:
                detailed_runs.append({
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "current_step": state.current_step,
                    "created_at": state.created_at.isoformat(),
                    "message": f"HITL approval required for {state.current_step}",
                    "pending_actions": state.pending_actions,
                    "expires_at": state.expires_at.isoformat() if state.expires_at else None,
                    "validation_summary": getattr(state, 'validation_summary', None)
                })
        
        return {
            "session_id": session_id,
            "runs": detailed_runs,
            "total": len(detailed_runs),
            "has_pending_approvals": len([r for r in detailed_runs if r["status"] == "awaiting_human"]) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/approvals/pending")
async def list_pending_approvals(user_id: Optional[str] = None) -> Dict[str, Any]:
    """List pending approvals"""
    
    try:
        approvals = await websocket_bridge.list_pending_approvals(user_id)
        
        return {
            "approvals": approvals,
            "total": len(approvals)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def list_providers() -> Dict[str, Any]:
    """List available providers"""
    
    try:
        providers = []
        for name, provider_class in ProviderRegistry._providers.items():
            providers.append({
                "name": name,
                "class": provider_class.__name__,
                "tools": [
                    tool.value for tool, mapped_provider 
                    in ProviderRegistry._tool_mappings.items()
                    if mapped_provider == name
                ]
            })
        
        return {
            "providers": providers,
            "total": len(providers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket message handler endpoint
@router.post("/websocket/message")
async def handle_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming WebSocket messages related to HITL"""
    
    try:
        from llm_backend.core.hitl.websocket_bridge import HITLWebSocketHandler
        
        handler = HITLWebSocketHandler(websocket_bridge)
        response = await handler.handle_message(message)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
