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
    approval_id: str
    action: str = Field(..., description="approve, edit, or reject")
    edits: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    approved_by: Optional[str] = None


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
    """Start a new HITL run"""
    
    try:
        # Get provider from tool configuration
        agent_tool_config = request.run_input.agent_tool_config
        
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
        
        # Initialize orchestrator
        orchestrator = HITLOrchestrator(
            provider_name=provider_name,
            config=hitl_config,
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
    """Get status of a HITL run"""
    
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
    """Handle human approval for a checkpoint"""
    
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
        orchestrator = HITLOrchestrator(
            provider_name=state.original_input.get("provider", "replicate"),
            config=state.config,
            state_manager=state_manager,
            websocket_bridge=websocket_bridge
        )
        
        background_tasks.add_task(
            orchestrator.execute_run,
            run_id
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
    """List HITL runs"""
    
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
