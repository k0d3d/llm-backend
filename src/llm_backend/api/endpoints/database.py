"""
Database API endpoints for client-side DuckDB operations with HITL

Provides endpoints for:
- Schema inspection (read-only, no approval)
- Database queries (SELECT only, auto-execute)
- SQL execution (INSERT/UPDATE/DELETE, requires HITL approval)
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import logging

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig
from llm_backend.core.hitl.shared_bridge import get_shared_state_manager, get_shared_websocket_bridge
from llm_backend.core.types.common import RunInput
from llm_backend.providers.database_provider import DatabaseProvider

router = APIRouter(prefix="/database", tags=["Database"])

# Set up logger
logger = logging.getLogger(__name__)

# Get shared HITL components
state_manager = get_shared_state_manager()
websocket_bridge = get_shared_websocket_bridge()


class DatabaseRunRequest(BaseModel):
    """Request for database operation"""
    prompt: str = Field(description="Natural language database request")
    session_id: str = Field(description="Session identifier")
    user_id: str = Field(description="User identifier")
    agent_tool_config: Dict[str, Any] = Field(
        description="Tool configuration with current_schema"
    )
    hitl_config: Optional[Dict[str, Any]] = None
    document_url: Optional[str] = None
    document_data: Optional[Dict[str, Any]] = None


class DatabaseRunResponse(BaseModel):
    """Response from database operation"""
    run_id: str
    status: str
    message: str
    operation_type: str  # schema_inspection, query, or execution
    requires_approval: bool
    auto_execute: bool
    checkpoint_data: Optional[Dict[str, Any]] = None
    websocket_url: Optional[str] = None


class DatabaseApprovalRequest(BaseModel):
    """Approval request for SQL execution"""
    run_id: str
    session_id: str
    approved: bool
    edits: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@router.post("/run", response_model=DatabaseRunResponse)
async def run_database_operation(
    request: DatabaseRunRequest,
    enable_hitl: bool = Query(default=True, description="Enable HITL for SQL execution")
) -> DatabaseRunResponse:
    """
    Execute database operation with Pydantic AI + HITL

    Flow:
    1. Parse user intent (schema inspection, query, or execution)
    2. Route to appropriate Pydantic agent (schema/query/execution)
    3. If SQL execution → HITL checkpoint for approval
    4. Return typed results or checkpoint data

    **Schema Inspection** (read-only):
    - "show tables", "describe customers"
    - Auto-executes, no approval needed
    - Returns table/column information

    **Database Query** (SELECT only):
    - "show me all invoices", "count customers"
    - Auto-executes, no approval needed
    - Returns query results

    **SQL Execution** (INSERT/UPDATE/DELETE):
    - "insert invoice data", "update customer", "delete old records"
    - Requires HITL approval
    - Shows preview before execution

    Args:
        request: Database operation request
        enable_hitl: Enable HITL for SQL execution (default: True)

    Returns:
        Operation result or HITL checkpoint
    """

    try:
        logger.info(f"🗄️ Database operation request: {request.prompt[:100]}...")

        # Extract current_schema from agent_tool_config
        client_db_config = request.agent_tool_config.get("client-database-tool", {})
        current_schema = client_db_config.get("current_schema", {"tables": []})

        logger.info(f"📊 Current schema: {len(current_schema.get('tables', []))} tables")

        # Initialize DatabaseProvider
        provider = DatabaseProvider(config={
            "current_schema": current_schema,
            "document_url": request.document_url,
            "document_data": request.document_data
        })

        # Convert to RunInput format
        run_input = RunInput(
            prompt=request.prompt,
            session_id=request.session_id,
            user_id=request.user_id,
            agent_tool_config=request.agent_tool_config,
            hitl_config=request.hitl_config or {},
            document_url=request.document_url
        )

        # Set run_input on provider
        provider.run_input = run_input

        # Create HITL config
        hitl_config = HITLConfig(**(request.hitl_config or {}))

        # Initialize HITL orchestrator
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=state_manager,
            websocket_bridge=websocket_bridge
        )

        # Link orchestrator to provider
        provider.set_orchestrator(orchestrator)

        logger.info(f"🚀 Starting HITL orchestrator for database operation")

        # Run orchestrator
        # This will:
        # 1. Create payload using DatabaseProvider (routes to Pydantic agents)
        # 2. Validate payload
        # 3. If requires_approval, pause at HITL checkpoint
        # 4. If auto_execute, return results immediately
        result = await orchestrator.run()

        logger.info(f"✅ Orchestrator completed with result: {result}")

        # Extract checkpoint data from result
        if isinstance(result, dict):
            checkpoint_data = result.get("checkpoint_data")
            message = result.get("message", "Operation completed")
            requires_approval = result.get("requires_approval", False)
            auto_execute = result.get("auto_execute", False)
            operation_type = result.get("operation", "unknown")
        else:
            checkpoint_data = None
            message = str(result)
            requires_approval = False
            auto_execute = False
            operation_type = "unknown"

        # Get WebSocket URL for real-time updates
        websocket_url = websocket_bridge.get_websocket_url(request.session_id)

        return DatabaseRunResponse(
            run_id=orchestrator.state.run_id,
            status=orchestrator.state.status.value,
            message=message,
            operation_type=operation_type,
            requires_approval=requires_approval,
            auto_execute=auto_execute,
            checkpoint_data=checkpoint_data,
            websocket_url=websocket_url
        )

    except Exception as e:
        logger.error(f"❌ Database operation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database operation failed: {str(e)}"
        )


@router.post("/approve")
async def approve_database_operation(
    request: DatabaseApprovalRequest
) -> Dict[str, Any]:
    """
    Approve or reject a database operation

    Used to approve SQL execution operations that require HITL approval.

    Args:
        request: Approval request with run_id, approved flag, and optional edits

    Returns:
        Updated run status
    """

    try:
        logger.info(f"📝 Approval request for run {request.run_id}: approved={request.approved}")

        # Get run state from state manager
        run_state = await state_manager.get_state(request.run_id)

        if not run_state:
            raise HTTPException(
                status_code=404,
                detail=f"Run {request.run_id} not found"
            )

        # Send approval via WebSocket bridge
        approval_data = {
            "run_id": request.run_id,
            "session_id": request.session_id,
            "approved": request.approved,
            "edits": request.edits,
            "reason": request.reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        await websocket_bridge.send_approval(
            session_id=request.session_id,
            approval_data=approval_data
        )

        logger.info(f"✅ Approval sent for run {request.run_id}")

        return {
            "status": "approval_sent",
            "run_id": request.run_id,
            "approved": request.approved,
            "message": "Approval processed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Approval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed: {str(e)}"
        )


@router.get("/run/{run_id}")
async def get_database_run_status(run_id: str) -> Dict[str, Any]:
    """
    Get status of a database operation run

    Args:
        run_id: Run identifier

    Returns:
        Run status and progress information
    """

    try:
        logger.info(f"📊 Getting status for run {run_id}")

        # Get run state from state manager
        run_state = await state_manager.get_state(run_id)

        if not run_state:
            raise HTTPException(
                status_code=404,
                detail=f"Run {run_id} not found"
            )

        return {
            "run_id": run_id,
            "status": run_state.status.value,
            "current_step": run_state.current_checkpoint.value if run_state.current_checkpoint else None,
            "created_at": run_state.created_at.isoformat(),
            "updated_at": run_state.updated_at.isoformat(),
            "metadata": run_state.metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get status failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get run status: {str(e)}"
        )


@router.get("/health")
async def database_health_check() -> Dict[str, str]:
    """
    Health check endpoint for database API

    Returns:
        Status indicating service health
    """
    return {
        "status": "healthy",
        "service": "database-api",
        "version": "1.0.0"
    }


# Missing import
from datetime import datetime
