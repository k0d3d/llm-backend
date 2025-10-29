
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Query
from rq import Queue

from llm_backend.agents.replicate_team import ReplicateTeam
from llm_backend.core.types.common import AgentTools, RunInput
from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig
from llm_backend.core.hitl.shared_bridge import get_shared_state_manager, get_shared_websocket_bridge
from llm_backend.workers.connection import get_redis_connection
from llm_backend.workers.tasks import process_hitl_orchestrator

logger = logging.getLogger(__name__)


router = APIRouter()

# Get shared HITL components
state_manager = get_shared_state_manager()
websocket_bridge = get_shared_websocket_bridge()

# Initialize Redis Queue
redis_conn = get_redis_connection()
task_queue = Queue('default', connection=redis_conn)


@router.post("/run")
async def run_replicate_team(
    run_input: RunInput,
    background_tasks: BackgroundTasks,
    enable_hitl: Optional[bool] = Query(False, description="Enable Human-in-the-Loop workflow"),
    user_id: Optional[str] = Query(None, description="User ID for HITL notifications"),
    session_id: Optional[str] = Query(None, description="Session ID for WebSocket communication")
):
    """
    Run Replicate team with optional HITL workflow
    
    Backward compatible endpoint that supports both legacy direct execution
    and new HITL workflow based on enable_hitl parameter.
    """
    
    # Debug logging for request parameters
    # print(f"🔍 DEBUG REQUEST: enable_hitl={enable_hitl}")
    # print(f"🔍 DEBUG REQUEST: user_id={user_id}")
    # print(f"🔍 DEBUG REQUEST: session_id={session_id}")
    # print(f"🔍 DEBUG REQUEST: run_input.prompt='{run_input.prompt}'")
    # print(f"🔍 DEBUG REQUEST: run_input.agent_tool_config={run_input.agent_tool_config}")
    # print(f"🔍 DEBUG REQUEST: run_input dict={run_input.dict()}")
    
    if enable_hitl:
        # Use new HITL orchestrator
        from llm_backend.providers.replicate_provider import ReplicateProvider
        
        # Get tool config for provider
        agent_tool_config = run_input.agent_tool_config
        # print(f"🔍 DEBUG: agent_tool_config keys: {list(agent_tool_config.keys()) if agent_tool_config else 'None'}")
        # print(f"🔍 DEBUG: AgentTools.REPLICATETOOL value: {AgentTools.REPLICATETOOL}")
        # print(f"🔍 DEBUG: Full agent_tool_config: {agent_tool_config}")
        
        replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)
        # print(f"🔍 DEBUG: replicate_agent_tool_config: {replicate_agent_tool_config}")
        
        if replicate_agent_tool_config is None:
            # Try alternative key formats
            for key in agent_tool_config.keys():
                if "replicate" in key.lower():
                    print(f"🔍 DEBUG: Found alternative replicate key: {key}")
                    replicate_agent_tool_config = agent_tool_config.get(key)
                    break
        
        tool_config = replicate_agent_tool_config.get("data", {}) if replicate_agent_tool_config else {}

        # Log tool configuration for debugging
        logger.debug(
            "Tool config extracted from request: model_name=%s, has_example_input=%s",
            tool_config.get('model_name', 'MISSING'),
            bool(tool_config.get('example_input'))
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("example_input: %s", tool_config.get('example_input', {}))

        # Create provider instance
        provider = ReplicateProvider(config={
            "name": tool_config.get("model_name", ""),
            "description": tool_config.get("description", ""),
            "example_input": tool_config.get("example_input", {}),
            "latest_version": tool_config.get("latest_version", "")
        })

        logger.debug("Provider created with config keys: %s", list(provider.config.keys()))
        
        hitl_config = HITLConfig(
            policy="auto_with_thresholds",
            allowed_steps=["information_review", "payload_review", "response_review"],
            use_natural_language_hitl=True  # Enable NL conversation mode
        )
        
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=state_manager,
            websocket_bridge=websocket_bridge
        )
        
        # Start HITL run and save initial state
        run_id = await orchestrator.start_run(
            original_input=run_input.dict(),
            user_id=run_input.user_id,
            session_id=run_input.session_id
        )

        # Queue job instead of background task
        logger.info("Queuing HITL orchestrator job for run_id=%s", run_id)
        job = task_queue.enqueue(
            process_hitl_orchestrator,
            run_input.dict(),
            hitl_config.dict(),
            "replicate",
            job_timeout='30m'
        )

        return {
            "run_id": run_id,
            "job_id": job.id,
            "status": "queued",
            "message": "HITL run started successfully",
            "websocket_url": websocket_bridge.websocket_url,
            "hitl_enabled": True
        }
    
    else:
        # Legacy direct execution
        agent_tool_config = run_input.agent_tool_config
        replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)

        replicate_team = ReplicateTeam(
            prompt=run_input.prompt,
            tool_config=replicate_agent_tool_config.get("data", {}),
            run_input=run_input,
            hitl_enabled=enable_hitl
        )

        return replicate_team.run()


@router.post("/run-hitl")
async def run_replicate_team_hitl(
    run_input: RunInput,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    hitl_config: Optional[HITLConfig] = None
):
    """
    Run Replicate team with HITL workflow (dedicated endpoint)
    
    This endpoint always uses the HITL orchestrator and provides more
    configuration options than the backward-compatible /run endpoint.
    """
    
    # Use provided config or default HITL config
    config = hitl_config or HITLConfig(
        require_human_approval=True,
        checkpoint_payload_suggestion=True,
    )
    
    from llm_backend.core.providers.registry import ProviderRegistry
    provider = ProviderRegistry.get_provider("replicate")
    print(f"🔧 Provider retrieved: {type(provider).__name__}")
    
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=config,
        run_input=run_input,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )
    
    # Start HITL run and save initial state
    run_id = await orchestrator.start_run(
        original_input=run_input.dict(),
        user_id=user_id,
        session_id=session_id
    )

    # Queue job instead of background task
    job = task_queue.enqueue(
        process_hitl_orchestrator,
        run_input.dict(),
        config.dict(),
        "replicate",
        job_timeout='30m'
    )

    return {
        "run_id": run_id,
        "job_id": job.id,
        "status": "queued",
        "message": "HITL run started successfully",
        "websocket_url": websocket_bridge.websocket_url if session_id else None,
        "hitl_enabled": True,
        "config": config.dict()
    }
