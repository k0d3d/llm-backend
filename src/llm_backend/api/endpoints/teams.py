

import os
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Query

from llm_backend.agents.replicate_team import ReplicateTeam
from llm_backend.core.types.common import AgentTools, RunInput
from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig
from llm_backend.core.hitl.persistence import create_state_manager
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge


router = APIRouter()

# Initialize HITL components for backward compatibility
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/tohju")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "wss://ws.tohju.com")
WEBSOCKET_API_KEY = os.getenv("WEBSOCKET_API_KEY")

state_manager = create_state_manager(DATABASE_URL)
websocket_bridge = WebSocketHITLBridge(WEBSOCKET_URL, WEBSOCKET_API_KEY, state_manager)


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
    # print(f"🔍 DEBUG REQUEST: run_input.document_url={run_input.document_url}")
    # print(f"🔍 DEBUG REQUEST: run_input.agent_tool_config={run_input.agent_tool_config}")
    # print(f"🔍 DEBUG REQUEST: run_input dict={run_input.dict()}")
    
    if enable_hitl:
        # Use new HITL orchestrator
        from llm_backend.providers.replicate_provider import ReplicateProvider
        
        # Get tool config for provider
        agent_tool_config = run_input.agent_tool_config
        print(f"🔍 DEBUG: agent_tool_config keys: {list(agent_tool_config.keys()) if agent_tool_config else 'None'}")
        print(f"🔍 DEBUG: AgentTools.REPLICATETOOL value: {AgentTools.REPLICATETOOL}")
        print(f"🔍 DEBUG: Full agent_tool_config: {agent_tool_config}")
        
        replicate_agent_tool_config = agent_tool_config.get(AgentTools.REPLICATETOOL)
        print(f"🔍 DEBUG: replicate_agent_tool_config: {replicate_agent_tool_config}")
        
        if replicate_agent_tool_config is None:
            # Try alternative key formats
            for key in agent_tool_config.keys():
                if "replicate" in key.lower():
                    print(f"🔍 DEBUG: Found alternative replicate key: {key}")
                    replicate_agent_tool_config = agent_tool_config.get(key)
                    break
        
        tool_config = replicate_agent_tool_config.get("data", {}) if replicate_agent_tool_config else {}
        print(f"🔍 DEBUG: Final tool_config: {tool_config}")
        
        # Create provider instance
        provider = ReplicateProvider(config={
            "name": tool_config.get("model_name", ""),
            "description": tool_config.get("description", ""),
            "example_input": tool_config.get("example_input", {}),
            "latest_version": tool_config.get("latest_version", "")
        })
        
        hitl_config = HITLConfig(
            policy="auto_with_thresholds",
            allowed_steps=["information_review", "payload_review", "response_review"]
        )
        
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=hitl_config,
            run_input=run_input
        )
        
        # Execute HITL run synchronously for debugging
        print("🔄 About to execute HITL orchestrator...")
        result = await orchestrator.execute()
        # print(f"🔄 HITL orchestrator result: {result}")
        
        return {
            "run_id": orchestrator.run_id,
            "status": "queued",
            "message": "HITL run started successfully",
            "websocket_url": WEBSOCKET_URL,
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
        checkpoint_validation_review=True,
        checkpoint_response_review=True
    )
    
    orchestrator = HITLOrchestrator(
        provider_name="replicate",
        config=config,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )
    
    # Start HITL run
    run_id = await orchestrator.start_run(
        original_input=run_input.dict(),
        user_id=user_id,
        session_id=session_id
    )
    
    # Execute run in background
    background_tasks.add_task(orchestrator.execute_run, run_id)
    
    return {
        "run_id": run_id,
        "status": "queued",
        "message": "HITL run started successfully",
        "websocket_url": WEBSOCKET_URL if session_id else None,
        "hitl_enabled": True,
        "config": config.dict()
    }
