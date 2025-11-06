"""Worker tasks for RQ"""
import os
import asyncio
import logging
from datetime import datetime
from rq import get_current_job
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus
from llm_backend.core.hitl.shared_bridge import get_shared_state_manager, get_shared_websocket_bridge
from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.core.types.common import RunInput

# Get shared components
state_manager = get_shared_state_manager()
websocket_bridge = get_shared_websocket_bridge()

logger = logging.getLogger(__name__)

def run_async(coro):
    """Helper to run async function in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)

def process_hitl_orchestrator(run_input_dict: dict, hitl_config_dict: dict, provider_name: str):
    """
    Process HITL orchestrator execution
    Args:
        run_input_dict: Serialized RunInput dictionary
        hitl_config_dict: Serialized HITLConfig dictionary
        provider_name: Provider name (e.g., "replicate")
    Returns:
        Final result
    """
    job = get_current_job()
    job.meta['status'] = 'processing'
    job.save_meta()

    logger.info("Starting HITL orchestrator for session: %s", run_input_dict.get('session_id'))

    # Log received data for debugging
    logger.debug("run_input_dict keys: %s", list(run_input_dict.keys()))
    if 'agent_tool_config' in run_input_dict and logger.isEnabledFor(logging.DEBUG):
        tool_config_data = run_input_dict['agent_tool_config']
        logger.debug("agent_tool_config keys: %s", list(tool_config_data.keys()) if tool_config_data else None)
        if tool_config_data:
            for key, val in tool_config_data.items():
                if isinstance(val, dict) and 'data' in val:
                    logger.debug("%s['data'] has example_input: %s", key, bool(val['data'].get('example_input')))

    # Reconstruct objects
    run_input = RunInput(**run_input_dict)
    hitl_config = HITLConfig(**hitl_config_dict)

    # FIX: Extract tool_config from run_input to pass to provider
    tool_config = {}
    if run_input.agent_tool_config:
        from llm_backend.core.types.common import AgentTools
        replicate_tool_config = run_input.agent_tool_config.get(AgentTools.REPLICATETOOL)
        if not replicate_tool_config:
            # Try alternative key formats
            for key in run_input.agent_tool_config.keys():
                if "replicate" in str(key).lower():
                    replicate_tool_config = run_input.agent_tool_config.get(key)
                    break

        if replicate_tool_config and isinstance(replicate_tool_config, dict):
            # Handle both flat and nested data formats
            # Format 1 (flat): {'name': 'flux-1.1-pro', 'example_input': {...}}
            # Format 2 (nested): {'data': {'name': 'nano-banana', 'example_input': {...}}, 'name': 'replicate-agent-tool'}
            if 'data' in replicate_tool_config and isinstance(replicate_tool_config.get('data'), dict) and replicate_tool_config['data']:
                tool_config = replicate_tool_config['data']
                logger.debug("Extracted config from nested 'data' key")
            else:
                tool_config = replicate_tool_config
                logger.debug("Using flat config structure")

    logger.debug(
        "Extracted tool_config: name=%s, has_example_input=%s",
        tool_config.get('name', 'MISSING'),
        bool(tool_config.get('example_input'))
    )

    # Create provider with proper config
    provider = ProviderRegistry.get_provider(provider_name, config=tool_config)
    logger.debug("Provider created with config keys: %s", list(provider.config.keys()))

    # Create orchestrator
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=hitl_config,
        run_input=run_input,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )

    # Load existing state if this is a resume
    run_id = run_input_dict.get('run_id')
    if run_id:
        state = run_async(state_manager.load_state(run_id))
        if state:
            orchestrator.state = state
            logger.info("Loaded existing state for run %s", run_id)

    # Execute orchestrator
    logger.info("Executing HITL orchestrator...")
    result = run_async(orchestrator.execute())

    logger.info("Orchestrator completed with status: %s", orchestrator.state.status)

    job.meta['status'] = 'completed'
    job.meta['run_id'] = orchestrator.state.run_id
    job.save_meta()

    return {
        "run_id": orchestrator.state.run_id,
        "status": orchestrator.state.status,
        "result": result
    }

def process_hitl_resume(run_id: str, approval_response: dict):
    """
    Process HITL resume after approval
    Args:
        run_id: Run identifier
        approval_response: Approval response dict
    Returns:
        Resume result
    """
    job = get_current_job()
    job.meta['status'] = 'processing'
    job.save_meta()

    logger.info("Resuming HITL run: %s", run_id)

    # Load state
    state = run_async(state_manager.load_state(run_id))

    if not state:
        raise ValueError(f"Run {run_id} not found")

    # Resume the run
    run_async(state_manager.resume_run(run_id, approval_response))

    # Get provider and recreate orchestrator
    provider_name = state.original_input.get("provider", "replicate")
    provider = ProviderRegistry.get_provider(provider_name)

    orchestrator = HITLOrchestrator(
        provider=provider,
        config=state.config,
        run_input=state.original_input,
        state_manager=state_manager,
        websocket_bridge=websocket_bridge
    )

    # Load state with human edits
    orchestrator.state = state
    logger.debug("Loaded state with human_edits: %s", getattr(state, 'human_edits', 'MISSING'))

    # Continue execution
    result = run_async(orchestrator.execute())

    logger.info("Resume completed with status: %s", orchestrator.state.status)

    job.meta['status'] = 'completed'
    job.meta['run_id'] = run_id
    job.save_meta()

    return {
        "run_id": run_id,
        "status": orchestrator.state.status,
        "result": result
    }
