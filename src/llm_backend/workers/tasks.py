"""Worker tasks for RQ"""
import os
import asyncio
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

    print(f"[HITLWorker] Starting HITL orchestrator for session: {run_input_dict.get('session_id')}")

    # Reconstruct objects
    run_input = RunInput(**run_input_dict)
    hitl_config = HITLConfig(**hitl_config_dict)
    provider = ProviderRegistry.get_provider(provider_name)

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
            print(f"[HITLWorker] Loaded existing state for run {run_id}")

    # Execute orchestrator
    print("[HITLWorker] Executing HITL orchestrator...")
    result = run_async(orchestrator.execute())

    print(f"[HITLWorker] Orchestrator completed with status: {orchestrator.state.status}")

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

    print(f"[HITLWorker] Resuming HITL run: {run_id}")

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
    print(f"[HITLWorker] Loaded state with human_edits: {getattr(state, 'human_edits', 'MISSING')}")

    # Continue execution
    result = run_async(orchestrator.execute())

    print(f"[HITLWorker] Resume completed with status: {orchestrator.state.status}")

    job.meta['status'] = 'completed'
    job.meta['run_id'] = run_id
    job.save_meta()

    return {
        "run_id": run_id,
        "status": orchestrator.state.status,
        "result": result
    }
