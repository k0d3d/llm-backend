"""Integration tests for HITL workflow with multi-shot examples"""

import pytest
from unittest.mock import MagicMock, patch
from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLState
from llm_backend.core.types.common import RunInput
from llm_backend.providers.replicate_provider import ReplicateProvider

# Multi-shot examples for testing image editing flows
IMAGE_EDIT_EXAMPLES = [
    {
        "prompt": "Add a cat to this photo",
        "attachments": ["https://serve-dev.tohju.com/test1.jpg"],
        "expected_edits": None,
        "expected_output": {
            "input_image": "https://serve-dev.tohju.com/test1.jpg",
            "prompt": "Add a cat to this photo"
        }
    },
    {
        "prompt": "Make this more vibrant",
        "attachments": ["https://replicate.delivery/test2.jpg"],
        "expected_edits": {
            "prompt": "Increase color saturation and vibrancy",
            "strength": 0.8
        },
        "expected_output": {
            "input_image": "https://replicate.delivery/test2.jpg",
            "prompt": "Increase color saturation and vibrancy",
            "strength": 0.8
        }
    }
]

@pytest.fixture
def mock_state_manager():
    manager = MagicMock()
    manager.save_state = MagicMock()
    manager.load_state = MagicMock()
    return manager

@pytest.fixture
def mock_websocket_bridge():
    bridge = MagicMock()
    bridge.request_human_approval = MagicMock()
    return bridge

@pytest.mark.parametrize("example", IMAGE_EDIT_EXAMPLES)
@pytest.mark.asyncio
async def test_hitl_image_edit_flow(example, mock_state_manager, mock_websocket_bridge):
    """Test HITL flow with various image editing scenarios"""
    
    # Setup provider
    provider_config = {
        "name": "flux-kontext-pro",
        "description": "Image editing model",
        "latest_version": "test123",
        "example_input": {
            "prompt": "Sample edit",
            "input_image": "https://example.com/test.jpg",
            "strength": 0.5
        }
    }
    provider = ReplicateProvider(provider_config)
    
    # Create run input
    run_input = RunInput(
        prompt=example["prompt"],
        user_email="test@example.com",
        user_id="test123",
        agent_email="agent@example.com",
        session_id="test-session",
        message_type="user_message",
        agent_tool_config={"model": "flux-kontext-pro"}
    )
    
    # Initialize orchestrator
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=HITLConfig(timeout_seconds=300),
        run_input=run_input,
        state_manager=mock_state_manager,
        websocket_bridge=mock_websocket_bridge
    )
    
    # Setup mock state for resume
    mock_state = HITLState(
        run_id=orchestrator.run_id,
        config=orchestrator.config,
        original_input={
            "prompt": example["prompt"],
            "attachments": example["attachments"]
        }
    )
    mock_state_manager.load_state.return_value = mock_state
    
    # Mock approval response
    mock_websocket_bridge.request_human_approval.return_value = {
        "action": "approve",
        "edits": example["expected_edits"],
        "approved_by": "test_user"
    }
    
    # Start HITL flow
    await orchestrator.start_run(
        original_input={"attachments": example["attachments"]},
        user_id="test123",
        session_id="test-session"
    )
    
    # Execute and verify
    result = await orchestrator.execute()
    
    # Verify payload matches expected output
    assert isinstance(result, dict)
    payload = result.get("payload")
    assert payload is not None
    
    for key, value in example["expected_output"].items():
        assert payload.get(key) == value

@pytest.mark.asyncio
async def test_hitl_resume_with_attachments(mock_state_manager):
    """Test resuming HITL flow with attachments after approval"""
    
    run_id = "test-resume-123"
    mock_state = {
        "run_id": run_id,
        "original_input": {
            "prompt": "Add a cat",
            "attachments": ["https://serve-dev.tohju.com/test.jpg"],
            "user_id": "123",
            "session_id": "test-session"
        },
        "config": HITLConfig(timeout_seconds=300).dict(),
        "human_edits": {
            "prompt": "Add two cats",
            "strength": 0.7
        }
    }
    mock_state_manager.load_state.return_value = mock_state
    
    provider = ReplicateProvider({
        "name": "flux-kontext-pro",
        "latest_version": "test123",
        "example_input": {"prompt": "test"}
    })
    
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=HITLConfig(timeout_seconds=300),
        run_input=mock_state["original_input"],
        state_manager=mock_state_manager
    )
    
    # Resume and verify
    await orchestrator.resume_from_state()
    
    # Check that provider received correct input
    assert provider.run_input is not None
    assert provider.run_input.get("prompt") == "Add a cat"
    assert provider.run_input.get("attachments") == ["https://serve-dev.tohju.com/test.jpg"]
    
    # Verify human edits were applied
    payload = await orchestrator.execute()
    assert payload["input"]["prompt"] == "Add two cats"
    assert payload["input"]["strength"] == 0.7
