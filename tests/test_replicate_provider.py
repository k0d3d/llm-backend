"""Tests for ReplicateProvider functionality"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from llm_backend.core.types.common import RunInput, OperationType
from llm_backend.providers.replicate_provider import ReplicateProvider, ReplicatePayload

@pytest.fixture
def provider_config():
    return {
        "name": "flux-kontext-pro",
        "description": "Image editing model",
        "latest_version": "test123",
        "example_input": {
            "prompt": "Make this a cartoon",
            "input_image": "https://example.com/test.jpg",
            "aspect_ratio": "match_input_image"
        }
    }

@pytest.fixture
def run_input():
    return RunInput(
        prompt="Add a cat",
        user_email="test@example.com",
        user_id="123",
        agent_email="agent@example.com",
        session_id="test-session",
        message_type="user_message",
        agent_tool_config={"model": "flux-kontext-pro"}
    )

@pytest.fixture
def provider(provider_config, run_input):
    provider = ReplicateProvider(provider_config)
    provider.set_run_input(run_input)
    return provider

def test_normalize_attachments(provider):
    """Test attachment URL normalization"""
    attachments = [
        "https://serve-dev.tohju.com/test.jpg",
        "https://replicate.delivery/test.png",
        "https://example.com/other.jpg"
    ]
    normalized = provider._normalize_attachments(attachments)
    assert len(normalized) == 3
    assert all(url in normalized for url in attachments)

def test_strip_attachment_mentions(provider):
    """Test removal of attachment URLs from prompt"""
    url = "https://example.com/test.jpg"
    prompt = f"Add a cat [image]({url}) to this photo {url}"
    clean_prompt = provider._strip_attachment_mentions(prompt, [url])
    assert url not in clean_prompt
    assert "[image]" not in clean_prompt
    assert clean_prompt.strip() == "Add a cat to this photo"

def test_create_payload_with_attachments(provider):
    """Test payload creation with image attachments"""
    prompt = "Add a cat"
    attachments = ["https://serve-dev.tohju.com/test.jpg"]
    payload = provider.create_payload(
        prompt=prompt,
        attachments=attachments,
        operation_type=OperationType.IMAGE_EDITING,
        config={},
        hitl_edits=None
    )
    assert isinstance(payload, ReplicatePayload)
    assert payload.input.get("input_image") == attachments[0]
    assert payload.input.get("prompt") == prompt

def test_create_payload_with_hitl_edits(provider):
    """Test payload creation with HITL edits"""
    prompt = "Add a cat"
    attachments = ["https://serve-dev.tohju.com/test.jpg"]
    hitl_edits = {
        "prompt": "Add two cats",
        "aspect_ratio": "square"
    }
    payload = provider.create_payload(
        prompt=prompt,
        attachments=attachments,
        operation_type=OperationType.IMAGE_EDITING,
        config={},
        hitl_edits=hitl_edits
    )
    assert isinstance(payload, ReplicatePayload)
    assert payload.input.get("prompt") == hitl_edits["prompt"]
    assert payload.input.get("aspect_ratio") == hitl_edits["aspect_ratio"]

def test_resume_with_dict_input(provider_config):
    """Test provider initialization with dict run input"""
    dict_input = {
        "prompt": "Add a cat",
        "user_email": "test@example.com",
        "user_id": "123",
        "agent_email": "agent@example.com",
        "session_id": "test-session",
        "message_type": "user_message",
        "agent_tool_config": {"model": "flux-kontext-pro"}
    }
    provider = ReplicateProvider(provider_config)
    provider.set_run_input(dict_input)
    assert provider.run_input is not None
    assert hasattr(provider.run_input, "prompt")
    assert provider.run_input.prompt == dict_input["prompt"]

@pytest.mark.asyncio
async def test_hitl_resume_flow():
    """Test complete HITL resume flow with attachments"""
    from llm_backend.core.hitl.orchestrator import HITLOrchestrator
    from llm_backend.core.hitl.types import HITLConfig
    
    # Setup test state
    run_id = "test-run-123"
    provider_config = {
        "name": "flux-kontext-pro",
        "description": "Image editing model",
        "latest_version": "test123",
        "example_input": {"prompt": "test"}
    }
    
    # Mock state manager with async methods
    mock_state = {
        "run_id": run_id,
        "original_input": {
            "prompt": "Add a cat",
            "user_id": "123",
            "attachments": ["https://example.com/test.jpg"]
        }
    }

    state_manager = MagicMock()
    save_state_mock = MagicMock()
    save_state_mock.__call__ = AsyncMock(return_value=True)
    state_manager.save_state = save_state_mock

    load_state_mock = MagicMock()
    load_state_mock.__call__ = AsyncMock(return_value=mock_state)
    state_manager.load_state = load_state_mock
    
    # Create orchestrator
    provider = ReplicateProvider(provider_config)
    # Initialize orchestrator with mock state
    orchestrator = HITLOrchestrator(
        provider=provider,
        config=HITLConfig(timeout_seconds=300),
        run_input=mock_state["original_input"],
        state_manager=state_manager
    )
    
    # Simulate resume
    await orchestrator.resume_from_state()
    
    # Verify provider received correct input
    assert provider.run_input is not None
    assert "prompt" in provider.run_input
    assert provider.run_input["prompt"] == "Add a cat"
