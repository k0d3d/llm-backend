"""
Tests for HITL form-based workflow functionality

This test suite covers the form-based HITL workflow features:
1. URL extraction and prompt cleaning
2. Form initialization with field classification
3. Attachment handling and mapping
4. Form pre-population from user input
5. HITL skip logic when form is complete
6. Integration tests for end-to-end flows
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep, HITLState
from llm_backend.core.types.common import RunInput, OperationType
from llm_backend.providers.replicate_provider import ReplicateProvider
from llm_backend.agents.form_field_classifier import (
    FormClassificationOutput,
    FieldClassification,
    FieldCategory
)


# Test Fixtures

@pytest.fixture
def mock_state_manager():
    """Mock state manager for testing"""
    manager = Mock()
    manager.save_state = AsyncMock()
    manager.load_state = AsyncMock()
    manager.delete_state = AsyncMock()
    return manager


@pytest.fixture
def mock_websocket_bridge():
    """Mock WebSocket bridge for testing"""
    bridge = Mock()
    bridge.request_human_approval = AsyncMock(return_value={
        "action": "approve",
        "timestamp": datetime.utcnow().isoformat()
    })
    bridge.send_status_update = AsyncMock()
    bridge.send_thinking_status = AsyncMock()
    bridge.send_step_completion = AsyncMock()
    bridge.send_error_notification = AsyncMock()
    return bridge


@pytest.fixture
def replicate_provider():
    """Mock Replicate provider with example_input"""
    config = {
        "name": "test-image-model",
        "description": "Test image editing model",
        "latest_version": "test123",
        "example_input": {
            "prompt": "Demo prompt",
            "image_input": [
                "https://replicate.delivery/demo/image1.jpg",
                "https://replicate.delivery/demo/image2.jpg"
            ],
            "output_format": "jpg"
        }
    }
    provider = ReplicateProvider(config)

    # Mock the execute method
    provider.execute = Mock(return_value={
        "id": "test-prediction-123",
        "status": "succeeded",
        "output": ["https://replicate.delivery/output/result.jpg"]
    })

    return provider


@pytest.fixture
def sample_classification():
    """Sample field classification for testing"""
    return FormClassificationOutput(
        field_classifications={
            "prompt": FieldClassification(
                field_name="prompt",
                category=FieldCategory.CONTENT,
                reset=True,
                required=True,
                default_value=None,
                user_prompt="Enter your prompt",
                collection=False,
                value_type="string"
            ),
            "image_input": FieldClassification(
                field_name="image_input",
                category=FieldCategory.CONTENT,
                reset=True,
                required=False,
                default_value=[],
                user_prompt="Upload images",
                collection=True,
                value_type="array"
            ),
            "output_format": FieldClassification(
                field_name="output_format",
                category=FieldCategory.CONFIG,
                reset=False,
                required=False,
                default_value="jpg",
                user_prompt="Select output format",
                collection=False,
                value_type="string"
            )
        },
        reasoning="Classified based on field types and usage patterns",
        required_fields=["prompt"],
        optional_fields=["image_input", "output_format"]
    )


@pytest.fixture
def hitl_config():
    """Default HITL configuration"""
    return HITLConfig(
        require_human_approval=True,
        checkpoint_information_review=True,
        checkpoint_payload_review=True,
        checkpoint_response_review=False,
        timeout_seconds=300
    )


@pytest_asyncio.fixture
async def orchestrator(replicate_provider, hitl_config, mock_state_manager, mock_websocket_bridge):
    """Create orchestrator with mocked dependencies"""
    run_input = RunInput(
        prompt="Test prompt",
        user_email="test@example.com",
        user_id="user123",
        agent_email="agent@example.com",
        session_id="session123",
        message_type="user_message",
        agent_tool_config={"model": "test-image-model"}
    )

    orchestrator = HITLOrchestrator(
        provider=replicate_provider,
        config=hitl_config,
        run_input=run_input,
        state_manager=mock_state_manager,
        websocket_bridge=mock_websocket_bridge
    )

    return orchestrator


# Test Classes

class TestURLExtraction:
    """Test URL extraction and prompt cleaning"""

    def test_extract_single_url_from_prompt(self, orchestrator):
        """Test extracting a single URL from prompt text"""
        prompt = "Add a character https://replicate.delivery/test/image.jpg"

        cleaned_prompt, extracted_urls = orchestrator._extract_and_clean_urls_from_prompt(prompt)

        assert cleaned_prompt == "Add a character"
        assert len(extracted_urls) == 1
        assert extracted_urls[0] == "https://replicate.delivery/test/image.jpg"

    def test_extract_multiple_urls_from_prompt(self, orchestrator):
        """Test extracting multiple URLs from prompt text"""
        prompt = "Compare https://example.com/img1.jpg and https://example.com/img2.jpg"

        cleaned_prompt, extracted_urls = orchestrator._extract_and_clean_urls_from_prompt(prompt)

        assert cleaned_prompt == "Compare and"
        assert len(extracted_urls) == 2
        assert "https://example.com/img1.jpg" in extracted_urls
        assert "https://example.com/img2.jpg" in extracted_urls

    def test_clean_prompt_removes_urls(self, orchestrator):
        """Test that URLs are completely removed from prompt"""
        prompt = "Process this image: https://serve-dev.tohju.com/test.jpg for me"

        cleaned_prompt, _ = orchestrator._extract_and_clean_urls_from_prompt(prompt)

        assert "https://" not in cleaned_prompt
        assert "serve-dev.tohju.com" not in cleaned_prompt
        assert cleaned_prompt == "Process this image: for me"

    def test_no_urls_in_prompt_returns_unchanged(self, orchestrator):
        """Test prompt without URLs remains unchanged"""
        prompt = "Just a regular prompt with no URLs"

        cleaned_prompt, extracted_urls = orchestrator._extract_and_clean_urls_from_prompt(prompt)

        assert cleaned_prompt == prompt
        assert len(extracted_urls) == 0

    def test_url_with_punctuation_cleaned_correctly(self, orchestrator):
        """Test URLs with trailing punctuation are cleaned"""
        prompt = "Check this out: https://example.com/image.jpg."

        cleaned_prompt, extracted_urls = orchestrator._extract_and_clean_urls_from_prompt(prompt)

        assert extracted_urls[0] == "https://example.com/image.jpg"
        assert not extracted_urls[0].endswith(".")

    def test_empty_prompt_handled_gracefully(self, orchestrator):
        """Test empty or None prompt is handled without error"""
        cleaned_prompt1, urls1 = orchestrator._extract_and_clean_urls_from_prompt("")
        cleaned_prompt2, urls2 = orchestrator._extract_and_clean_urls_from_prompt(None)

        assert cleaned_prompt1 == ""
        assert len(urls1) == 0
        assert cleaned_prompt2 is None
        assert len(urls2) == 0


class TestFormInitialization:
    """Test form initialization from example_input"""

    @pytest.mark.asyncio
    async def test_form_initialized_with_example_input(self, orchestrator, sample_classification):
        """Test that form is initialized from provider's example_input"""
        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            result = await orchestrator._step_form_initialization()

            assert result["continue"] is True
            assert orchestrator.state.form_data is not None
            assert "schema" in orchestrator.state.form_data
            assert "classification" in orchestrator.state.form_data
            assert "current_values" in orchestrator.state.form_data

    @pytest.mark.asyncio
    async def test_form_arrays_reset_to_empty(self, orchestrator, sample_classification):
        """Test that array fields are reset to empty regardless of category"""
        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()

            current_values = orchestrator.state.form_data["current_values"]

            # image_input is an array and should be empty
            assert "image_input" in current_values
            assert current_values["image_input"] == []

    @pytest.mark.asyncio
    async def test_form_config_fields_keep_defaults(self, orchestrator, sample_classification):
        """Test that CONFIG fields keep their default values"""
        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()

            current_values = orchestrator.state.form_data["current_values"]

            # output_format is CONFIG and should keep default
            assert current_values["output_format"] == "jpg"


class TestAttachmentHandling:
    """Test attachment gathering and mapping"""

    @pytest.mark.asyncio
    async def test_user_prompt_only_no_attachments(self, replicate_provider, hitl_config,
                                                    mock_state_manager, mock_websocket_bridge):
        """Test user provides only prompt, no attachments"""
        run_input = RunInput(
            prompt="Just a simple prompt",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        attachments = orchestrator._gather_user_supplied_attachments()

        assert len(attachments) == 0

    @pytest.mark.asyncio
    async def test_user_prompt_with_embedded_url(self, replicate_provider, hitl_config,
                                                  mock_state_manager, mock_websocket_bridge):
        """Test URL embedded in prompt is extracted"""
        run_input = RunInput(
            prompt="Process https://example.com/image.jpg",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        cleaned_prompt, extracted_urls = orchestrator._extract_and_clean_urls_from_prompt(run_input.prompt)

        assert cleaned_prompt == "Process"
        assert len(extracted_urls) == 1
        assert extracted_urls[0] == "https://example.com/image.jpg"

    def test_no_example_urls_in_user_attachments(self, orchestrator):
        """Test that example URLs from config don't appear in user attachments"""
        # _gather_user_supplied_attachments should NOT return example URLs
        user_attachments = orchestrator._gather_user_supplied_attachments()

        # Should be empty since no explicit user attachments
        assert len(user_attachments) == 0

        # Example URLs should not be in the list
        for url in user_attachments:
            assert "replicate.delivery/demo" not in url


class TestFormPrePopulation:
    """Test form pre-population with user input"""

    @pytest.mark.asyncio
    async def test_prompt_prepopulated_without_urls(self, replicate_provider, hitl_config,
                                                     mock_state_manager, mock_websocket_bridge,
                                                     sample_classification):
        """Test that prompt is pre-populated with URLs removed"""
        run_input = RunInput(
            prompt="Add character https://example.com/img.jpg",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()

            current_values = orchestrator.state.form_data["current_values"]

            # Prompt should be cleaned (URL removed)
            assert current_values["prompt"] == "Add character"
            assert "https://" not in current_values["prompt"]

    @pytest.mark.asyncio
    async def test_attachments_mapped_to_correct_fields(self, replicate_provider, hitl_config,
                                                        mock_state_manager, mock_websocket_bridge,
                                                        sample_classification):
        """Test that extracted URLs are mapped to attachment fields"""
        run_input = RunInput(
            prompt="Edit this https://example.com/image.jpg",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()

            current_values = orchestrator.state.form_data["current_values"]

            # URL should be in image_input array
            assert "image_input" in current_values
            assert len(current_values["image_input"]) == 1
            assert current_values["image_input"][0] == "https://example.com/image.jpg"


class TestHITLSkipLogic:
    """Test HITL pause/skip logic based on form completeness"""

    @pytest.mark.asyncio
    async def test_skip_hitl_when_all_required_fields_filled(self, replicate_provider, hitl_config,
                                                             mock_state_manager, mock_websocket_bridge,
                                                             sample_classification):
        """Test that HITL is skipped when all required fields are filled"""
        run_input = RunInput(
            prompt="Process this image",  # Fills required 'prompt' field
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            # Initialize form
            await orchestrator._step_form_initialization()

            # Information review should skip
            result = await orchestrator._step_information_review()

            # Should continue without pausing
            assert result["continue"] is True

            # Should not have requested approval
            mock_websocket_bridge.request_human_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_with_prompt_and_embedded_url(self, replicate_provider, hitl_config,
                                                      mock_state_manager, mock_websocket_bridge,
                                                      sample_classification):
        """Test HITL skip when user provides prompt with embedded URL"""
        run_input = RunInput(
            prompt="Add character https://example.com/img.jpg",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()
            result = await orchestrator._step_information_review()

            # Should skip since prompt (required) is filled
            assert result["continue"] is True
            mock_websocket_bridge.request_human_approval.assert_not_called()


class TestFormWorkflowIntegration:
    """Integration tests for complete form workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_prompt_with_url(self, replicate_provider, hitl_config,
                                               mock_state_manager, mock_websocket_bridge,
                                               sample_classification):
        """Test complete workflow: user provides prompt with embedded URL"""
        run_input = RunInput(
            prompt="Add a cat https://replicate.delivery/user/image.jpg",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            # Step 1: Form initialization
            await orchestrator._step_form_initialization()

            form_data = orchestrator.state.form_data
            current_values = form_data["current_values"]

            # Verify form state
            assert current_values["prompt"] == "Add a cat"  # URL removed
            assert current_values["image_input"] == ["https://replicate.delivery/user/image.jpg"]  # URL mapped
            assert current_values["output_format"] == "jpg"  # Default kept

            # Step 2: Information review (should skip)
            result = await orchestrator._step_information_review()
            assert result["continue"] is True

            # Step 3: Verify no example URLs leaked
            for value in current_values.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and "http" in item:
                            assert "replicate.delivery/demo" not in item

    @pytest.mark.asyncio
    async def test_form_data_used_in_payload_creation(self, replicate_provider, hitl_config,
                                                       mock_state_manager, mock_websocket_bridge,
                                                       sample_classification):
        """Test that form data is used for payload creation, not example URLs"""
        run_input = RunInput(
            prompt="Simple prompt",
            user_email="test@example.com",
            user_id="user123",
            agent_email="agent@example.com",
            session_id="session123",
            message_type="user_message",
            agent_tool_config={"model": "test-image-model"}
        )

        orchestrator = HITLOrchestrator(
            provider=replicate_provider,
            config=hitl_config,
            run_input=run_input,
            state_manager=mock_state_manager,
            websocket_bridge=mock_websocket_bridge
        )

        with patch('llm_backend.agents.form_field_classifier.classify_form_fields',
                   new_callable=AsyncMock, return_value=sample_classification):

            await orchestrator._step_form_initialization()

            # Extract attachments for payload creation
            if orchestrator.state.form_data:
                current_values = orchestrator.state.form_data.get("current_values", {})
                form_attachments = []

                for field_name, value in current_values.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        form_attachments.extend(value)

                # Should be empty since user provided no attachments
                assert len(form_attachments) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
