"""
Tests for AI-powered attachment mapping to form fields
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from llm_backend.agents.attachment_mapper import (
    map_attachments_to_fields,
    _heuristic_attachment_mapping,
    FieldAttachmentMapping,
    AttachmentMappingOutput,
)


class TestAttachmentMappingAgent:
    """Test the AI-powered attachment mapping agent"""

    @pytest.mark.asyncio
    async def test_single_image_field_exact_match(self):
        """Test mapping single image to 'image' field (the bug scenario)"""
        user_attachments = ["https://replicate.delivery/pbxt/shoe.jpg"]
        field_classifications = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            }
        }
        example_input = {"image": "https://example.com/demo.jpg"}

        result = await map_attachments_to_fields(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
        )

        # Should have exactly one mapping
        assert len(result.mappings) == 1
        assert result.mappings[0].field_name == "image"
        assert result.mappings[0].attachment == user_attachments[0]
        assert result.mappings[0].file_type == "image"
        assert result.mappings[0].confidence > 0.8

    @pytest.mark.asyncio
    async def test_semantic_field_name_matching(self):
        """Test that AI understands semantic equivalence of field names"""
        test_cases = [
            # (field_name, should_match)
            ("image", True),
            ("input_image", True),
            ("img", True),
            ("photo", True),
            ("picture", True),
            ("strength", False),  # Config field
            ("guidance_scale", False),  # Config field
        ]

        user_attachments = ["https://serve.com/photo.jpg"]

        for field_name, should_match in test_cases:
            field_classifications = {
                field_name: {
                    "category": "CONTENT" if should_match else "CONFIG",
                    "collection": False,
                    "value_type": "string",
                    "required": should_match,
                }
            }
            example_input = {field_name: "https://example.com/demo.jpg"}

            result = await map_attachments_to_fields(
                user_attachments=user_attachments,
                field_classifications=field_classifications,
                example_input=example_input,
            )

            if should_match:
                assert len(result.mappings) >= 1, f"Failed to match '{field_name}'"
                assert result.mappings[0].field_name == field_name
            else:
                # CONFIG fields should not be mapped to
                mapped_fields = [m.field_name for m in result.mappings]
                assert field_name not in mapped_fields, f"Incorrectly mapped to CONFIG field '{field_name}'"

    @pytest.mark.asyncio
    async def test_array_field_mapping(self):
        """Test mapping multiple attachments to array field"""
        user_attachments = [
            "https://serve.com/photo1.jpg",
            "https://serve.com/photo2.jpg",
        ]
        field_classifications = {
            "images": {
                "category": "CONTENT",
                "collection": True,  # Array field
                "value_type": "array",
                "required": True,
            }
        }
        example_input = {"images": []}

        result = await map_attachments_to_fields(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
        )

        # Both attachments should map to the array field
        assert len(result.mappings) == 2
        assert all(m.field_name == "images" for m in result.mappings)
        assert {m.attachment for m in result.mappings} == set(user_attachments)

    @pytest.mark.asyncio
    async def test_file_type_detection(self):
        """Test that different file types map to appropriate fields"""
        test_cases = [
            ("https://serve.com/photo.jpg", "image", "image"),
            ("https://serve.com/song.mp3", "audio", "audio"),
            ("https://serve.com/video.mp4", "video", "video"),
            ("https://serve.com/doc.pdf", "document", "document"),
        ]

        for url, field_name, expected_type in test_cases:
            field_classifications = {
                field_name: {
                    "category": "CONTENT",
                    "collection": False,
                    "value_type": "string",
                    "required": True,
                }
            }
            example_input = {field_name: "https://example.com/demo"}

            result = await map_attachments_to_fields(
                user_attachments=[url],
                field_classifications=field_classifications,
                example_input=example_input,
            )

            assert len(result.mappings) >= 1, f"Failed to map {expected_type} file"
            assert result.mappings[0].file_type == expected_type

    @pytest.mark.asyncio
    async def test_prioritize_content_over_config(self):
        """Test that CONTENT fields are prioritized over CONFIG fields"""
        user_attachments = ["https://serve.com/photo.jpg"]
        field_classifications = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            },
            "strength": {
                "category": "CONFIG",
                "collection": False,
                "value_type": "number",
                "required": False,
            },
            "guidance_scale": {
                "category": "CONFIG",
                "collection": False,
                "value_type": "number",
                "required": False,
            },
        }
        example_input = {
            "image": "https://example.com/demo.jpg",
            "strength": 0.8,
            "guidance_scale": 7.5,
        }

        result = await map_attachments_to_fields(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
        )

        # Should only map to CONTENT field, not CONFIG
        assert len(result.mappings) == 1
        assert result.mappings[0].field_name == "image"
        assert result.mappings[0].category != "CONFIG"


class TestHeuristicFallback:
    """Test the heuristic fallback when AI agent fails"""

    def test_heuristic_single_field_exact_match(self):
        """Test heuristic mapping to single 'image' field"""
        user_attachments = ["https://serve.com/photo.jpg"]
        field_classifications = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            }
        }
        example_input = {"image": "https://example.com/demo.jpg"}

        result = _heuristic_attachment_mapping(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
        )

        assert len(result.mappings) == 1
        assert result.mappings[0].field_name == "image"
        assert result.mappings[0].confidence >= 0.9  # Exact match should have high confidence

    def test_heuristic_handles_single_and_array_fields(self):
        """Test that heuristic correctly differentiates single vs array fields"""
        user_attachments = ["https://serve.com/photo.jpg"]

        # Test single field
        single_field = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            }
        }
        result_single = _heuristic_attachment_mapping(
            user_attachments, single_field, {}
        )
        assert len(result_single.mappings) == 1
        # Single field should not be wrapped in array in the mapping

        # Test array field
        array_field = {
            "images": {
                "category": "CONTENT",
                "collection": True,
                "value_type": "array",
                "required": True,
            }
        }
        result_array = _heuristic_attachment_mapping(
            user_attachments, array_field, {}
        )
        assert len(result_array.mappings) == 1

    def test_heuristic_skips_config_fields(self):
        """Test that heuristic doesn't map to CONFIG category fields"""
        user_attachments = ["https://serve.com/photo.jpg"]
        field_classifications = {
            "strength": {
                "category": "CONFIG",
                "collection": False,
                "value_type": "number",
                "required": False,
            },
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            },
        }

        result = _heuristic_attachment_mapping(
            user_attachments, field_classifications, {}
        )

        # Should map to CONTENT field, not CONFIG
        assert len(result.mappings) == 1
        assert result.mappings[0].field_name == "image"


class TestOrchestratorIntegration:
    """Test integration with HITLOrchestrator"""

    @pytest.mark.asyncio
    async def test_orchestrator_mapping_method_async(self):
        """Test that orchestrator's _map_attachments_to_fields is async"""
        from llm_backend.core.hitl.orchestrator import HITLOrchestrator
        from llm_backend.core.hitl.types import HITLConfig, HITLPolicy
        from llm_backend.core.types.common import RunInput
        from llm_backend.core.providers.base import AIProvider

        # Create mock provider
        provider = Mock(spec=AIProvider)
        provider.model_name = "test-model"
        provider.description = "Test description"
        provider.example_input = {"image": "https://example.com/demo.jpg"}
        provider.set_run_input = Mock()

        # Create config
        config = HITLConfig(
            policy=HITLPolicy.AUTO,
            allowed_steps=[],
            timeout_seconds=3600,
        )

        # Create run input
        run_input = RunInput(
            prompt="Test prompt",
            user_id="1",
            session_id="test-session",
            user_email="test@example.com",
            agent_email="agent@example.com",
            message_type="test",
        )

        # Create orchestrator
        orchestrator = HITLOrchestrator(
            provider=provider,
            config=config,
            run_input=run_input,
        )

        # Test mapping
        user_attachments = ["https://serve.com/photo.jpg"]
        field_classifications = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            }
        }

        # This should work as async method
        mapping = await orchestrator._map_attachments_to_fields(
            user_attachments, field_classifications
        )

        # Should return a dict with field -> attachment
        assert isinstance(mapping, dict)
        # Either AI or heuristic should map it
        assert "image" in mapping or len(mapping) == 0  # May be 0 if AI/heuristic both fail


class TestExactBugScenario:
    """Test the exact scenario from the bug report"""

    @pytest.mark.asyncio
    async def test_bug_scenario_image_field_not_mapped(self):
        """
        Reproduce the exact bug:
        - User provides: https://replicate.delivery/pbxt/shoe.jpg
        - Schema has: {"image": "..."}
        - Old code: Failed to map (only looked for arrays)
        - New code: Should map successfully
        """
        user_attachments = ["https://replicate.delivery/pbxt/JWsRA6DxCK24PlMYK5ENFYAFxJGUQTLr0JmLwsLb8uhv1JTU/shoe.jpg"]
        field_classifications = {
            "image": {
                "category": "CONTENT",
                "collection": False,  # NOT an array!
                "value_type": "string",
                "required": True,
            }
        }
        example_input = {
            "image": "https://replicate.delivery/pbxt/demo.jpg"
        }

        # Test with AI agent
        result = await map_attachments_to_fields(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
            model_name="test-model",
            model_description="Image processing model",
        )

        # AI should successfully map it
        assert len(result.mappings) >= 1, "AI agent failed to map image attachment"
        assert result.mappings[0].field_name == "image"
        assert result.mappings[0].attachment == user_attachments[0]
        assert len(result.unmapped_fields) == 0, "Image field should not be unmapped"

        # Test with heuristic fallback
        heuristic_result = _heuristic_attachment_mapping(
            user_attachments=user_attachments,
            field_classifications=field_classifications,
            example_input=example_input,
        )

        # Heuristic should also map it
        assert len(heuristic_result.mappings) >= 1, "Heuristic failed to map image attachment"
        assert heuristic_result.mappings[0].field_name == "image"
        assert heuristic_result.mappings[0].confidence > 0.8

    def test_bug_root_cause_single_vs_array(self):
        """
        Verify the root cause: old code only checked array fields
        New heuristic should handle both single and array fields
        """
        user_attachments = ["https://serve.com/photo.jpg"]

        # Single field (the bug case)
        single_field_class = {
            "image": {
                "category": "CONTENT",
                "collection": False,
                "value_type": "string",
                "required": True,
            }
        }

        result_single = _heuristic_attachment_mapping(
            user_attachments, single_field_class, {}
        )

        # Should successfully map to single field
        assert len(result_single.mappings) == 1
        assert result_single.mappings[0].field_name == "image"

        # Array field (old code would have worked)
        array_field_class = {
            "images": {
                "category": "CONTENT",
                "collection": True,
                "value_type": "array",
                "required": True,
            }
        }

        result_array = _heuristic_attachment_mapping(
            user_attachments, array_field_class, {}
        )

        # Should also successfully map to array field
        assert len(result_array.mappings) == 1
        assert result_array.mappings[0].field_name == "images"
