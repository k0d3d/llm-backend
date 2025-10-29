"""
Integration tests for natural language HITL agents
Tests both NL prompt generator and NL response parser with real OpenAI calls
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pytestmark = pytest.mark.asyncio


class TestNLPromptGenerator:
    """Test the natural language prompt generator agent"""

    async def test_generate_prompt_for_flux(self):
        """Test generating NL prompt for Flux text-to-image model"""
        from llm_backend.agents.nl_prompt_generator import generate_natural_language_prompt

        # Simulate form_field_classifier output
        classification = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True,
                    "default_value": None,
                    "user_prompt": "Describe what image you want to create"
                },
                "aspect_ratio": {
                    "category": "CONFIG",
                    "value_type": "string",
                    "required": False,
                    "default_value": "16:9",
                    "user_prompt": "Choose aspect ratio"
                },
                "num_outputs": {
                    "category": "CONFIG",
                    "value_type": "integer",
                    "required": False,
                    "default_value": 1,
                    "user_prompt": "Number of images to generate"
                }
            }
        }

        # Test: Missing required prompt
        current_values = {
            "aspect_ratio": "16:9",
            "num_outputs": 1
        }

        result = await generate_natural_language_prompt(
            classification=classification,
            current_values=current_values,
            model_name="flux-dev",
            model_description="A text-to-image generation model"
        )

        # Verify
        assert result is not None
        assert not result.all_fields_satisfied
        assert "prompt" in result.missing_field_names
        assert len(result.message) > 0
        print(f"✅ Generated prompt: {result.message}")

    async def test_auto_skip_when_all_fields_present(self):
        """Test that generator recognizes when all fields are satisfied"""
        from llm_backend.agents.nl_prompt_generator import generate_natural_language_prompt

        classification = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True,
                    "default_value": None,
                    "user_prompt": "Describe what image you want"
                }
            }
        }

        # All required fields present
        current_values = {
            "prompt": "A photo of a dog"
        }

        result = await generate_natural_language_prompt(
            classification=classification,
            current_values=current_values,
            model_name="flux-dev",
            model_description="Text-to-image model"
        )

        # Verify auto-skip
        assert result.all_fields_satisfied == True
        assert len(result.missing_field_names) == 0
        assert "everything" in result.message.lower() or "perfect" in result.message.lower()
        print(f"✅ Auto-skip message: {result.message}")


class TestNLResponseParser:
    """Test the natural language response parser agent"""

    async def test_parse_simple_prompt(self):
        """Test parsing a simple prompt"""
        from llm_backend.agents.nl_response_parser import parse_natural_language_response

        expected_schema = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True
                }
            }
        }

        user_message = "A photo of a sunset over mountains"

        result = await parse_natural_language_response(
            user_message=user_message,
            expected_schema=expected_schema,
            current_values={},
            model_description="Text-to-image model"
        )

        # Verify
        assert result is not None
        assert "prompt" in result.extracted_fields
        assert "sunset" in result.extracted_fields["prompt"].lower()
        assert result.confidence > 0.7
        print(f"✅ Extracted: {result.extracted_fields}")
        print(f"   Confidence: {result.confidence}")

    async def test_parse_complex_response(self):
        """Test parsing a complex response with multiple fields"""
        from llm_backend.agents.nl_response_parser import parse_natural_language_response

        expected_schema = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True
                },
                "aspect_ratio": {
                    "category": "CONFIG",
                    "value_type": "string",
                    "required": False
                },
                "num_outputs": {
                    "category": "CONFIG",
                    "value_type": "integer",
                    "required": False
                }
            }
        }

        user_message = "Create a photo of a sunset over mountains in 4:3 format, I need 3 variations"

        result = await parse_natural_language_response(
            user_message=user_message,
            expected_schema=expected_schema,
            current_values={},
            model_description="Text-to-image model"
        )

        # Verify
        assert result is not None
        assert "prompt" in result.extracted_fields
        assert "sunset" in result.extracted_fields["prompt"].lower()

        # Check if aspect_ratio was extracted
        if "aspect_ratio" in result.extracted_fields:
            assert result.extracted_fields["aspect_ratio"] == "4:3"
            print("✅ Extracted aspect_ratio: 4:3")

        # Check if num_outputs was extracted
        if "num_outputs" in result.extracted_fields:
            assert result.extracted_fields["num_outputs"] == 3
            print("✅ Extracted num_outputs: 3")

        assert result.confidence > 0.7
        print(f"✅ Full extraction: {result.extracted_fields}")
        print(f"   Confidence: {result.confidence}")

    async def test_parse_with_natural_language_numbers(self):
        """Test parsing natural language numbers like 'three variations'"""
        from llm_backend.agents.nl_response_parser import parse_natural_language_response

        expected_schema = {
            "field_classifications": {
                "prompt": {"category": "CONTENT", "value_type": "string", "required": True},
                "num_outputs": {"category": "CONFIG", "value_type": "integer", "required": False}
            }
        }

        user_message = "A dog, give me three variations"

        result = await parse_natural_language_response(
            user_message=user_message,
            expected_schema=expected_schema,
            current_values={},
            model_description="Image generation"
        )

        # Verify
        assert "prompt" in result.extracted_fields
        if "num_outputs" in result.extracted_fields:
            assert result.extracted_fields["num_outputs"] == 3
            print("✅ Parsed 'three' → 3")

    async def test_fallback_parsing(self):
        """Test that fallback parsing works when AI fails"""
        from llm_backend.agents.nl_response_parser import _fallback_response_parsing

        field_classifications = {
            "prompt": {
                "category": "CONTENT",
                "value_type": "string",
                "required": True
            },
            "aspect_ratio": {
                "category": "CONFIG",
                "value_type": "string",
                "required": False
            }
        }

        user_message = "A sunset in 16:9 format"

        result = _fallback_response_parsing(
            user_message=user_message,
            field_classifications=field_classifications,
            current_values={}
        )

        # Verify fallback extraction
        assert "prompt" in result.extracted_fields
        assert result.extracted_fields["prompt"] == "A sunset in 16:9 format"

        # Check if aspect_ratio was extracted by heuristics
        if "aspect_ratio" in result.extracted_fields:
            assert result.extracted_fields["aspect_ratio"] == "16:9"
            print("✅ Fallback extracted aspect_ratio: 16:9")

        print(f"✅ Fallback extraction: {result.extracted_fields}")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping NL agent tests"
)
class TestNLAgentsOpenAIConnectivity:
    """Verify NL agents can connect to OpenAI"""

    async def test_openai_key_available(self):
        """Test that OpenAI API key is available"""
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None
        assert api_key.startswith("sk-")
        print("✅ OpenAI API key available for NL agents")

    async def test_prompt_generator_with_openai(self):
        """Test prompt generator makes successful OpenAI call"""
        from llm_backend.agents.nl_prompt_generator import generate_natural_language_prompt

        classification = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True,
                    "user_prompt": "What do you want to create?"
                }
            }
        }

        result = await generate_natural_language_prompt(
            classification=classification,
            current_values={},
            model_name="test-model",
            model_description="A test model"
        )

        assert result is not None
        assert len(result.message) > 0
        print(f"✅ Prompt generator OpenAI call succeeded: {result.message}")

    async def test_response_parser_with_openai(self):
        """Test response parser makes successful OpenAI call"""
        from llm_backend.agents.nl_response_parser import parse_natural_language_response

        expected_schema = {
            "field_classifications": {
                "prompt": {
                    "category": "CONTENT",
                    "value_type": "string",
                    "required": True
                }
            }
        }

        result = await parse_natural_language_response(
            user_message="A beautiful landscape",
            expected_schema=expected_schema,
            current_values={},
            model_description="Test model"
        )

        assert result is not None
        assert len(result.extracted_fields) > 0
        print(f"✅ Response parser OpenAI call succeeded: {result.extracted_fields}")


if __name__ == "__main__":
    """Run NL agent tests standalone"""
    import asyncio

    print("🧪 Running Natural Language Agent Tests...\\n")

    # Test 1: Prompt generator
    print("1. Testing NL prompt generator...")
    test = TestNLPromptGenerator()
    try:
        asyncio.run(test.test_generate_prompt_for_flux())
    except Exception as e:
        print(f"   ❌ Failed: {e}\\n")

    # Test 2: Auto-skip
    print("\\n2. Testing auto-skip detection...")
    try:
        asyncio.run(test.test_auto_skip_when_all_fields_present())
    except Exception as e:
        print(f"   ❌ Failed: {e}\\n")

    # Test 3: Response parser simple
    print("\\n3. Testing NL response parser (simple)...")
    test = TestNLResponseParser()
    try:
        asyncio.run(test.test_parse_simple_prompt())
    except Exception as e:
        print(f"   ❌ Failed: {e}\\n")

    # Test 4: Response parser complex
    print("\\n4. Testing NL response parser (complex)...")
    try:
        asyncio.run(test.test_parse_complex_response())
    except Exception as e:
        print(f"   ❌ Failed: {e}\\n")

    # Test 5: Fallback parsing
    print("\\n5. Testing fallback parsing...")
    try:
        asyncio.run(test.test_fallback_parsing())
    except Exception as e:
        print(f"   ❌ Failed: {e}\\n")

    print("\\n✅ Natural Language Agent test suite completed!")
