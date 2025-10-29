"""
AI Agent for classifying form fields from example_input into CONTENT vs CONFIG categories
Determines which fields should be reset/emptied vs which should keep defaults
"""

from typing import Dict, Any, List, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from enum import Enum


class FieldCategory(str, Enum):
    """Categories for form field classification"""
    CONTENT = "CONTENT"  # User must provide (images, prompts, etc.)
    CONFIG = "CONFIG"     # Optional parameters with sensible defaults
    HYBRID = "HYBRID"     # Optional content (can be empty or user-provided)


class FieldClassification(BaseModel):
    """Classification for a single field"""
    field_name: str
    category: FieldCategory
    reset: bool = Field(description="Whether to empty this field on form initialization")
    required: bool = Field(description="Whether this field is required for execution")
    default_value: Any = Field(default=None, description="Default value for the field")
    user_prompt: str = Field(description="User-friendly prompt for this field")
    collection: bool = Field(default=False, description="Whether this field is an array/list")
    nested_classification: Optional[Dict[str, Any]] = Field(default=None, description="Classification for nested object fields")
    value_type: str = Field(description="Data type of the field (string, number, boolean, array, object)")


class FormClassificationInput(BaseModel):
    """Input for form field classifier agent"""
    example_input: Dict[str, Any]
    model_name: str
    model_description: str
    field_metadata: Optional[Dict[str, Any]] = None


class FormClassificationOutput(BaseModel):
    """Output from form field classifier agent"""
    field_classifications: Dict[str, FieldClassification]
    reasoning: str
    required_fields: List[str]
    optional_fields: List[str]


# System prompt for the classifier agent
CLASSIFIER_SYSTEM_PROMPT = """You are an expert at analyzing AI model parameters and determining which fields are:
1. CONTENT fields - The actual data the model processes (images, audio, prompts, text). These should be RESET (emptied) on form initialization so users must provide their own data.
2. CONFIG fields - Configuration parameters with sensible defaults (guidance_scale, num_steps, scheduler). These should KEEP their defaults.
3. HYBRID fields - Optional content that can be empty or user-provided (negative_prompt, style_preset).

Key rules:
- ALL arrays/lists should be RESET to empty [] regardless of category
- Media fields (image, video, audio, file URLs) are CONTENT - must be RESET
- User prompt/text input fields (prompt, text, instruction) are CONTENT - must be RESET
- System prompts (system_prompt, system_message) are CONFIG - keep defaults (optional)
- Numeric parameters (scale, steps, strength) are CONFIG - keep defaults
- Enum/choice parameters (scheduler, format, quality) are CONFIG - keep defaults
- Seed values are CONFIG - keep defaults (or null for random)

For nested objects, recursively classify each nested field.

Example input often contains placeholder/demo values. Your job is to identify which fields need real user data (CONTENT) vs which have useful defaults (CONFIG).
"""


def create_classifier_agent() -> Agent[FormClassificationInput, FormClassificationOutput]:
    """Create the form field classifier agent"""

    agent = Agent(
        model="openai:gpt-4o-mini",
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
        output_type=FormClassificationOutput,
        retries=3
    )

    return agent


def infer_value_type(value: Any) -> str:
    """Infer the type of a field value"""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, (list, tuple)):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return "unknown"


def generate_user_prompt(field_name: str, category: FieldCategory, value_type: str) -> str:
    """Generate user-friendly prompt for a field based on its classification"""
    field_lower = field_name.lower()

    # Content field prompts
    if category == FieldCategory.CONTENT:
        if "image" in field_lower:
            return f"Upload an image for {field_name}"
        elif "video" in field_lower:
            return f"Upload a video for {field_name}"
        elif "audio" in field_lower:
            return f"Upload an audio file for {field_name}"
        elif "prompt" in field_lower or "text" in field_lower:
            return f"Enter text for {field_name}"
        elif "file" in field_lower:
            return f"Upload a file for {field_name}"
        else:
            return f"Provide value for {field_name}"

    # Config field prompts
    elif category == FieldCategory.CONFIG:
        return f"Adjust {field_name} (optional)"

    # Hybrid field prompts
    else:
        return f"Optionally provide {field_name}"


async def classify_form_fields(
    example_input: Dict[str, Any],
    model_name: str,
    model_description: str,
    field_metadata: Optional[Dict[str, Any]] = None
) -> FormClassificationOutput:
    """
    Classify form fields using AI agent

    Args:
        example_input: The example_input schema from tool config
        model_name: Name of the model
        model_description: Description of the model
        field_metadata: Optional metadata about fields

    Returns:
        FormClassificationOutput with field classifications
    """

    agent = create_classifier_agent()

    input_data = FormClassificationInput(
        example_input=example_input,
        model_name=model_name,
        model_description=model_description,
        field_metadata=field_metadata
    )

    try:
        result = await agent.run(input_data.model_dump())
        classification_result = result.output

        # Validate that we got actual classifications
        if not classification_result.field_classifications or len(classification_result.field_classifications) == 0:
            print(f"⚠️ AI classification returned empty results, falling back to heuristic classification")
            return _fallback_classification(example_input, model_name, model_description)

        return classification_result
    except Exception as e:
        print(f"⚠️ AI classification failed: {e}, falling back to heuristic classification")
        return _fallback_classification(example_input, model_name, model_description)


def _fallback_classification(
    example_input: Dict[str, Any],
    model_name: str,
    model_description: str
) -> FormClassificationOutput:
    """
    Fallback heuristic-based classification when AI agent fails
    """

    classifications = {}
    required_fields = []
    optional_fields = []

    def classify_field(field_name: str, value: Any, parent_path: str = "") -> FieldClassification:
        """Classify a single field using heuristics"""

        field_lower = field_name.lower()
        value_type = infer_value_type(value)
        is_collection = isinstance(value, (list, tuple))

        # Determine category using heuristics
        content_patterns = [
            "image", "video", "audio", "file", "prompt", "text",
            "instruction", "source", "target", "input", "media"
        ]

        config_patterns = [
            "scale", "steps", "iterations", "strength", "guidance",
            "temperature", "seed", "scheduler", "sampler", "format",
            "quality", "resolution", "num_", "max_", "min_", "cfg",
            "system_prompt", "system_message", "system"
        ]

        # Classify based on patterns
        is_content = any(pattern in field_lower for pattern in content_patterns)
        is_config = any(pattern in field_lower for pattern in config_patterns)

        # Special case: system_prompt is configuration, not required content
        if "system" in field_lower and "prompt" in field_lower:
            is_content = False
            is_config = True

        # Override logic
        if is_collection:
            # Arrays are usually content
            category = FieldCategory.CONTENT
            reset = True
            default_value = []
        elif is_content and not is_config:
            category = FieldCategory.CONTENT
            reset = True
            default_value = None if value_type != "string" else ""
        elif is_config:
            category = FieldCategory.CONFIG
            reset = False
            default_value = value
        elif value_type == "string" and isinstance(value, str) and value.startswith("http"):
            # URLs are usually content
            category = FieldCategory.CONTENT
            reset = True
            default_value = None
        elif value_type in ["number", "integer", "boolean"]:
            # Numbers and booleans are usually config
            category = FieldCategory.CONFIG
            reset = False
            default_value = value
        else:
            # Default to hybrid
            category = FieldCategory.HYBRID
            reset = True
            default_value = None if value_type != "string" else ""

        # Determine if required (heuristic: content fields are usually required)
        required = category == FieldCategory.CONTENT and not is_collection

        # Handle nested objects
        nested_classification = None
        if value_type == "object" and isinstance(value, dict):
            nested_classification = {
                nested_field: classify_field(nested_field, nested_value, f"{parent_path}.{field_name}").model_dump()
                for nested_field, nested_value in value.items()
            }

        return FieldClassification(
            field_name=field_name,
            category=category,
            reset=reset,
            required=required,
            default_value=default_value,
            user_prompt=generate_user_prompt(field_name, category, value_type),
            collection=is_collection,
            nested_classification=nested_classification,
            value_type=value_type
        )

    # Classify all top-level fields
    for field_name, value in example_input.items():
        classification = classify_field(field_name, value)
        classifications[field_name] = classification

        if classification.required:
            required_fields.append(field_name)
        else:
            optional_fields.append(field_name)

    return FormClassificationOutput(
        field_classifications=classifications,
        reasoning="Fallback heuristic classification used due to AI agent failure",
        required_fields=required_fields,
        optional_fields=optional_fields
    )


# Convenience function for synchronous usage
def classify_form_fields_sync(
    example_input: Dict[str, Any],
    model_name: str,
    model_description: str,
    field_metadata: Optional[Dict[str, Any]] = None
) -> FormClassificationOutput:
    """Synchronous wrapper for classify_form_fields"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    classify_form_fields(example_input, model_name, model_description, field_metadata)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                classify_form_fields(example_input, model_name, model_description, field_metadata)
            )
    except Exception:
        # Fall back to heuristic classification
        return _fallback_classification(example_input, model_name, model_description)
