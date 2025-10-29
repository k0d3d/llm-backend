"""
Natural Language Prompt Generator Agent
Converts structured form field requirements into natural language explanations for users
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class NaturalLanguagePrompt(BaseModel):
    """Output from the natural language prompt generator"""
    message: str = Field(description="Natural language explanation of what the system needs from the user")
    all_fields_satisfied: bool = Field(description="True if all required fields are already filled")
    missing_field_names: List[str] = Field(default_factory=list, description="Names of fields that are missing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Structured context for response parsing")
    tone: str = Field(default="helpful", description="Conversational tone (helpful, brief, detailed)")


class PromptGeneratorInput(BaseModel):
    """Input to the prompt generator agent"""
    field_classifications: Dict[str, Any] = Field(description="Field classifications from form_field_classifier")
    current_values: Dict[str, Any] = Field(description="Current field values")
    model_name: str = Field(description="Name of the model being configured")
    model_description: str = Field(default="", description="Description of what the model does")


# System prompt for the natural language generator
GENERATOR_SYSTEM_PROMPT = """You are a helpful assistant that explains technical requirements in natural language.

Your job is to convert structured form field requirements into friendly, conversational messages for users.

Guidelines:
1. Be concise and friendly - users don't want technical jargon
2. Focus on what's missing or needs attention
3. Mention current/default values for optional fields in a natural way
4. Use examples when helpful
5. Keep the tone conversational, not robotic
6. If everything is already provided, acknowledge that

Examples of good messages:
- "I need a text prompt describing what image you want to create. You can also set the aspect ratio (currently 16:9) and number of outputs (currently 1). What would you like to generate?"
- "I need an image to edit and instructions on what changes to make. What image should I edit and what changes should I make?"
- "Perfect! I have everything I need: your prompt and settings. Proceeding with generation."

Bad examples (too technical):
- "The 'prompt' field (type: string, category: CONTENT) is required but not provided."
- "Please fill the following fields: ['prompt']"
"""


def create_prompt_generator_agent() -> Agent[PromptGeneratorInput, NaturalLanguagePrompt]:
    """Create the natural language prompt generator agent"""
    agent = Agent(
        model="openai:gpt-4o-mini",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        output_type=NaturalLanguagePrompt,
        retries=3
    )
    return agent


# Singleton agent instance
nl_prompt_generator_agent = create_prompt_generator_agent()


async def generate_natural_language_prompt(
    classification: Dict[str, Any],
    current_values: Dict[str, Any],
    model_name: str,
    model_description: str = ""
) -> NaturalLanguagePrompt:
    """
    Generate a natural language prompt explaining what fields are needed

    Args:
        classification: Field classifications from form_field_classifier
        current_values: Current field values
        model_name: Name of the model
        model_description: Description of what the model does

    Returns:
        NaturalLanguagePrompt with message and context
    """
    # Extract field classifications
    field_classifications = classification.get("field_classifications", {})

    # Identify missing required fields
    missing_fields = []
    filled_fields = []
    optional_fields_info = []

    for field_name, field_class in field_classifications.items():
        # Handle both object and dict forms
        if hasattr(field_class, 'required'):
            required = field_class.required
            default_value = getattr(field_class, 'default_value', None)
            value_type = field_class.value_type
            user_prompt = getattr(field_class, 'user_prompt', f"Provide {field_name}")
        else:
            required = field_class.get("required", False)
            default_value = field_class.get("default_value")
            value_type = field_class.get("value_type", "string")
            user_prompt = field_class.get("user_prompt", f"Provide {field_name}")

        current_value = current_values.get(field_name)
        is_empty = current_value is None or current_value == "" or (isinstance(current_value, list) and len(current_value) == 0)

        if required:
            if is_empty:
                missing_fields.append({
                    "name": field_name,
                    "prompt": user_prompt,
                    "type": value_type
                })
            else:
                filled_fields.append(field_name)
        else:
            # Track optional fields with their current/default values
            display_value = current_value if not is_empty else default_value
            optional_fields_info.append({
                "name": field_name,
                "current_value": display_value,
                "default_value": default_value,
                "type": value_type
            })

    # Prepare input for agent
    agent_input = PromptGeneratorInput(
        field_classifications=field_classifications,
        current_values=current_values,
        model_name=model_name,
        model_description=model_description
    )

    # Build context for parsing
    context = {
        "model_name": model_name,
        "model_description": model_description,
        "required_fields": [f["name"] for f in missing_fields],
        "filled_fields": filled_fields,
        "optional_fields": {f["name"]: f["current_value"] for f in optional_fields_info},
        "field_types": {name: cls.get("value_type") if isinstance(cls, dict) else cls.value_type
                       for name, cls in field_classifications.items()}
    }

    # If all required fields are satisfied, return early
    if len(missing_fields) == 0:
        return NaturalLanguagePrompt(
            message=f"Perfect! I have everything I need to run {model_name}. Proceeding with your request.",
            all_fields_satisfied=True,
            missing_field_names=[],
            context=context,
            tone="brief"
        )

    # Use AI agent to generate natural language prompt
    try:
        # Build a structured prompt for the agent
        user_message = f"""
Generate a friendly, conversational message asking the user for the missing information.

Model: {model_name}
Description: {model_description or 'No description provided'}

Missing required fields:
{chr(10).join(f"- {f['name']}: {f['prompt']}" for f in missing_fields)}

Already filled:
{chr(10).join(f"- {f}" for f in filled_fields) if filled_fields else "None"}

Optional settings:
{chr(10).join(f"- {f['name']}: currently {f['current_value']}" for f in optional_fields_info) if optional_fields_info else "None"}

Generate a single, natural message that:
1. Explains what you need (the missing fields)
2. Mentions optional settings briefly
3. Asks what the user wants to do
4. Is friendly and conversational

Keep it concise - 2-3 sentences maximum.
"""

        result = await nl_prompt_generator_agent.run(user_message, deps=agent_input)

        # Ensure missing_field_names is populated
        if not result.output.missing_field_names:
            result.output.missing_field_names = [f["name"] for f in missing_fields]

        # Ensure context is populated
        if not result.output.context:
            result.output.context = context

        return result.output

    except Exception as e:
        # Fallback to deterministic prompt generation if AI fails
        print(f"⚠️ NL prompt generator failed, using fallback: {e}")
        return _fallback_prompt_generation(missing_fields, optional_fields_info, model_name, context)


def _fallback_prompt_generation(
    missing_fields: List[Dict[str, Any]],
    optional_fields: List[Dict[str, Any]],
    model_name: str,
    context: Dict[str, Any]
) -> NaturalLanguagePrompt:
    """Fallback deterministic prompt generation if AI agent fails"""

    # Build message parts
    parts = []

    # Required fields
    if len(missing_fields) == 1:
        field = missing_fields[0]
        parts.append(f"I need {field['prompt'].lower()}")
    else:
        field_names = ", ".join(f["prompt"].lower() for f in missing_fields[:-1])
        parts.append(f"I need {field_names}, and {missing_fields[-1]['prompt'].lower()}")

    # Optional fields (mention briefly)
    if optional_fields and len(optional_fields) <= 3:
        optional_mentions = []
        for field in optional_fields[:3]:
            if field['current_value']:
                optional_mentions.append(f"{field['name']} (currently {field['current_value']})")

        if optional_mentions:
            parts.append(f"You can also adjust {', '.join(optional_mentions)}")

    # Call to action
    parts.append("What would you like to do?")

    message = ". ".join(parts) + "."

    return NaturalLanguagePrompt(
        message=message,
        all_fields_satisfied=False,
        missing_field_names=[f["name"] for f in missing_fields],
        context=context,
        tone="helpful"
    )
