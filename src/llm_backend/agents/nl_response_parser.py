"""
Natural Language Response Parser Agent
Extracts structured field values from natural language user responses
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class ParsedFieldValues(BaseModel):
    """Output from the natural language response parser"""
    extracted_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed field values extracted from user's natural language response"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score (0.0-1.0) for the parsing accuracy"
    )
    ambiguities: List[str] = Field(
        default_factory=list,
        description="Fields that were ambiguous or unclear in the response"
    )
    clarification_needed: Optional[str] = Field(
        default=None,
        description="Follow-up question if clarification is needed"
    )
    parsing_notes: List[str] = Field(
        default_factory=list,
        description="Notes about parsing decisions made"
    )


class ParserInput(BaseModel):
    """Input to the response parser agent"""
    user_message: str = Field(description="Natural language response from user")
    expected_schema: Dict[str, Any] = Field(description="Expected field schema from form_field_classifier")
    current_values: Dict[str, Any] = Field(description="Current field values for context")
    model_description: str = Field(default="", description="Description of the model being configured")


# System prompt for the response parser
PARSER_SYSTEM_PROMPT = """You are an expert at extracting structured data from natural language text.

Your job is to parse a user's natural language response and extract specific field values they're providing.

Guidelines:
1. Extract field values semantically - understand intent, not just keywords
2. Handle natural variations: "3 variations" → num_outputs: 3, "in 4:3" → aspect_ratio: "4:3"
3. Preserve natural descriptions for prompt fields
4. Convert natural language to appropriate types (numbers, booleans, etc.)
5. If a field isn't mentioned, don't invent a value - leave it out
6. Assign confidence based on clarity (0.9+ = very clear, 0.7-0.9 = pretty clear, <0.7 = ambiguous)
7. Request clarification if critical fields are ambiguous

Type conversions to handle:
- Numbers: "three" → 3, "2.5" → 2.5, "half" → 0.5
- Booleans: "yes/yeah/yep" → true, "no/nope" → false
- Arrays: "3 variations" → num_outputs: 3, "with 5 steps" → num_inference_steps: 5
- Aspect ratios: "4:3", "16 by 9", "widescreen" → "16:9", "square" → "1:1"
- Formats: "mp3", "wav", "png", "jpg" etc.

Examples:

User: "Create a photo of a sunset over mountains in 4:3 format, I need 3 variations"
Extracted:
  - prompt: "a photo of a sunset over mountains"
  - aspect_ratio: "4:3"
  - num_outputs: 3
Confidence: 0.95

User: "A dog"
Extracted:
  - prompt: "a dog"
Confidence: 0.85 (brief but clear)

User: "Change the sky to purple with more contrast"
Extracted:
  - instruction: "change the sky to purple"
  - guidance_scale: [inferred: higher for "more contrast"]
Confidence: 0.75 (guidance_scale is inferred, may need clarification)

User: "Make it better"
Extracted: {}
Confidence: 0.3
Clarification: "What specific improvements would you like? For example, changing colors, adjusting composition, adding/removing elements?"
"""


def create_response_parser_agent() -> Agent[ParserInput, ParsedFieldValues]:
    """Create the natural language response parser agent"""
    agent = Agent(
        model="openai:gpt-4o-mini",
        system_prompt=PARSER_SYSTEM_PROMPT,
        output_type=ParsedFieldValues,
        retries=3
    )
    return agent


# Singleton agent instance
nl_response_parser_agent = create_response_parser_agent()


async def parse_natural_language_response(
    user_message: str,
    expected_schema: Dict[str, Any],
    current_values: Dict[str, Any],
    model_description: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> ParsedFieldValues:
    """
    Parse natural language response to extract structured field values

    Args:
        user_message: Natural language response from user
        expected_schema: Expected field schema (from form_field_classifier)
        current_values: Current field values for context
        model_description: Description of what the model does
        conversation_history: Full conversation context for better parsing (NEW)

    Returns:
        ParsedFieldValues with extracted fields and confidence score
    """
    # Extract field classifications
    field_classifications = expected_schema.get("field_classifications", {})

    # Build field information for the agent
    field_info = []
    for field_name, field_class in field_classifications.items():
        if hasattr(field_class, 'value_type'):
            value_type = field_class.value_type
            category = field_class.category
            user_prompt = getattr(field_class, 'user_prompt', f"Provide {field_name}")
        else:
            value_type = field_class.get("value_type", "string")
            category = field_class.get("category", "CONTENT")
            user_prompt = field_class.get("user_prompt", f"Provide {field_name}")

        field_info.append({
            "name": field_name,
            "type": value_type,
            "category": category,
            "description": user_prompt
        })

    # Prepare input for agent
    agent_input = ParserInput(
        user_message=user_message,
        expected_schema=field_classifications,
        current_values=current_values,
        model_description=model_description
    )

    # Build conversation context if provided
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            role = msg.get("role", "user")
            content = msg.get("message") or msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content[:200]}")

        if history_lines:
            history_context = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    # Build structured prompt for parsing
    user_prompt = f"""
{history_context}Parse the following user message and extract field values:

User message: "{user_message}"

Expected fields:
{chr(10).join(f"- {f['name']} ({f['type']}): {f['description']}" for f in field_info)}

Current values (for context):
{chr(10).join(f"- {k}: {v}" for k, v in current_values.items()) if current_values else "None"}

Model description: {model_description or 'Not provided'}

Extract any field values mentioned in the user's message. Consider the conversation history, especially if they're fixing a validation error or clarifying a previous response.

Be smart about semantic matching:
- "3 variations" likely means num_outputs: 3
- "in 4:3" or "4:3 format" means aspect_ratio: "4:3"
- "square" means aspect_ratio: "1:1"
- "Use 1:1" (in response to error about aspect_ratio) means aspect_ratio: "1:1"
- Natural descriptions should go to prompt/instruction fields
- Convert natural language to appropriate types

Return:
1. extracted_fields: Dict of field_name → value
2. confidence: 0.0-1.0 score
3. ambiguities: List of unclear fields
4. clarification_needed: Question if critical info is missing (null if not needed)
5. parsing_notes: Notes about your decisions
"""

    try:
        result = await nl_response_parser_agent.run(user_prompt, deps=agent_input)
        return result.output

    except Exception as e:
        # Fallback to heuristic parsing if AI fails
        print(f"⚠️ NL response parser failed, using fallback: {e}")
        return _fallback_response_parsing(
            user_message,
            field_classifications,
            current_values
        )


def _fallback_response_parsing(
    user_message: str,
    field_classifications: Dict[str, Any],
    current_values: Dict[str, Any]
) -> ParsedFieldValues:
    """Fallback heuristic response parsing if AI agent fails"""

    extracted = {}
    parsing_notes = []
    ambiguities = []
    confidence = 0.7  # Default moderate confidence for heuristics

    msg_lower = user_message.lower()

    # Find the most likely prompt/instruction field
    prompt_fields = []
    numeric_fields = []
    for field_name, field_class in field_classifications.items():
        value_type = field_class.get("value_type") if isinstance(field_class, dict) else field_class.value_type
        category = field_class.get("category") if isinstance(field_class, dict) else field_class.category

        if category in ["CONTENT", "HYBRID"] and value_type == "string":
            prompt_fields.append(field_name)
        elif value_type in ["integer", "number"]:
            numeric_fields.append(field_name)

    # Extract main prompt/instruction (use full message as default)
    if prompt_fields:
        main_field = prompt_fields[0]
        extracted[main_field] = user_message.strip()
        parsing_notes.append(f"Assigned full message to {main_field}")

    # Extract numeric values with heuristics
    import re

    # Aspect ratio patterns
    aspect_ratio_match = re.search(r'(\d+):(\d+)|(\d+)\s*by\s*(\d+)', msg_lower)
    if aspect_ratio_match:
        if aspect_ratio_match.group(1):
            ratio = f"{aspect_ratio_match.group(1)}:{aspect_ratio_match.group(2)}"
        else:
            ratio = f"{aspect_ratio_match.group(3)}:{aspect_ratio_match.group(4)}"

        # Find aspect_ratio field
        for field_name in field_classifications.keys():
            if "aspect" in field_name.lower() or "ratio" in field_name.lower():
                extracted[field_name] = ratio
                parsing_notes.append(f"Extracted aspect ratio: {ratio}")
                break

    # Common numeric patterns
    number_patterns = [
        (r'(\d+)\s*variations?', 'num_outputs'),
        (r'(\d+)\s*outputs?', 'num_outputs'),
        (r'(\d+)\s*images?', 'num_outputs'),
        (r'(\d+)\s*steps?', 'num_inference_steps'),
        (r'strength\s*[:=]?\s*([\d.]+)', 'strength'),
        (r'scale\s*[:=]?\s*([\d.]+)', 'guidance_scale'),
    ]

    for pattern, likely_field in number_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            # Find matching field
            for field_name in field_classifications.keys():
                if likely_field in field_name.lower():
                    extracted[field_name] = value
                    parsing_notes.append(f"Extracted {field_name}: {value}")
                    break

    # Check for common format keywords
    format_keywords = {
        'square': ('aspect_ratio', '1:1'),
        'widescreen': ('aspect_ratio', '16:9'),
        'portrait': ('aspect_ratio', '9:16'),
        'landscape': ('aspect_ratio', '16:9'),
    }

    for keyword, (field_type, value) in format_keywords.items():
        if keyword in msg_lower:
            for field_name in field_classifications.keys():
                if field_type in field_name.lower():
                    extracted[field_name] = value
                    parsing_notes.append(f"Matched keyword '{keyword}' to {field_name}: {value}")
                    break

    # Confidence adjustment based on extraction quality
    if len(extracted) == 0:
        confidence = 0.3
        ambiguities.append("No clear field values detected")
    elif len(extracted) >= len([f for f, c in field_classifications.items()
                                  if (c.get("required") if isinstance(c, dict) else c.required)]):
        confidence = 0.9

    clarification = None
    if confidence < 0.7:
        clarification = "I'm not sure I understood everything. Could you provide more details?"

    return ParsedFieldValues(
        extracted_fields=extracted,
        confidence=confidence,
        ambiguities=ambiguities,
        clarification_needed=clarification,
        parsing_notes=parsing_notes
    )
