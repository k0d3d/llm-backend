"""
AI Agent for intelligently mapping user attachments to form fields
"""

from typing import Dict, Any, List, Optional
from pydantic_ai import Agent, Tool
from pydantic import BaseModel, Field

from llm_backend.agents.field_analyzer import analyze_url_pattern, analyze_field_name


class AttachmentMappingInput(BaseModel):
    """Input for attachment mapping agent"""
    user_attachments: List[str]
    field_classifications: Dict[str, Any]
    example_input: Dict[str, Any]
    model_name: str = ""
    model_description: str = ""


class FieldAttachmentMapping(BaseModel):
    """Mapping for a single field"""
    field_name: str
    attachment: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str
    file_type: str  # image, audio, video, document, etc.


class AttachmentMappingOutput(BaseModel):
    """Output from attachment mapping agent"""
    mappings: List[FieldAttachmentMapping]
    unmapped_attachments: List[str]
    unmapped_fields: List[str]
    overall_reasoning: str


attachment_mapper_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=AttachmentMappingOutput,
    tools=[
        Tool(analyze_url_pattern, takes_ctx=False),
        Tool(analyze_field_name, takes_ctx=False),
    ],
    system_prompt="""You are an AI agent that intelligently maps user-provided attachments (URLs) to form fields in AI model input schemas.

Your job is to:
1. Analyze each user attachment to determine its file type (image, audio, video, document, etc.)
2. Analyze each field in the schema to understand what type of input it expects
3. Create smart mappings between attachments and fields based on semantic understanding
4. Provide confidence scores and reasoning for each mapping

Key Rules:
1. USE SEMANTIC UNDERSTANDING - Don't just match exact field names
   - "image" = "input_image" = "img" = "photo" = "picture" = same thing!
   - "audio" = "input_audio" = "sound" = "audio_file" = same thing!

2. ANALYZE FILE TYPES from URLs
   - .jpg, .png, .webp → image fields
   - .mp3, .wav, .m4a → audio fields
   - .mp4, .mov, .avi → video fields
   - .pdf, .doc, .txt → document fields

3. HANDLE BOTH SINGLE AND ARRAY FIELDS
   - Single field (string): Map one attachment
   - Array field (list): Can map multiple attachments
   - Arrays in form classification have "collection": true or "value_type": "array"

4. PRIORITIZE based on field classification categories:
   - CONTENT fields (category="CONTENT") are highest priority - user MUST provide these
   - HYBRID fields (category="HYBRID") are medium priority
   - CONFIG fields (category="CONFIG") should NOT receive attachments (they're settings)

5. BE CONFIDENT with obvious matches:
   - User uploads image.jpg + schema has "image" field → confidence: 0.95+
   - User uploads song.mp3 + schema has "audio" field → confidence: 0.95+

6. USE CONTEXT from model description:
   - If model is "image editor", prioritize image fields
   - If model is "audio generator", prioritize audio fields

Example Analysis:
Input:
  - Attachments: ["https://serve.com/photo.jpg"]
  - Fields: {"image": {...}, "strength": {...}}
  - Classifications: {"image": {"category": "CONTENT", "collection": false}}

Output:
  - Map "photo.jpg" → "image" field (confidence: 0.95)
  - Reasoning: "Single image attachment matches CONTENT image field"

Use the analyze_url_pattern and analyze_field_name tools to help with analysis.
""",
)


async def map_attachments_to_fields(
    user_attachments: List[str],
    field_classifications: Dict[str, Any],
    example_input: Dict[str, Any],
    model_name: str = "",
    model_description: str = "",
) -> AttachmentMappingOutput:
    """
    Use AI to intelligently map user attachments to form fields

    Args:
        user_attachments: List of user-provided URLs/files
        field_classifications: Field metadata from FormFieldClassifierAgent
        example_input: Original example_input schema
        model_name: Name of the AI model
        model_description: Description of what the model does

    Returns:
        AttachmentMappingOutput with field mappings and reasoning
    """

    if not user_attachments:
        return AttachmentMappingOutput(
            mappings=[],
            unmapped_attachments=[],
            unmapped_fields=list(field_classifications.keys()),
            overall_reasoning="No attachments provided"
        )

    if not field_classifications:
        return AttachmentMappingOutput(
            mappings=[],
            unmapped_attachments=user_attachments,
            unmapped_fields=[],
            overall_reasoning="No fields to map to"
        )

    input_data = AttachmentMappingInput(
        user_attachments=user_attachments,
        field_classifications=field_classifications,
        example_input=example_input,
        model_name=model_name,
        model_description=model_description,
    )

    # Build context about fields for the AI
    field_context = []
    for field_name, field_class in field_classifications.items():
        # Handle both object and dict forms
        if hasattr(field_class, 'category'):
            category = field_class.category
            is_collection = field_class.collection
            value_type = getattr(field_class, 'value_type', 'string')
            required = field_class.required
        else:
            category = field_class.get('category', 'HYBRID')
            is_collection = field_class.get('collection', False)
            value_type = field_class.get('value_type', 'string')
            required = field_class.get('required', False)

        field_context.append(
            f"  - {field_name}: category={category}, "
            f"is_array={is_collection}, type={value_type}, required={required}"
        )

    field_context_str = "\n".join(field_context)

    prompt = f"""Map user attachments to form fields in this AI model schema:

MODEL: {model_name}
DESCRIPTION: {model_description}

USER ATTACHMENTS:
{user_attachments}

FIELD CLASSIFICATIONS:
{field_context_str}

EXAMPLE INPUT SCHEMA:
{example_input}

Task:
1. For each attachment, determine its file type using analyze_url_pattern tool
2. For each field, analyze if it expects file input using analyze_field_name tool
3. Create intelligent mappings between attachments and fields
4. Remember: "image" = "input_image" = "img" (semantic matching!)
5. Prioritize CONTENT category fields (they need user input)
6. Don't map to CONFIG fields (those are settings)
7. Handle both single fields and array fields correctly

Provide confidence scores and clear reasoning for each mapping."""

    try:
        result = await attachment_mapper_agent.run(prompt, deps=input_data)
        return result.output
    except Exception as agent_error:
        print(f"⚠️ Attachment mapping agent failed: {agent_error}")
        # Fallback to heuristic mapping
        return _heuristic_attachment_mapping(
            user_attachments,
            field_classifications,
            example_input
        )


def _heuristic_attachment_mapping(
    user_attachments: List[str],
    field_classifications: Dict[str, Any],
    example_input: Dict[str, Any],
) -> AttachmentMappingOutput:
    """
    Fallback heuristic mapping when AI agent fails
    Uses smart pattern matching and file type detection
    """

    mappings: List[FieldAttachmentMapping] = []
    unmapped_attachments = list(user_attachments)

    # Analyze each attachment's file type
    attachment_types: Dict[str, str] = {}
    for url in user_attachments:
        analysis = analyze_url_pattern(url)
        attachment_types[url] = analysis.get('file_type', 'unknown')

    # Priority patterns for different file types
    field_patterns = {
        'image': ['image', 'img', 'photo', 'picture', 'pic', 'input_image', 'source_image', 'image_input'],
        'audio': ['audio', 'sound', 'music', 'voice', 'input_audio', 'audio_file', 'audio_input'],
        'video': ['video', 'movie', 'clip', 'input_video', 'video_file', 'video_input'],
        'document': ['document', 'doc', 'file', 'text', 'input_file', 'document_input'],
    }

    # Try to map each attachment
    for attachment in user_attachments:
        file_type = attachment_types.get(attachment, 'unknown')

        # Get relevant field patterns for this file type
        relevant_patterns = field_patterns.get(file_type, ['input', 'file', 'attachment'])

        # Score each field for this attachment
        field_scores: List[tuple[str, float, str]] = []

        for field_name, field_class in field_classifications.items():
            # Get field metadata
            if hasattr(field_class, 'category'):
                category = str(field_class.category)
                is_collection = field_class.collection
                required = field_class.required
            else:
                category = field_class.get('category', 'HYBRID')
                is_collection = field_class.get('collection', False)
                required = field_class.get('required', False)

            # Skip CONFIG fields - they're not for attachments
            if 'CONFIG' in category:
                continue

            # Calculate match score
            score = 0.0
            reasoning_parts = []

            field_lower = field_name.lower()
            field_analysis = analyze_field_name(field_name)

            # Exact or semantic match with file type patterns
            for pattern in relevant_patterns:
                if pattern == field_lower:
                    score += 0.8
                    reasoning_parts.append(f"exact match '{pattern}'")
                    break
                elif pattern in field_lower or field_lower in pattern:
                    score += 0.6
                    reasoning_parts.append(f"contains '{pattern}'")
                    break

            # Boost for CONTENT category (these need user input)
            if 'CONTENT' in category:
                score += 0.2
                reasoning_parts.append("CONTENT field")

            # Boost for required fields
            if required:
                score += 0.1
                reasoning_parts.append("required")

            # Penalty for collections if we only have one attachment
            if is_collection and len(user_attachments) == 1:
                score -= 0.1
                reasoning_parts.append("array field with single attachment")

            if score > 0:
                field_scores.append((
                    field_name,
                    min(score, 0.95),  # Cap at 0.95
                    ", ".join(reasoning_parts)
                ))

        # Map to highest scoring field
        if field_scores:
            field_scores.sort(key=lambda x: x[1], reverse=True)
            best_field, confidence, reasoning = field_scores[0]

            mappings.append(FieldAttachmentMapping(
                field_name=best_field,
                attachment=attachment,
                confidence=confidence,
                reasoning=f"Heuristic match: {reasoning}",
                file_type=file_type
            ))

            if attachment in unmapped_attachments:
                unmapped_attachments.remove(attachment)

    # Identify unmapped fields (CONTENT category fields without attachments)
    mapped_fields = {m.field_name for m in mappings}
    unmapped_fields = [
        fname for fname, fclass in field_classifications.items()
        if fname not in mapped_fields and (
            (hasattr(fclass, 'category') and 'CONTENT' in str(fclass.category)) or
            (isinstance(fclass, dict) and 'CONTENT' in fclass.get('category', ''))
        )
    ]

    return AttachmentMappingOutput(
        mappings=mappings,
        unmapped_attachments=unmapped_attachments,
        unmapped_fields=unmapped_fields,
        overall_reasoning="Heuristic fallback mapping based on field name patterns and file type analysis"
    )
