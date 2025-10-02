"""
AI Agent for resolving attachment conflicts in Replicate payloads
"""

from typing import Dict, Any, List, Optional
from pydantic_ai import Agent
from pydantic import BaseModel

from llm_backend.agents.field_analyzer import analyze_replaceable_fields, analyze_field_name



class AttachmentResolutionInput(BaseModel):
    """Input for attachment resolution agent"""
    user_attachments: List[str]
    example_input: Dict[str, Any]
    current_payload: Dict[str, Any]
    prompt: str


class AttachmentResolutionOutput(BaseModel):
    """Output from attachment resolution agent"""
    resolved_payload: Dict[str, Any]
    reasoning: str
    changes_made: List[str]


attachment_resolver_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=AttachmentResolutionOutput,
    system_prompt="""You are an AI agent that resolves attachment conflicts in model payloads.

Your job is to ensure that user-provided attachments take precedence over placeholder URLs in example inputs.

Rules:
1. ALWAYS prioritize user attachments over example/placeholder URLs
2. Replace placeholder URLs with user's actual files based on file type matching
3. Preserve all other payload fields unchanged
4. Use intelligent field analysis to identify replaceable fields
5. Match user attachments to appropriate fields based on file type
6. Explain your reasoning clearly

File type matching:
- Images (.jpg, .png, etc.) → image-related fields (input_image, image, source_image)
- Audio (.mp3, .wav, etc.) → audio-related fields (audio_file, input_audio, sound)
- Video (.mp4, .mov, etc.) → video-related fields (video_file, input_video)
- Documents (.pdf, .txt, etc.) → document fields (document, file, input_file)

Example scenario:
- User uploads: https://serve-dev.tohju.com/user-photo.jpg (image)
- Example has: {"input_image": "https://replicate.delivery/example.png", "strength": 0.8}
- Result: Replace input_image with user's URL, keep strength unchanged
""",
)


async def resolve_attachment_conflicts(
    user_attachments: List[str],
    example_input: Dict[str, Any],
    current_payload: Dict[str, Any],
    prompt: str,
    example_urls: Optional[List[str]] = None,
) -> AttachmentResolutionOutput:
    """Resolve conflicts between user attachments and example input URLs"""

    # Extract example URLs to filter them out from user attachments
    if example_urls is None:
        example_urls = _extract_example_urls(example_input)
    
    replaceable_fields: List[str] = []
    analysis_summary = "No analysis available"

    try:
        analysis = await analyze_replaceable_fields(
            example_input=example_input,
            user_attachments=user_attachments,
            model_description=""
        )

        replaceable_fields = [
            field for field in analysis.replaceable_fields
            if field in current_payload
        ]

        analysis_summary = (
            "Field analysis results:\n"
            f"- Replaceable fields: {analysis.replaceable_fields}\n"
            f"- Field types: {analysis.field_types}\n"
            f"- Confidence scores: {analysis.confidence_scores}"
        )
    except Exception as e:
        print(f"⚠️ Field analysis failed: {e}")
        replaceable_fields = _heuristic_replaceable_fields(current_payload)
        analysis_summary = (
            "Field analysis failed; using heuristic detection.\n"
        )

    if not replaceable_fields:
        replaceable_fields = _heuristic_replaceable_fields(current_payload)
        analysis_summary += (
            "\nNo high-confidence fields identified; applying heuristic fallback: "
            f"{replaceable_fields}"
        )

    if not replaceable_fields and example_input:
        replaceable_fields = _heuristic_fields_from_example(example_input)
        analysis_summary += (
            "\nNo heuristic fields identified; applying example-based fallback: "
            f"{replaceable_fields}"
        )

    if not replaceable_fields:
        replaceable_fields = _default_media_fields()
        analysis_summary += (
            "\nNo example-based fields identified; applying default media fields fallback: "
            f"{replaceable_fields}"
        )

    input_data = AttachmentResolutionInput(
        user_attachments=user_attachments,
        example_input=example_input,
        current_payload=current_payload,
        prompt=prompt,
    )

    try:
        result = await attachment_resolver_agent.run(
            f"""Resolve attachment conflicts in this payload:

                User attachments: {user_attachments}
                Example input: {example_input}
                Current payload: {current_payload}
                User prompt: {prompt}

                {analysis_summary}

                Replace the values for these fields with user attachments when appropriate: {replaceable_fields}.
                Preserve all other fields and explain any changes you make.""",
            deps=input_data,
        )

        return result.output
    except Exception as agent_error:
        print(f"⚠️ Attachment resolver agent failed: {agent_error}")
        resolved_payload, changes = _manual_attachment_resolution(
            user_attachments,
            current_payload,
            replaceable_fields,
            example_urls=example_urls
        )

        return AttachmentResolutionOutput(
            resolved_payload=resolved_payload,
            reasoning=(
                "Manual fallback applied due to agent failure. " +
                f"Error: {agent_error}. " +
                f"Replaceable fields considered: {replaceable_fields}"
            ),
            changes_made=changes,
        )


def _extract_example_urls(example_input: Dict[str, Any]) -> List[str]:
    """Extract all URL values from example input to filter them out"""
    urls = []
    for value in example_input.values():
        if isinstance(value, str) and value.startswith("http"):
            urls.append(value)
    return urls


def _heuristic_replaceable_fields(payload: Dict[str, Any]) -> List[str]:
    """Identify replaceable fields using simple heuristics"""
    candidate_fields: List[str] = []
    indicators = ["input", "file", "url", "image", "audio", "video", "document", "media"]

    for key, value in payload.items():
        if not isinstance(value, str):
            continue

        value_lower = value.lower()
        key_lower = key.lower()

        if value_lower.startswith("http") or any(ind in key_lower for ind in indicators):
            candidate_fields.append(key)

    return candidate_fields


def _heuristic_fields_from_example(example_input: Dict[str, Any]) -> List[str]:
    """Derive likely replaceable fields from example input schema"""
    candidate_fields: List[str] = []
    for key, value in example_input.items():
        if isinstance(value, str):
            analysis = analyze_field_name(key)
            if analysis["suggests_file"]:
                candidate_fields.append(key)

            # Include fields whose values look like URLs
            if value.startswith("http") and key not in candidate_fields:
                candidate_fields.append(key)

    return candidate_fields


def _default_media_fields() -> List[str]:
    """Fallback list of common media fields when detection fails"""
    return [
        "input_image",
        "image",
        "image_input",
        "source_image",
        "image_url",
        "input_file",
        "file",
        "input",
        "media",
    ]


def _manual_attachment_resolution(
    user_attachments: List[str],
    payload: Dict[str, Any],
    replaceable_fields: List[str],
    example_urls: List[str],
) -> tuple[Dict[str, Any], List[str]]:
    """Fallback manual replacement logic when agent execution fails"""
    if not user_attachments:
        return payload, []

    resolved = payload.copy()
    changes: List[str] = []
    
    # Filter out example URLs and known placeholder domains
    placeholder_domains = [
        "example.com",
        "replicate.delivery",
        "placeholder.com",
        "sample.com",
        "test.com",
    ]
    
    # Filter attachments by both example URLs and placeholder domains
    actual_user_attachments = [
        url for url in user_attachments
        if url not in example_urls and not any(domain in url for domain in placeholder_domains)
    ]
    
    # Use actual user attachment if available, otherwise fall back to first non-example URL
    primary_attachment = None
    if actual_user_attachments:
        primary_attachment = actual_user_attachments[0]
    else:
        # Try to find first non-example URL
        for url in user_attachments:
            if url not in example_urls:
                primary_attachment = url
                break
        # Last resort: use first attachment
        if not primary_attachment and user_attachments:
            primary_attachment = user_attachments[0]

    # Map generic field names to actual schema fields
    field_aliases = {
        "image": ["input_image", "image", "source_image", "image_url"],
        "audio": ["input_audio", "audio_file", "audio"],
        "video": ["input_video", "video_file", "video"],
        "file": ["input_file", "file", "input"],
        "document": ["document", "input_document", "file"]
    }
    
    # Expand replaceable_fields to include actual schema fields
    expanded_fields = set(replaceable_fields)
    for field in replaceable_fields:
        if field in field_aliases:
            expanded_fields.update(field_aliases[field])
    
    # Try to replace fields in order of preference
    for field in expanded_fields:
        value = resolved.get(field)

        if value is None:
            resolved[field] = primary_attachment
            changes.append(f"{field}: <missing> -> {primary_attachment}")
            break  # Only set one field to avoid duplicates

        if isinstance(value, str) and (any(domain in value for domain in placeholder_domains) or value.startswith("http")):
            resolved[field] = primary_attachment
            changes.append(f"{field}: {value} -> {primary_attachment}")
            break  # Only replace one field to avoid duplicates

    if not changes:
        # As a last resort, replace the first placeholder we find or add to common field
        for field, value in resolved.items():
            if isinstance(value, str) and any(domain in value for domain in placeholder_domains):
                resolved[field] = primary_attachment
                changes.append(f"{field}: {value} -> {primary_attachment}")
                break
        
        # If still no changes, add to most common field
        if not changes:
            resolved["input_image"] = primary_attachment
            changes.append(f"input_image: <added> -> {primary_attachment}")

    return resolved, changes
