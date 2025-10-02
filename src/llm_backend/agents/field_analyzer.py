"""
AI Agent for analyzing and identifying replaceable file/URL fields in model schemas
"""

from typing import Dict, Any, List
from pydantic_ai import Agent, Tool
from pydantic import BaseModel


class FieldAnalysisInput(BaseModel):
    """Input for field analysis agent"""
    example_input: Dict[str, Any]
    user_attachments: List[str]
    model_description: str


class FieldAnalysisOutput(BaseModel):
    """Output from field analysis agent"""
    replaceable_fields: List[str]
    field_types: Dict[str, str]  # field_name -> file_type (image, audio, video, pdf, etc.)
    reasoning: str
    confidence_scores: Dict[str, float]  # field_name -> confidence (0.0-1.0)


def analyze_url_pattern(url: str) -> Dict[str, Any]:
    """Analyze a URL to determine if it's a placeholder and what type of file it represents"""
    import re
    from urllib.parse import urlparse
    
    if not url or not isinstance(url, str):
        return {"is_placeholder": False, "file_type": "unknown"}
    
    # Check if it's a URL
    if not url.startswith(('http://', 'https://')):
        return {"is_placeholder": False, "file_type": "unknown"}
    
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Common placeholder domains
    placeholder_domains = [
        'example.com', 'replicate.delivery', 'placeholder.com',
        'test.com', 'demo.com', 'sample.com'
    ]
    
    is_placeholder = any(domain in parsed.netloc for domain in placeholder_domains)
    
    # Determine file type from extension
    file_type = "unknown"
    if re.search(r'\.(jpe?g|png|gif|webp|svg|bmp|tiff)$', path):
        file_type = "image"
    elif re.search(r'\.(mp3|wav|flac|aac|ogg|m4a)$', path):
        file_type = "audio"
    elif re.search(r'\.(mp4|avi|mov|mkv|webm|flv)$', path):
        file_type = "video"
    elif re.search(r'\.(pdf|docx?|txt|rtf|xlsx|odf|odt|md)$', path):
        file_type = "document"
    elif re.search(r'\.(json|csv|xml|ya?ml)$', path):
        file_type = "data"
    
    return {
        "is_placeholder": is_placeholder,
        "file_type": file_type,
        "domain": parsed.netloc,
        "extension": path.split('.')[-1] if '.' in path else None
    }


def analyze_field_name(field_name: str) -> Dict[str, Any]:
    """Analyze a field name to determine if it suggests file input"""
    field_lower = field_name.lower()
    
    # File input indicators
    file_indicators = [
        'input', 'file', 'url', 'image', 'audio', 'video', 'document',
        'attachment', 'upload', 'source', 'media', 'asset'
    ]
    
    # Type-specific indicators
    type_indicators = {
        'image': ['image', 'img', 'photo', 'picture', 'pic', 'frame'],
        'audio': ['audio', 'sound', 'music', 'voice', 'speech', 'wav', 'mp3'],
        'video': ['video', 'movie', 'clip', 'film', 'mp4'],
        'document': ['doc', 'pdf', 'text', 'file'],
        'data': ['data', 'json', 'csv', 'config']
    }
    
    suggests_file = any(indicator in field_lower for indicator in file_indicators)
    
    suggested_type = "unknown"
    for file_type, indicators in type_indicators.items():
        if any(indicator in field_lower for indicator in indicators):
            suggested_type = file_type
            break
    
    return {
        "suggests_file": suggests_file,
        "suggested_type": suggested_type,
        "confidence": 0.8 if suggests_file else 0.2
    }


field_analyzer_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=FieldAnalysisOutput,
    tools=[
        Tool(analyze_url_pattern, takes_ctx=False),
        Tool(analyze_field_name, takes_ctx=False),
    ],
    system_prompt="""You are an AI agent that analyzes model input schemas to identify fields that should be replaced with user-provided files/URLs.

Your job is to:
1. Examine the example_input structure and values
2. Identify fields that contain URLs or file references that should be replaced with user attachments
3. Determine the expected file type for each field (image, audio, video, pdf, etc.)
4. Provide confidence scores for your analysis

Rules for identifying replaceable fields:
1. Look for fields with URL values (http/https)
2. Consider field names that suggest file input (input_*, *_file, *_url, *_image, etc.)
3. Analyze the model description for context about expected inputs
4. Consider file extensions in URLs (.jpg, .mp3, .pdf, etc.)
5. Be conservative - only mark fields as replaceable if you're confident

File type detection:
- image: .jpg, .jpeg, .png, .gif, .webp, .svg, .bmp, .tiff
- audio: .mp3, .wav, .flac, .aac, .ogg, .m4a
- video: .mp4, .avi, .mov, .mkv, .webm, .flv
- document: .pdf, .doc, .docx, .txt, .rtf
- data: .json, .csv, .xml, .yaml

Example analysis:
Input: {"prompt": "Edit this", "input_image": "https://example.com/photo.jpg", "strength": 0.8}
Output: replaceable_fields=["input_image"], field_types={"input_image": "image"}

Use the analyze_url_pattern and analyze_field_name tools to help with your analysis.
""",
)


async def analyze_replaceable_fields(
    example_input: Dict[str, Any],
    user_attachments: List[str],
    model_description: str = ""
) -> FieldAnalysisOutput:
    """Analyze example input to identify fields that should be replaced with user attachments"""
    
    input_data = FieldAnalysisInput(
        example_input=example_input,
        user_attachments=user_attachments,
        model_description=model_description
    )
    
    result = await field_analyzer_agent.run(
        f"""Analyze this model input schema to identify fields that should be replaced with user attachments:

Example Input: {example_input}
User Attachments: {user_attachments}
Model Description: {model_description}

For each field in the example input:
1. Check if the value is a URL that looks like a placeholder
2. Analyze the field name for file input indicators
3. Determine what type of file is expected
4. Assign a confidence score

Focus on fields that clearly need user-provided files, not configuration parameters.""",
        deps=input_data
    )
    
    return result.output
