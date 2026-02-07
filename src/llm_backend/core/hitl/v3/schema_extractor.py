from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum
import re

class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"

class SchemaField(BaseModel):
    name: str
    type: FieldType
    description: Optional[str] = None
    default_value: Any = None
    is_required: bool = False
    is_content: bool = False  # Content = needs user input (e.g. prompt, image) vs Config = tuning parameters

class ModelSchema(BaseModel):
    fields: Dict[str, SchemaField]
    description: Optional[str] = None
    required_fields: List[str] = []

class SchemaExtractor:
    """
    Extracts a strict ModelSchema from a loose example_input dictionary.
    Focuses on separating 'Structure' from 'Values' to prevent data leakage.
    """

    @classmethod
    def extract(cls, tool_config: Dict[str, Any]) -> ModelSchema:
        example_input = tool_config.get("example_input", {})
        description = tool_config.get("description", "")
        
        fields = {}
        required_fields = []

        # Heuristic for editing models
        is_editing = any(word in description.lower() for word in ["edit", "style", "paint", "convert", "modify", "transform"])

        for key, value in example_input.items():
            field_type = cls._infer_type(value)
            is_content = cls._is_content_field(key, value)
            
            # Smarter requirement heuristic:
            # 1. Content fields are candidate for requirement
            # 2. Lists are required ONLY if it's an editing model (where image_input=[] is a failure)
            # 3. Specifically optional fields are excluded
            is_required = is_content
            
            if isinstance(value, list) and not is_editing:
                is_required = False
            
            # Negative signals for requirement
            if any(k in key.lower() for k in ["negative", "optional", "mask", "face"]):
                is_required = False

            # Determine default value
            # CRITICAL: For CONTENT fields, we explicit force None/Empty to avoid leakage
            if is_content:
                default_val = [] if field_type == FieldType.ARRAY else None
            else:
                default_val = value

            field = SchemaField(
                name=key,
                type=field_type,
                default_value=default_val,
                is_required=is_required,
                is_content=is_content
            )
            
            fields[key] = field
            if is_required:
                required_fields.append(key)

        return ModelSchema(
            fields=fields,
            description=description,
            required_fields=required_fields
        )

    @staticmethod
    def _infer_type(value: Any) -> FieldType:
        if isinstance(value, bool):
            return FieldType.BOOLEAN
        elif isinstance(value, int):
            return FieldType.INTEGER
        elif isinstance(value, float):
            return FieldType.NUMBER
        elif isinstance(value, str):
            return FieldType.STRING
        elif isinstance(value, list):
            return FieldType.ARRAY
        elif isinstance(value, dict):
            return FieldType.OBJECT
        return FieldType.UNKNOWN

    @staticmethod
    def _is_content_field(key: str, value: Any) -> bool:
        """
        Determines if a field is 'Content' (user data) or 'Config' (tuning).
        Content fields are reset to empty. Config fields keep defaults.
        """
        key_lower = key.lower()
        
        # 1. Strong signals for Content
        content_keywords = [
            "prompt", "text", "input", "image", "audio", "video", "file", 
            "source", "target", "mask", "face"
        ]
        if any(k in key_lower for k in content_keywords):
            # Exception: "system_prompt" is usually config (hidden instruction)
            if "system" in key_lower:
                return False
            # Exception: "guidance_scale" or "num_inference_steps" might contain "input" (rare, but possible)
            if "scale" in key_lower or "steps" in key_lower:
                return False
            return True

        # 2. Strong signals for Config
        config_keywords = [
            "seed", "scheduler", "width", "height", "steps", "scale", 
            "strength", "guidance", "num_", "count", "version", "format", 
            "quality", "fps", "model", "temperature", "top_k", "top_p"
        ]
        if any(k in key_lower for k in config_keywords):
            return False

        # 3. Value-based heuristics
        # URLs are likely content
        if isinstance(value, str) and value.startswith("http"):
            return True
            
        return False
