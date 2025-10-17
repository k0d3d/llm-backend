"""
Replicate provider implementation wrapping existing ReplicateTeam logic
"""

import time
import re
from copy import deepcopy
from typing import Dict, Any, List, Optional

from llm_backend.core.providers.base import (
    AIProvider, ProviderPayload, ProviderResponse, 
    ProviderCapabilities, ValidationIssue, OperationType
)
from llm_backend.tools.replicate_tool import run_replicate


class ReplicatePayload(ProviderPayload):
    """Replicate-specific payload"""
    input: Dict[str, Any]
    operation_type: OperationType
    model_version: str
    webhook_url: str = None


class ReplicateProvider(AIProvider):
    """Provider implementation for Replicate models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.example_input = config.get("example_input", {})
        self.description = config.get("description", "")
        self.latest_version = config.get("latest_version", "")
        self.model_name = config.get("name", "")
        self.field_metadata = self._build_field_metadata()
        self.hitl_alias_metadata = self._build_hitl_alias_metadata(self.field_metadata)
        self._orchestrator = None  # Will be set by orchestrator if using form-based workflow
    
    def _build_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Derive lightweight schema metadata from the example input."""
        metadata: Dict[str, Dict[str, Any]] = {}
        if not isinstance(self.example_input, dict):
            return metadata

        for field, value in self.example_input.items():
            field_meta: Dict[str, Any] = {
                "type": type(value).__name__,
                "collection": isinstance(value, (list, tuple)),
            }

            if isinstance(value, (list, tuple)) and value:
                field_meta["item_type"] = type(value[0]).__name__
            if isinstance(value, dict):
                field_meta["nested_keys"] = list(value.keys())

            metadata[field] = field_meta

        return metadata

    def _build_hitl_alias_metadata(self, field_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create alias metadata that captures collection/dict hints for HITL edits."""
        base_aliases: Dict[str, List[str]] = {
            "input_image": ["input_image", "source_image", "image", "image_url", "image_input"],
            "source_image": ["source_image", "input_image", "image", "image_url"],
            "image": ["image", "input_image", "source_image", "image_url", "image_input"],
            "driven_audio": ["driven_audio", "audio_file", "audio"],
            "audio_file": ["audio_file", "driven_audio", "audio"],
            "prompt": ["prompt", "text", "instruction", "input"],
            "instruction": ["instruction", "prompt", "text"],
        }

        alias_metadata: Dict[str, Dict[str, Any]] = {}

        def collection_hint(target: str) -> bool:
            return field_metadata.get(target, {}).get("collection", False)

        def dict_hint(target: str) -> bool:
            return field_metadata.get(target, {}).get("type") == "dict"

        for alias_key, targets in base_aliases.items():
            alias_metadata[alias_key] = {
                "targets": targets,
                "collection": any(collection_hint(t) for t in targets),
                "dict": any(dict_hint(t) for t in targets),
            }

        for field, meta in field_metadata.items():
            alias_metadata.setdefault(field, {
                "targets": [field],
                "collection": meta.get("collection", False),
                "dict": meta.get("type") == "dict",
            })

            # Provide generic aliases for multi-valued media fields
            if meta.get("collection") and "image" in field.lower():
                alias_metadata.setdefault("image_input", {
                    "targets": [field, "image_input"],
                    "collection": True,
                    "dict": False,
                })

        return alias_metadata

    def _coerce_payload_to_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure payload fields align with the inferred schema metadata."""
        if not payload:
            return payload

        coerced = payload.copy()
        for field, meta in self.field_metadata.items():
            if field not in coerced:
                continue

            value = coerced[field]

            if meta.get("collection") and not isinstance(value, (list, tuple)):
                coerced[field] = [value]
                continue

            if meta.get("type") == "dict" and isinstance(value, dict):
                continue

            if meta.get("type") == "dict" and not isinstance(value, dict):
                coerced[field] = {"value": value}

        return coerced

    def _filter_payload_to_schema(
        self,
        payload: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Strip payload fields that are not present in the example_input schema."""

        if not isinstance(payload, dict):
            return payload

        schema = schema if schema is not None else self.example_input
        if not isinstance(schema, dict):
            return payload

        filtered: Dict[str, Any] = {}

        for field, schema_value in schema.items():
            if field not in payload:
                continue

            value = payload[field]

            if isinstance(schema_value, dict) and isinstance(value, dict):
                filtered[field] = self._filter_payload_to_schema(value, schema_value)
            elif isinstance(schema_value, list):
                normalized_list = value if isinstance(value, list) else [value]

                if schema_value and isinstance(schema_value[0], dict):
                    filtered[field] = [
                        self._filter_payload_to_schema(item, schema_value[0])
                        if isinstance(item, dict)
                        else item
                        for item in normalized_list
                    ]
                else:
                    filtered[field] = normalized_list
            else:
                filtered[field] = value

        return filtered
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Return Replicate model capabilities"""
        operation_type = self._infer_operation_type()
        
        return ProviderCapabilities(
            name=self.model_name,
            description=self.description,
            version=self.latest_version,
            input_schema=self.example_input,
            supported_operations=[operation_type],
            safety_features=["content_filter", "nsfw_detection"],
            rate_limits={"requests_per_minute": 60},
            max_input_size=10 * 1024 * 1024,  # 10MB
            cost_per_request=0.01  # Estimate
        )
    
    def _normalize_attachments(self, attachments: List[str]) -> List[str]:
        """Normalize attachment URLs to ensure they're accessible to Replicate"""
        if not attachments:
            return []

        normalized = []
        for url in attachments:
            # Handle local dev server URLs
            if 'serve-dev.tohju.com' in url:
                normalized.append(url)
            # Handle replicate URLs
            elif 'replicate.delivery' in url:
                normalized.append(url)
            # Handle other URLs (could add S3, etc)
            else:
                normalized.append(url)
        return normalized

    def _strip_attachment_mentions(self, prompt: str, attachments: List[str]) -> str:
        """Remove attachment URL mentions from prompt text"""
        if not prompt or not attachments:
            return prompt

        clean_prompt = prompt
        for url in attachments:
            # Remove both the raw URL and any markdown-style links
            clean_prompt = clean_prompt.replace(url, '')
            clean_prompt = re.sub(r'\[.*?\]\([^)]*\)', '', clean_prompt)
            # Clean up any double spaces from removals
            clean_prompt = re.sub(r'\s+', ' ', clean_prompt)

        # Strip breadcrumb markers inserted for attachment references
        clean_prompt = re.sub(r'\s*:->\s*attached document:?\s*', ' ', clean_prompt, flags=re.IGNORECASE)
        clean_prompt = re.sub(r'""', '"', clean_prompt)

        return clean_prompt.strip()

    async def _resolve_attachment_conflicts(self, user_attachments: List[str], payload: Dict[str, Any], prompt: str, example_urls: List[str]) -> Dict[str, Any]:
        """Use AI agent to resolve conflicts between user attachments and example URLs"""
        from llm_backend.agents.attachment_resolver import resolve_attachment_conflicts
        
        try:
            result = await resolve_attachment_conflicts(
                user_attachments=user_attachments,
                example_input=self.example_input,
                current_payload=payload,
                prompt=prompt,
                example_urls=example_urls
            )
            
            print(f"🤖 Attachment resolver: {result.reasoning}")
            for change in result.changes_made:
                print(f"🔧 Attachment change: {change}")
            
            return result.resolved_payload
        except Exception as e:
            print(f"⚠️ Attachment resolution failed: {e}")
            # Fallback: manually replace example URLs with user attachments
            return self._manual_attachment_resolution(user_attachments, payload)
    
    def _manual_attachment_resolution(self, user_attachments: List[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Manual fallback for attachment resolution"""
        if not user_attachments:
            return payload
            
        resolved = payload.copy()
        primary_attachment = user_attachments[0]
        
        # Replace any example URLs with user's primary attachment
        for key, value in resolved.items():
            if isinstance(value, str) and "replicate.delivery" in value:
                resolved[key] = primary_attachment
                print(f"🔧 Manual replacement: {key} -> {primary_attachment}")

        return resolved

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference to access form data"""
        self._orchestrator = orchestrator
        print(f"📋 Orchestrator linked to provider for form-based payload creation")

    async def _create_payload_from_form(self, form_data: Dict[str, Any], operation_type: OperationType, config: Dict) -> ReplicatePayload:
        """Create payload directly from form data (bypasses intelligent agent)"""
        current_values = form_data.get("current_values", {})

        print(f"📋 Creating payload from form with {len(current_values)} fields")

        # Apply schema coercion to ensure proper types
        payload_input = self._coerce_payload_to_schema(current_values)

        # Filter to only include fields from example_input schema
        payload_input = self._filter_payload_to_schema(payload_input)

        print(f"📦 Final payload input: {list(payload_input.keys())}")

        # Build webhook URL - handle None case
        webhook_url = config.get("webhook_url")
        if webhook_url is None:
            webhook_url = ""  # Use empty string instead of None

        return ReplicatePayload(
            input=payload_input,
            operation_type=operation_type,
            model_version=self.latest_version,
            webhook_url=webhook_url,
            provider_name=self.model_name  # ADD: Missing provider_name field
        )

    async def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict, hitl_edits: Dict = None) -> ReplicatePayload:
        """Create Replicate-specific payload from form data or intelligent agent mapping"""
        from llm_backend.agents.replicate_team import ReplicateTeam
        from llm_backend.core.types.replicate import ExampleInput
        import asyncio

        if self.run_input is None:
            raise ValueError("ReplicateProvider requires run_input to be set before creating payloads")

        # NEW: Check if we have form data from HITL orchestrator
        if hasattr(self, '_orchestrator') and self._orchestrator and hasattr(self._orchestrator.state, 'form_data'):
            form_data = self._orchestrator.state.form_data
            if form_data and form_data.get("current_values"):
                print("📋 Using form data directly for payload creation")
                return await self._create_payload_from_form(form_data, operation_type, config)

        print("🤖 No form data available - using intelligent agent for payload creation")

        normalized_attachments = self._normalize_attachments(attachments or [])
        
        # Extract example URLs to filter them out from user attachments
        example_urls = set()
        if isinstance(self.example_input, dict):
            for value in self.example_input.values():
                if isinstance(value, str) and value.startswith("http"):
                    example_urls.add(value)
        
        # Filter out example URLs to get actual user attachments
        actual_user_attachments = [url for url in normalized_attachments if url not in example_urls]
        
        # Use actual user attachment as primary, fallback to first if none found
        primary_attachment = (actual_user_attachments[0] if actual_user_attachments 
                            else (normalized_attachments[0] if normalized_attachments else None))
        
        clean_prompt = self._strip_attachment_mentions(prompt, normalized_attachments)

        example_input_data = deepcopy(self.example_input) if isinstance(self.example_input, dict) else {}

        if primary_attachment:
            image_fields = [
                "input_image",
                "image",
                "image_input",
                "source_image",
                "image_url",
            ]

            assigned = False
            for field in image_fields:
                if field in example_input_data:
                    example_input_data[field] = primary_attachment
                    assigned = True
                    break

            if not assigned:
                example_input_data.setdefault("input_image", primary_attachment)

        replicate_team = ReplicateTeam(
            prompt=clean_prompt,
            tool_config={
                "example_input": example_input_data,
                "description": self.description,
                "latest_version": self.latest_version,
                "model_name": self.model_name,
                "field_metadata": self.field_metadata,
                "hitl_alias_metadata": self.hitl_alias_metadata,
            },
            run_input=self.run_input,
            hitl_enabled=True  # Enable intelligent mapping
        )

        agent_input = ExampleInput(
            prompt=clean_prompt,
            example_input=example_input_data,
            description=self.description,
            attachments=normalized_attachments or None,
            image_file=primary_attachment,
            hitl_edits=hitl_edits,
            schema_metadata=self.field_metadata,
            hitl_field_metadata=self.hitl_alias_metadata,
        )

        try:
            replicate_agent = replicate_team.replicate_agent()

            async def run_agent() -> Any:
                # print(f"🔍 Agent deps: {agent_input.model_dump()}")
                return await replicate_agent.run(
                    "Generate an optimal Replicate payload using the provided inputs.",
                    deps=agent_input,
                )

            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(run_agent()))
                    agent_payload = future.result()
            else:
                agent_payload = asyncio.run(run_agent())

            agent_output = getattr(agent_payload, "output", None)
            if agent_output is None:
                raise ValueError("Agent did not return an output payload")

            agent_input_payload = agent_output.input.model_dump()
            
            # Set initial prompt if needed
            prompt_targets = ["prompt", "text", "input", "instruction", "query"]
            for target in prompt_targets:
                if target in agent_input_payload or target in self.example_input:
                    agent_input_payload[target] = clean_prompt
                    break

            # Apply HITL edits after setting initial values
            if hitl_edits:
                agent_input_payload = self._apply_hitl_edits(agent_input_payload, hitl_edits)

            # Extract example URLs to filter them out from user attachments
            example_urls = set()
            if isinstance(self.example_input, dict):
                for value in self.example_input.values():
                    if isinstance(value, str) and value.startswith("http"):
                        example_urls.add(value)

            # Resolve attachment conflicts using AI agent
            if normalized_attachments:
                agent_input_payload = await self._resolve_attachment_conflicts(
                    normalized_attachments, agent_input_payload, prompt, list(example_urls)
                )

            if primary_attachment:
                attachment_fields = [
                    "input_image",
                    "image",
                    "image_input",
                    "source_image",
                    "image_url",
                ]
                if not any(agent_input_payload.get(field) for field in attachment_fields):
                    agent_input_payload[attachment_fields[0]] = primary_attachment

            agent_input_payload = self._filter_payload_to_schema(agent_input_payload)

            print(f"🤖 Agent created payload: {agent_input_payload}")

            metadata = {
                "original_example": self.example_input,
                "agent_generated": True,
                "agent_messages": getattr(agent_payload, "messages", None),
                "hitl_applied": bool(hitl_edits),
                "schema_metadata": self.field_metadata,
            }

            return ReplicatePayload(
                provider_name="replicate",
                input=self._coerce_payload_to_schema(agent_input_payload),
                operation_type=operation_type,
                model_version=self.latest_version,
                metadata=metadata,
            )

        except Exception as e:
            print(f"⚠️ Agent payload creation failed: {e}")
            print(f"🔍 Agent input details: prompt='{prompt}', hitl_edits={hitl_edits}, example_input={self.example_input}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            print("🔧 Falling back to static mapping")

            # Start with example input and set initial prompt
            fallback_input = deepcopy(example_input_data)
            prompt_targets = ["prompt", "text", "input", "instruction", "query"]
            for target in prompt_targets:
                if target in fallback_input:
                    fallback_input[target] = clean_prompt
                    break

            # Apply HITL edits after setting initial values
            if hitl_edits:
                fallback_input = self._apply_hitl_edits(fallback_input, hitl_edits)

            # Handle attachments in fallback
            if primary_attachment:
                fallback_attachment_fields = [
                    "image", "image_url", "input_image", "image_input",
                    "source_image", "first_frame_image", "subject_reference",
                    "start_image", "init_image"
                ]
                for field in fallback_attachment_fields:
                    if field in fallback_input:
                        fallback_input[field] = primary_attachment
                        break

            fallback_input = self._filter_payload_to_schema(fallback_input)

        return ReplicatePayload(
            provider_name="replicate",
            input=self._coerce_payload_to_schema(fallback_input),
            operation_type=operation_type,
            model_version=self.latest_version,
            metadata={
                "original_example": self.example_input,
                "fallback_used": True,
                "hitl_mapped": bool(hitl_edits),
            },
        )

    def _apply_hitl_edits(self, base_input: Dict[str, Any], hitl_edits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply human edits onto a payload input dict with smart alias handling."""
        if not hitl_edits:
            return base_input

        updated_input = base_input.copy()

        example_input = getattr(self, "example_input", {}) or {}
        alias_metadata = self.hitl_alias_metadata or {}

        def _coerce_collection(target_key: str, new_value: Any) -> Any:
            """Ensure new_value matches the collection type of the target."""
            current_value = updated_input.get(target_key, example_input.get(target_key))

            if isinstance(current_value, (list, tuple)):
                if isinstance(new_value, (list, tuple)):
                    return list(new_value)
                return [new_value]

            if isinstance(current_value, dict) and isinstance(new_value, dict):
                merged = current_value.copy()
                merged.update(new_value)
                return merged

            return new_value

        for key, value in hitl_edits.items():
            if value in (None, ""):
                continue

            alias_info = alias_metadata.get(key, {"targets": [key], "collection": False, "dict": False})
            targets = alias_info.get("targets", [key])
            applied = False

            for target in targets:
                if target in updated_input or target in example_input:
                    updated_input[target] = _coerce_collection(target, value)
                    applied = True

            if not applied and targets:
                primary_target = targets[0]
                updated_input[primary_target] = _coerce_collection(primary_target, value)

            print(f"🔧 Applied HITL overlay: {key} -> {value}")

        return updated_input
    
    def validate_payload(self, payload: ReplicatePayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        """Enhanced validation with HITL parameter detection"""
        issues = []
        
        # Analyze required parameters for this model
        required_params = self._analyze_model_requirements()
        missing_critical = self._check_critical_parameters(payload, prompt, attachments, required_params)
        
        # Add critical missing parameter issues
        for missing in missing_critical:
            issues.append(ValidationIssue(
                field=missing["field"],
                issue=missing["issue"],
                severity="error",
                suggested_fix=missing["suggested_fix"],
                auto_fixable=missing["auto_fixable"]
            ))
        
        # Check if prompt is in payload when required
        if prompt and self._requires_text_input():
            prompt_found = any(prompt in str(value) for value in payload.input.values())
            if not prompt_found:
                issues.append(ValidationIssue(
                    field="input",
                    issue="Prompt not found in payload",
                    severity="error",
                    suggested_fix=f"Add '{prompt}' to one of: {list(payload.input.keys())}",
                    auto_fixable=True
                ))
        
        # Check if required files are missing
        if self._requires_file_input() and not attachments:
            file_type = self._get_required_file_type()
            issues.append(ValidationIssue(
                field="input",
                issue=f"Required {file_type} file is missing",
                severity="error",
                suggested_fix=f"Upload a {file_type} file",
                auto_fixable=False
            ))
        elif attachments:
            file_found = any(att in str(payload.input) for att in attachments)
            if not file_found:
                issues.append(ValidationIssue(
                    field="input",
                    issue="Uploaded file not found in payload",
                    severity="error",
                    suggested_fix="Map uploaded file to appropriate field",
                    auto_fixable=True
                ))
        
        # Check for empty required fields
        for field_name, field_info in required_params.items():
            if field_info["required"] and field_name in payload.input:
                if not payload.input[field_name]:
                    issues.append(ValidationIssue(
                        field=field_name,
                        issue=f"Required field {field_name} is empty",
                        severity="error",
                        suggested_fix=field_info["description"],
                        auto_fixable=False
                    ))
        
        return issues
    
    def _analyze_model_requirements(self) -> dict:
        """Analyze what parameters this specific model requires"""
        model_lower = self.model_name.lower()
        requirements = {}
        
        # Model-specific requirements
        if "remove-bg" in model_lower or "background" in model_lower:
            requirements.update({
                "image": {"required": True, "type": "image", "description": "Upload an image to remove background from"},
                "input_image": {"required": True, "type": "image", "description": "Source image for background removal"}
            })
        elif "whisper" in model_lower or "transcrib" in model_lower:
            requirements.update({
                "audio": {"required": True, "type": "audio", "description": "Upload an audio file to transcribe"},
                "file": {"required": True, "type": "file", "description": "Audio file for transcription"}
            })
        elif "text-to-speech" in model_lower or "tts" in model_lower or "kokoro" in model_lower:
            requirements.update({
                "text": {"required": True, "type": "text", "description": "Enter text to convert to speech"},
                "prompt": {"required": True, "type": "text", "description": "Text content for speech synthesis"}
            })
        elif "upscal" in model_lower or "enhance" in model_lower or "clarity" in model_lower:
            requirements.update({
                "image": {"required": True, "type": "image", "description": "Upload image to upscale/enhance"},
                "input_image": {"required": True, "type": "image", "description": "Source image for enhancement"}
            })
        elif "nano-banana" in model_lower or "image-edit" in model_lower or "transform" in model_lower:
            requirements.update({
                "image": {"required": True, "type": "image", "description": "Upload image to transform"},
                "prompt": {"required": True, "type": "text", "description": "Describe the transformation you want"},
                "instruction": {"required": True, "type": "text", "description": "Instructions for image editing"}
            })
        
        # Add general requirements from example_input
        for key, value in self.example_input.items():
            if key not in requirements:
                requirements[key] = {
                    "required": self._is_field_required(key, value),
                    "type": self._infer_field_type(key, value),
                    "description": self._get_field_description(key)
                }
        
        return requirements
    
    def _check_critical_parameters(self, payload: ReplicatePayload, prompt: str, attachments: List[str], required_params: dict) -> list:
        """Check for critical missing parameters that prevent execution"""
        missing = []
        
        # Check for missing text input when required
        if self._requires_text_input() and not prompt:
            missing.append({
                "field": "prompt",
                "issue": "Text input is required but not provided",
                "suggested_fix": "Provide a text prompt or instruction",
                "auto_fixable": False
            })
        
        # Check for missing file input when required
        if self._requires_file_input() and not attachments:
            file_type = self._get_required_file_type()
            missing.append({
                "field": "file_input",
                "issue": f"{file_type.title()} file is required but not uploaded",
                "suggested_fix": f"Upload a {file_type} file",
                "auto_fixable": False
            })
        
        return missing
    
    def _requires_text_input(self) -> bool:
        """Check if this model requires text input"""
        model_lower = self.model_name.lower()
        text_required_models = ["nano-banana", "text-to-speech", "tts", "kokoro", "instruct", "chat"]
        return any(keyword in model_lower for keyword in text_required_models)
    
    def _requires_file_input(self) -> bool:
        """Check if this model requires file input"""
        model_lower = self.model_name.lower()
        file_required_models = ["remove-bg", "whisper", "upscal", "enhance", "clarity", "nano-banana", "image-edit"]
        return any(keyword in model_lower for keyword in file_required_models)
    
    def _get_required_file_type(self) -> str:
        """Get the type of file required by this model"""
        model_lower = self.model_name.lower()
        if any(keyword in model_lower for keyword in ["whisper", "transcrib", "speech"]):
            return "audio"
        elif any(keyword in model_lower for keyword in ["image", "upscal", "enhance", "remove-bg", "nano-banana"]):
            return "image"
        else:
            return "file"
    
    def _is_field_required(self, key: str, value: any) -> bool:
        """Determine if a field is required"""
        key_lower = key.lower()

        # Special case: system_prompt is optional configuration
        if "system" in key_lower and "prompt" in key_lower:
            return False

        # Fields that are typically required when present
        required_keywords = ["prompt", "text", "instruction", "image", "audio", "file", "input"]
        return any(keyword in key_lower for keyword in required_keywords) and not value
    
    def _infer_field_type(self, key: str, value: any) -> str:
        """Infer field type from key name and value"""
        key_lower = key.lower()
        if any(word in key_lower for word in ["image", "img", "photo"]):
            return "image"
        elif any(word in key_lower for word in ["audio", "sound", "speech"]):
            return "audio"
        elif any(word in key_lower for word in ["video", "clip"]):
            return "video"
        elif any(word in key_lower for word in ["prompt", "text", "instruction"]):
            return "text"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return "text"
    
    def _get_field_description(self, key: str) -> str:
        """Get human-readable description for field"""
        descriptions = {
            "prompt": "Describe what you want to create or do",
            "text": "Enter the text content",
            "instruction": "Provide instructions for the AI",
            "image": "Upload an image file",
            "input_image": "Upload the source image",
            "audio": "Upload an audio file",
            "file": "Upload a file",
            "strength": "Adjustment strength (0.0 to 1.0)",
            "steps": "Number of processing steps",
            "guidance_scale": "How closely to follow the prompt"
        }
        return descriptions.get(key.lower(), f"Set the {key} parameter")
    
    def execute(self, payload: ReplicatePayload) -> ProviderResponse:
        """Execute Replicate model request"""
        start_time = time.time()
        
        try:
            # Use existing run_replicate function
            run, status_code = run_replicate(
                run_input=self.run_input,
                model_params={
                    "example_input": self.example_input,
                    "latest_version": self.latest_version,
                },
                input=payload.input,
                operation_type=payload.operation_type.value,
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ProviderResponse(
                raw_response=run,
                processed_response=str(run),
                metadata={
                    "model_version": payload.model_version,
                    "operation_type": payload.operation_type.value,
                    "model_name": self.model_name
                },
                execution_time_ms=execution_time,
                status_code=status_code
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ProviderResponse(
                raw_response=None,
                processed_response="",
                metadata={},
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def audit_response(self, response: ProviderResponse) -> str:
        """Clean and audit Replicate response for user consumption"""
        if response.error:
            return f"Error: {response.error}"
        
        # Start with processed response
        cleaned_response = str(response.processed_response)
        
        # Remove provider-specific information
        cleaned_response = re.sub(r'https?://replicate\.com[^\s]*', '', cleaned_response)
        cleaned_response = re.sub(r'https?://[^\s]*\.replicate\.delivery[^\s]*', '[Generated Content]', cleaned_response)
        
        # Remove technical metadata
        cleaned_response = re.sub(r'\b(prediction|replicate|model)\b', '', cleaned_response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        
        return cleaned_response
    
    def estimate_cost(self, payload: ReplicatePayload) -> float:
        """Estimate cost of executing this payload"""
        base_cost = 0.01  # $0.01 base
        
        # Adjust based on operation type
        if payload.operation_type == OperationType.IMAGE_GENERATION:
            return base_cost * 2
        elif payload.operation_type == OperationType.VIDEO_GENERATION:
            return base_cost * 10
        elif payload.operation_type == OperationType.AUDIO_GENERATION:
            return base_cost * 3
        else:
            return base_cost
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for Replicate"""
        # This would ideally query Replicate API for actual limits
        return {
            "requests_remaining": 100,  # Placeholder
            "reset_time": time.time() + 3600,
            "current_usage": 0,
            "limit_per_minute": 60
        }
    
    def _infer_operation_type(self) -> OperationType:
        """Infer operation type from description and example input"""
        description_lower = self.description.lower()
        
        # Check description for keywords
        if any(word in description_lower for word in ["image", "picture", "photo", "visual", "generate image"]):
            if any(word in description_lower for word in ["edit", "modify", "change", "inpaint"]):
                return OperationType.IMAGE_EDITING
            return OperationType.IMAGE_GENERATION
        elif any(word in description_lower for word in ["video", "movie", "animation", "clip"]):
            return OperationType.VIDEO_GENERATION
        elif any(word in description_lower for word in ["audio", "sound", "music", "speech", "voice"]):
            return OperationType.AUDIO_GENERATION
        elif any(word in description_lower for word in ["embed", "vector", "similarity"]):
            return OperationType.EMBEDDING
        elif any(word in description_lower for word in ["classify", "category", "label"]):
            return OperationType.CLASSIFICATION
        else:
            return OperationType.TEXT_GENERATION
    
    def supports_streaming(self) -> bool:
        """Replicate doesn't typically support streaming"""
        return False
    
    def supports_cancellation(self) -> bool:
        """Replicate supports prediction cancellation"""
        return True
