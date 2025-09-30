"""
Replicate provider implementation wrapping existing ReplicateTeam logic
"""

import time
import re
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
    
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict, hitl_edits: Dict = None) -> ReplicatePayload:
        """Create Replicate-specific payload using intelligent agent-based field mapping"""
        from llm_backend.agents.replicate_team import ReplicateTeam
        from llm_backend.core.types.replicate import ExampleInput
        import asyncio
        
        if self.run_input is None:
            raise ValueError("ReplicateProvider requires run_input to be set before creating payloads")

        replicate_team = ReplicateTeam(
            prompt=prompt,
            tool_config={
                "example_input": self.example_input,
                "description": self.description,
                "latest_version": self.latest_version,
                "model_name": self.model_name
            },
            run_input=self.run_input,
            hitl_enabled=True  # Enable intelligent mapping
        )
        
        # Prepare agent input with HITL edits
        agent_input = ExampleInput(
            prompt=prompt,
            example_input=self.example_input,
            description=self.description,
            image_file=attachments[0] if attachments else None,
            hitl_edits=hitl_edits  # Pass HITL edits for intelligent integration
        )
        
        # Use AI agent as primary logic for intelligent payload creation
        try:
            replicate_agent = replicate_team.replicate_agent()

            async def run_agent() -> Any:
                print(f"🔍 Agent deps: {agent_input.model_dump()}")
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

            if hitl_edits:
                agent_input_payload = self._apply_hitl_edits(agent_input_payload, hitl_edits)

            print(f"🤖 Agent created payload: {agent_input_payload}")

            metadata = {
                "original_example": self.example_input,
                "agent_generated": True,
                "agent_messages": getattr(agent_payload, "messages", None),
                "hitl_applied": bool(hitl_edits)
            }

            return ReplicatePayload(
                provider_name="replicate",
                input=agent_input_payload,
                operation_type=operation_type,
                model_version=self.latest_version,
                metadata=metadata
            )

        except Exception as e:
            print(f"⚠️ Agent payload creation failed: {e}")
            print(f"🔍 Agent input details: prompt='{prompt}', hitl_edits={hitl_edits}, example_input={self.example_input}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            print("🔧 Falling back to static mapping")

        print(f"🔧 Using enhanced fallback mapping with HITL edits: {hitl_edits}")

        # Fallback to static mapping if agent fails
        input_data = self._apply_hitl_edits(self.example_input.copy(), hitl_edits)
        
        # Map prompt to appropriate field
        prompt_fields = ["prompt", "text", "input", "query", "instruction"]
        for field in prompt_fields:
            if field in input_data:
                input_data[field] = prompt
                break
        
        # Map attachments to appropriate fields (only if no HITL edits provided)
        if attachments and not hitl_edits:
            image_fields = [
                "image", "image_url", "input_image", "first_frame_image", 
                "subject_reference", "start_image", "init_image"
            ]
            for field in image_fields:
                if field in input_data and attachments:
                    input_data[field] = attachments[0]
                    break
        
        return ReplicatePayload(
            provider_name="replicate",
            input=input_data,
            operation_type=operation_type,
            model_version=self.latest_version,
            metadata={"original_example": self.example_input, "fallback_used": True, "hitl_mapped": bool(hitl_edits)}
        )

    def _apply_hitl_edits(self, base_input: Dict[str, Any], hitl_edits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply human edits onto a payload input dict with smart alias handling."""
        if not hitl_edits:
            return base_input

        updated_input = base_input.copy()

        alias_map = {
            "input_image": ["input_image", "source_image", "image", "image_url"],
            "source_image": ["source_image", "input_image", "image", "image_url"],
            "image": ["image", "input_image", "source_image", "image_url"],
            "driven_audio": ["driven_audio", "audio_file", "audio"],
            "audio_file": ["audio_file", "driven_audio", "audio"],
            "prompt": ["prompt", "text", "instruction", "input"],
        }

        for key, value in hitl_edits.items():
            if value in (None, ""):
                continue

            targets = alias_map.get(key, [key])
            applied = False

            for target in targets:
                if target in updated_input:
                    updated_input[target] = value
                    applied = True

            if not applied:
                primary_target = targets[0]
                updated_input[primary_target] = value

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
