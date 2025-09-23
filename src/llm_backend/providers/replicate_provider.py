"""
Replicate provider implementation wrapping existing ReplicateTeam logic
"""

import time
import re
from typing import Dict, Any, List

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
    
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict) -> ReplicatePayload:
        """Create Replicate-specific payload from generic inputs"""
        input_data = self.example_input.copy()
        
        # Map prompt to appropriate field
        prompt_fields = ["prompt", "text", "input", "query", "instruction"]
        for field in prompt_fields:
            if field in input_data:
                input_data[field] = prompt
                break
        
        # Map attachments to appropriate fields
        if attachments:
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
            metadata={"original_example": self.example_input}
        )
    
    def validate_payload(self, payload: ReplicatePayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        """Validate Replicate payload and return issues"""
        issues = []
        
        # Check if prompt is in payload
        prompt_found = any(prompt in str(value) for value in payload.input.values())
        if not prompt_found:
            issues.append(ValidationIssue(
                field="input",
                issue="Prompt not found in payload",
                severity="error",
                suggested_fix=f"Add '{prompt}' to one of: {list(payload.input.keys())}",
                auto_fixable=True
            ))
        
        # Check if required image is missing
        if attachments:
            image_found = any(att in str(payload.input) for att in attachments)
            if not image_found:
                issues.append(ValidationIssue(
                    field="input",
                    issue="Required image not found in payload",
                    severity="error",
                    suggested_fix="Map image to appropriate field",
                    auto_fixable=True
                ))
        
        # Check for required fields based on example_input
        for key, value in self.example_input.items():
            if key not in payload.input:
                issues.append(ValidationIssue(
                    field=key,
                    issue=f"Missing field from example_input: {key}",
                    severity="warning",
                    suggested_fix=f"Add {key} with default value: {value}",
                    auto_fixable=True
                ))
        
        # Check for empty required fields
        required_fields = ["prompt", "text", "input"]
        for field in required_fields:
            if field in payload.input and not payload.input[field]:
                issues.append(ValidationIssue(
                    field=field,
                    issue=f"Required field {field} is empty",
                    severity="error",
                    suggested_fix=f"Provide value for {field}",
                    auto_fixable=False
                ))
        
        return issues
    
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
