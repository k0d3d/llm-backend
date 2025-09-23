# Provider Interface Specification

This document defines the `AIProvider` interface that all AI service providers must implement to work with the HITL orchestration system.

## Base Classes

### AIProvider Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum

class ProviderPayload(BaseModel):
    """Base class for provider-specific payloads"""
    provider_name: str
    created_at: datetime
    metadata: Dict[str, Any] = {}

class ProviderResponse(BaseModel):
    """Base class for provider responses"""
    raw_response: Any
    processed_response: str
    metadata: Dict[str, Any]
    execution_time_ms: int
    status_code: Optional[int] = None
    error: Optional[str] = None

class OperationType(str, Enum):
    """Supported operation types across providers"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_GENERATION = "video_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"

class ProviderCapabilities(BaseModel):
    """Describes what a provider can do"""
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    supported_operations: List[OperationType]
    safety_features: List[str]
    rate_limits: Dict[str, int]
    cost_per_request: Optional[float] = None
    max_input_size: Optional[int] = None
    max_output_size: Optional[int] = None

class ValidationIssue(BaseModel):
    """Represents a validation issue with suggested fix"""
    field: str
    issue: str
    severity: str  # "error", "warning", "info"
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities and schema"""
        pass
    
    @abstractmethod
    def validate_payload(self, payload: ProviderPayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        """Validate payload and return list of issues/suggestions"""
        pass
    
    @abstractmethod
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict) -> ProviderPayload:
        """Create provider-specific payload from generic inputs"""
        pass
    
    @abstractmethod
    def execute(self, payload: ProviderPayload) -> ProviderResponse:
        """Execute the request with the provider"""
        pass
    
    @abstractmethod
    def audit_response(self, response: ProviderResponse) -> str:
        """Clean/audit the response for user consumption"""
        pass
    
    @abstractmethod
    def estimate_cost(self, payload: ProviderPayload) -> float:
        """Estimate cost of executing this payload"""
        pass
    
    @abstractmethod
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass
    
    # Optional methods with default implementations
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses"""
        return False
    
    def supports_cancellation(self) -> bool:
        """Whether this provider supports request cancellation"""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        return {
            "name": self.config.get("model_name", "unknown"),
            "version": self.config.get("model_version", "unknown"),
            "provider": self.provider_name
        }
```

## Provider-Specific Implementations

### ReplicateProvider

```python
from llm_backend.core.providers.base import AIProvider, ProviderPayload, ProviderResponse, ProviderCapabilities, ValidationIssue, OperationType
from llm_backend.tools.replicate_tool import run_replicate

class ReplicatePayload(ProviderPayload):
    input: Dict[str, Any]
    operation_type: OperationType
    model_version: str
    webhook_url: Optional[str] = None

class ReplicateProvider(AIProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.example_input = config.get("example_input", {})
        self.description = config.get("description", "")
        self.latest_version = config.get("latest_version", "")
        self.model_name = config.get("name", "")
    
    def get_capabilities(self) -> ProviderCapabilities:
        # Infer operation type from description and example_input
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
        )
    
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict) -> ReplicatePayload:
        # Map generic inputs to Replicate-specific format
        input_data = self.example_input.copy()
        
        # Map prompt to appropriate field
        prompt_fields = ["prompt", "text", "input", "query"]
        for field in prompt_fields:
            if field in input_data:
                input_data[field] = prompt
                break
        
        # Map attachments to appropriate fields
        if attachments:
            image_fields = ["image", "image_url", "input_image", "first_frame_image", "subject_reference", "start_image"]
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
        issues = []
        
        # Check if prompt is in payload
        if not any(prompt in str(value) for value in payload.input.values()):
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
                    issue=f"Missing required field: {key}",
                    severity="warning",
                    suggested_fix=f"Add {key} with default value: {value}",
                    auto_fixable=True
                ))
        
        return issues
    
    def execute(self, payload: ReplicatePayload) -> ProviderResponse:
        start_time = time.time()
        
        try:
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
                    "operation_type": payload.operation_type.value
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
        if response.error:
            return f"Error: {response.error}"
        
        # Remove provider-specific information
        cleaned_response = str(response.processed_response)
        
        # Remove URLs to replicate.com
        cleaned_response = re.sub(r'https?://replicate\.com[^\s]*', '', cleaned_response)
        
        # Remove technical metadata
        cleaned_response = re.sub(r'\b(prediction|replicate|model)\b', '', cleaned_response, flags=re.IGNORECASE)
        
        return cleaned_response.strip()
    
    def estimate_cost(self, payload: ReplicatePayload) -> float:
        # Estimate based on input size and operation type
        base_cost = 0.01  # $0.01 base
        
        if payload.operation_type == OperationType.IMAGE_GENERATION:
            return base_cost * 2
        elif payload.operation_type == OperationType.VIDEO_GENERATION:
            return base_cost * 10
        else:
            return base_cost
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        return {
            "requests_remaining": 100,  # Would query Replicate API
            "reset_time": time.time() + 3600,
            "current_usage": 0
        }
    
    def _infer_operation_type(self) -> OperationType:
        """Infer operation type from description and example input"""
        description_lower = self.description.lower()
        
        if any(word in description_lower for word in ["image", "picture", "photo", "visual"]):
            if any(word in description_lower for word in ["edit", "modify", "change"]):
                return OperationType.IMAGE_EDITING
            return OperationType.IMAGE_GENERATION
        elif any(word in description_lower for word in ["video", "movie", "animation"]):
            return OperationType.VIDEO_GENERATION
        elif any(word in description_lower for word in ["audio", "sound", "music", "speech"]):
            return OperationType.AUDIO_GENERATION
        else:
            return OperationType.TEXT_GENERATION
```

## Implementation Guidelines

### Error Handling
- Always wrap provider calls in try-catch blocks
- Return structured error information in `ProviderResponse`
- Log errors with sufficient context for debugging

### Validation
- Implement comprehensive payload validation
- Provide actionable error messages and suggested fixes
- Support auto-fixing of common issues where possible

### Performance
- Track execution times for monitoring
- Implement appropriate timeouts
- Handle rate limiting gracefully

### Security
- Sanitize inputs to prevent injection attacks
- Redact sensitive information from logs
- Validate file uploads and URLs

### Testing
- Provide mock implementations for testing
- Include unit tests for all validation logic
- Test error conditions and edge cases

## Provider Registration

```python
# llm_backend/core/providers/registry.py
from typing import Dict, Type
from llm_backend.core.providers.base import AIProvider

class ProviderRegistry:
    _providers: Dict[str, Type[AIProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[AIProvider]):
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, name: str, config: Dict) -> AIProvider:
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name](config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        return list(cls._providers.keys())

# Register providers
ProviderRegistry.register("replicate", ReplicateProvider)
ProviderRegistry.register("openai", OpenAIProvider)
ProviderRegistry.register("anthropic", AnthropicProvider)
```

## Usage Example

```python
from llm_backend.core.providers.registry import ProviderRegistry

# Get provider
provider = ProviderRegistry.get_provider("replicate", config)

# Get capabilities
capabilities = provider.get_capabilities()

# Create payload
payload = provider.create_payload(
    prompt="Generate a sunset image",
    attachments=[],
    operation_type=OperationType.IMAGE_GENERATION,
    config={}
)

# Validate
issues = provider.validate_payload(payload, "Generate a sunset image", [])

# Execute if valid
if not any(issue.severity == "error" for issue in issues):
    response = provider.execute(payload)
    result = provider.audit_response(response)
```
