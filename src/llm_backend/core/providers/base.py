"""
Base provider interface and data models for HITL system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
import time


class OperationType(str, Enum):
    """Supported operation types across providers"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_GENERATION = "video_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    WEB_AUTOMATION = "web_automation"


class ProviderPayload(BaseModel):
    """Base class for provider-specific payloads"""
    provider_name: str
    created_at: datetime = datetime.utcnow()
    metadata: Dict[str, Any] = {}


class ProviderResponse(BaseModel):
    """Base class for provider responses"""
    raw_response: Any
    processed_response: str
    metadata: Dict[str, Any]
    execution_time_ms: int
    status_code: Optional[int] = None
    error: Optional[str] = None


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
        self.run_input = None  # Will be set by orchestrator
    
    def set_run_input(self, run_input):
        """Set run input for provider execution"""
        # Handle both Pydantic models and plain dicts
        if hasattr(run_input, 'model_dump'):
            self.run_input = run_input
        elif isinstance(run_input, dict):
            # Convert dict to RunInput if needed
            from llm_backend.core.types.common import RunInput
            try:
                self.run_input = RunInput(**run_input)
            except Exception:
                # If conversion fails, store as-is for backward compatibility
                self.run_input = run_input
        else:
            self.run_input = run_input
    
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
