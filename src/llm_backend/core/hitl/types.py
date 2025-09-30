"""
HITL orchestrator types and enums
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime
import uuid


class HITLPolicy(str, Enum):
    """HITL execution policies"""
    AUTO = "auto"
    REQUIRE_HUMAN = "require_human"
    AUTO_WITH_THRESHOLDS = "auto_with_thresholds"


class HITLStep(str, Enum):
    """HITL workflow steps"""
    CREATED = "created"
    INFORMATION_REVIEW = "information_review"
    PAYLOAD_REVIEW = "payload_review"
    API_CALL = "api_call"
    RESPONSE_REVIEW = "response_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HITLStatus(str, Enum):
    """HITL run status"""
    QUEUED = "queued"
    AWAITING_HUMAN = "awaiting_human"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationIssue(BaseModel):
    """Validation issue found during parameter checking"""
    field: str
    issue: str
    severity: str  # "error" | "warning"
    suggested_fix: str
    auto_fixable: bool = False


class HITLConfig(BaseModel):
    """HITL configuration"""
    policy: HITLPolicy = HITLPolicy.AUTO
    review_thresholds: Optional[Dict[str, Any]] = None
    allowed_steps: List[HITLStep] = []
    timeout_seconds: int = 3600
    auto_approve_confidence_threshold: float = 0.9
    require_approval_safety_flags: List[str] = ["nsfw", "pii", "copyright"]
    max_payload_changes: int = 5
    enable_streaming: bool = False


class StepEvent(BaseModel):
    """Record of a step event"""
    step: HITLStep
    status: HITLStatus
    timestamp: datetime = datetime.utcnow()
    actor: str  # "system", "human", "agent_name"
    message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class HITLState(BaseModel):
    """Complete state of a HITL run"""
    run_id: str = str(uuid.uuid4())
    current_step: HITLStep = HITLStep.CREATED
    status: HITLStatus = HITLStatus.QUEUED
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    expires_at: Optional[datetime] = None
    
    # Configuration
    config: HITLConfig
    original_input: Dict[str, Any]
    
    # Step artifacts
    capabilities: Optional[Dict[str, Any]] = None
    suggested_payload: Optional[Dict[str, Any]] = None
    validation_issues: List[Dict[str, Any]] = []
    validation_checkpoints: Optional[Dict[str, Any]] = None
    validation_summary: Optional[Dict[str, Any]] = None
    user_friendly_message: Optional[str] = None
    raw_response: Optional[Any] = None
    processed_response: Optional[str] = None
    final_result: Optional[str] = None
    
    # Human interactions
    pending_actions: List[str] = []
    approval_token: Optional[str] = None
    checkpoint_context: Optional[Dict[str, Any]] = None
    last_approval: Optional[Dict[str, Any]] = None
    human_edits: Dict[str, Any] = {}
    
    # Audit trail
    step_history: List[StepEvent] = []
    
    # Metrics
    total_execution_time_ms: int = 0
    human_review_time_ms: int = 0
    provider_execution_time_ms: int = 0
