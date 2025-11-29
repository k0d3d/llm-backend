from pydantic import BaseModel, ConfigDict, field_validator
from typing import Any, Dict, List, Literal, Optional


class Props(BaseModel):
    all_props: List[str]
    affected_props: List[str]

class PayloadInput(BaseModel):
    # allow any property
    model_config = ConfigDict(extra="allow")

class OperationType(BaseModel):
    type: Literal['image', 'video', 'text', 'audio']


class AgentPayload(BaseModel):
    input: PayloadInput
    operationType: OperationType

    @field_validator('input')
    @classmethod
    def input_not_empty(cls, v: Dict) -> Dict:
        if not v:
            raise ValueError('input dictionary cannot be empty')
        return v


class ExampleInput(BaseModel):
    prompt: str
    example_input: dict
    description: str
    props: Optional[Props] = None
    image_file: Optional[str] = None
    video_file: Optional[str] = None
    attachments: Optional[List[str]] = None
    hitl_edits: Optional[Dict[str, Any]] = None  # HITL human edits to integrate
    schema_metadata: Optional[Dict[str, Any]] = None  # Example input schema hints
    hitl_field_metadata: Optional[Dict[str, Any]] = None  # Alias/collection hints for edits
    structured_form_values: Optional[Dict[str, Any]] = None  # NEW: Structured values from form-based HITL
    conversation: Optional[List[Dict[str, Any]]] = None  # Chat history for context (allows dict props)


class AttachmentDiscoveryContext(BaseModel):
    prompt: str
    text_context: List[str] = []
    candidate_urls: List[str] = []
    schema_metadata: Dict[str, Any] = {}
    hitl_field_metadata: Dict[str, Any] = {}
    expected_media_fields: List[str] = []


class AttachmentDiscoveryResult(BaseModel):
    attachments: List[str]
    mapping: Optional[Dict[str, str]] = None
    reasoning: Optional[str] = None

class InformationInputResponse(BaseModel):
    continue_run: bool
    response_information: str

class InformationInputPayload(BaseModel):
    example_input: dict
    description: str
    attached_file: Optional[str] = None


class ValidationIssueDetail(BaseModel):
    field: str
    issue: str
    severity: Literal["info", "warning", "error"] = "error"
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


class FileRequirementContext(BaseModel):
    prompt: str
    example_input: Dict[str, Any]
    model_name: str
    model_description: str
    existing_payload: Dict[str, Any]
    hitl_edits: Dict[str, Any] = {}
    attachments: List[str] = []


class FileRequirementAnalysis(BaseModel):
    required_files: List[str]
    blocking_issues: List[ValidationIssueDetail]
    ready: bool
    suggestions: List[str] = []


class PayloadValidationContext(BaseModel):
    prompt: str
    example_input: Dict[str, Any]
    candidate_payload: AgentPayload
    required_files: List[str] = []
    hitl_edits: Dict[str, Any] = {}
    attachments: List[str] = []
    operation_type: Literal['image', 'video', 'text', 'audio'] = 'image'


class PayloadValidationOutput(BaseModel):
    payload: AgentPayload
    blocking_issues: List[ValidationIssueDetail]
    warnings: List[ValidationIssueDetail] = []
    ready: bool = True
    auto_fixes: Dict[str, Any] = {}
    summary: Optional[str] = None


class FinalGuardContext(BaseModel):
    prompt: str
    example_input: Dict[str, Any]
    candidate_payload: AgentPayload
    model_name: str
    model_description: str
    operation_type: Literal['image', 'video', 'text', 'audio'] = 'image'


class FinalGuardDecision(BaseModel):
    approved: bool
    payload: AgentPayload
    blocking_issues: List[ValidationIssueDetail] = []
    diff_summary: Optional[str] = None
