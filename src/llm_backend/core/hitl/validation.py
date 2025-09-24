"""
HITL Validation System - Pre-execution checkpoints and parameter validation
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from ..types.common import RunInput
from .types import ValidationIssue


class CheckpointType(Enum):
    PARAMETER_REVIEW = "parameter_review"
    INPUT_VALIDATION = "input_validation"
    MODEL_SELECTION = "model_selection"
    FILE_VALIDATION = "file_validation"
    EXECUTION_APPROVAL = "execution_approval"


@dataclass
class ValidationCheckpoint:
    """Represents a validation checkpoint in the HITL workflow"""
    checkpoint_type: CheckpointType
    title: str
    description: str
    required: bool
    validation_issues: List[ValidationIssue]
    user_input_required: bool = False
    auto_fixable: bool = False
    
    def is_blocking(self) -> bool:
        """Check if this checkpoint blocks execution"""
        return self.required and (
            any(issue.severity == "error" for issue in self.validation_issues) or
            self.user_input_required
        )


class HITLValidator:
    """Comprehensive validation system for HITL workflows"""
    
    def __init__(self, run_input: RunInput, tool_config: Dict[str, Any]):
        self.run_input = run_input
        self.tool_config = tool_config
        self.model_name = tool_config.get("model_name", "")
        self.example_input = tool_config.get("example_input", {})
        self.description = tool_config.get("description", "")
    
    def validate_pre_execution(self) -> List[ValidationCheckpoint]:
        """Run all pre-execution validation checkpoints"""
        checkpoints = []
        
        # 1. Parameter Review Checkpoint
        param_checkpoint = self._validate_parameters()
        checkpoints.append(param_checkpoint)
        
        # 2. Input Validation Checkpoint
        input_checkpoint = self._validate_inputs()
        checkpoints.append(input_checkpoint)
        
        # 3. File Validation Checkpoint
        file_checkpoint = self._validate_files()
        checkpoints.append(file_checkpoint)
        
        # 4. Model Selection Checkpoint
        model_checkpoint = self._validate_model_selection()
        checkpoints.append(model_checkpoint)
        
        # 5. Final Execution Approval Checkpoint
        execution_checkpoint = self._validate_execution_readiness(checkpoints)
        checkpoints.append(execution_checkpoint)
        
        return checkpoints
    
    def _validate_parameters(self) -> ValidationCheckpoint:
        """Validate that all required parameters are present and valid"""
        issues = []
        
        # Check for missing prompt when required
        if self._requires_text_input() and not self.run_input.prompt:
            issues.append(ValidationIssue(
                field="prompt",
                issue="Text prompt is required but not provided",
                severity="error",
                suggested_fix="Please provide a clear text prompt describing what you want to do",
                auto_fixable=False
            ))
        
        # Check for vague or unclear prompts
        if self.run_input.prompt and len(self.run_input.prompt.strip()) < 10:
            issues.append(ValidationIssue(
                field="prompt",
                issue="Prompt is too short or vague",
                severity="warning",
                suggested_fix="Please provide more detailed instructions",
                auto_fixable=False
            ))
        
        # Validate example_input completeness
        required_fields = self._get_required_fields()
        for field in required_fields:
            if field not in self.example_input or not self.example_input[field]:
                issues.append(ValidationIssue(
                    field=field,
                    issue=f"Required parameter '{field}' is missing or empty",
                    severity="error",
                    suggested_fix=f"Please provide a value for {field}",
                    auto_fixable=False
                ))
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.PARAMETER_REVIEW,
            title="Parameter Review",
            description="Verify all required parameters are present and valid",
            required=True,
            validation_issues=issues,
            user_input_required=len(issues) > 0
        )
    
    def _validate_inputs(self) -> ValidationCheckpoint:
        """Validate input quality and completeness"""
        issues = []
        
        # Check prompt quality for specific model types
        if self.run_input.prompt:
            prompt_issues = self._analyze_prompt_quality(self.run_input.prompt)
            issues.extend(prompt_issues)
        
        # Check for conflicting parameters
        conflict_issues = self._check_parameter_conflicts()
        issues.extend(conflict_issues)
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.INPUT_VALIDATION,
            title="Input Validation",
            description="Validate input quality and detect conflicts",
            required=True,
            validation_issues=issues,
            user_input_required=any(issue.severity == "error" for issue in issues)
        )
    
    def _validate_files(self) -> ValidationCheckpoint:
        """Validate uploaded files and attachments"""
        issues = []
        
        # Check for required files
        if self._requires_file_input():
            if not self.run_input.document_url:
                file_type = self._get_required_file_type()
                issues.append(ValidationIssue(
                    field="file_input",
                    issue=f"Required {file_type} file is missing",
                    severity="error",
                    suggested_fix=f"Please upload a {file_type} file",
                    auto_fixable=False
                ))
            else:
                # Validate file type and accessibility
                file_issues = self._validate_file_accessibility()
                issues.extend(file_issues)
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.FILE_VALIDATION,
            title="File Validation",
            description="Verify required files are uploaded and accessible",
            required=self._requires_file_input(),
            validation_issues=issues,
            user_input_required=any(issue.severity == "error" for issue in issues)
        )
    
    def _validate_model_selection(self) -> ValidationCheckpoint:
        """Validate model selection and configuration"""
        issues = []
        
        # Check if model is appropriate for the task
        if not self.model_name:
            issues.append(ValidationIssue(
                field="model_name",
                issue="No model selected",
                severity="error",
                suggested_fix="Please select an appropriate model",
                auto_fixable=False
            ))
        else:
            # Validate model compatibility with inputs
            compatibility_issues = self._check_model_compatibility()
            issues.extend(compatibility_issues)
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.MODEL_SELECTION,
            title="Model Selection",
            description="Verify the selected model is appropriate for the task",
            required=True,
            validation_issues=issues,
            user_input_required=len(issues) > 0
        )
    
    def _validate_execution_readiness(self, previous_checkpoints: List[ValidationCheckpoint]) -> ValidationCheckpoint:
        """Final checkpoint to confirm execution readiness"""
        issues = []
        
        # Check if any previous checkpoints are blocking
        blocking_checkpoints = [cp for cp in previous_checkpoints if cp.is_blocking()]
        
        if blocking_checkpoints:
            for checkpoint in blocking_checkpoints:
                issues.append(ValidationIssue(
                    field="execution_readiness",
                    issue=f"Checkpoint '{checkpoint.title}' has blocking issues",
                    severity="error",
                    suggested_fix="Please resolve the issues in the previous checkpoints",
                    auto_fixable=False
                ))
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.EXECUTION_APPROVAL,
            title="Execution Approval",
            description="Final confirmation before model execution",
            required=True,
            validation_issues=issues,
            user_input_required=len(issues) == 0  # Only require approval if no blocking issues
        )
    
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
    
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for this model"""
        model_lower = self.model_name.lower()
        
        if "remove-bg" in model_lower:
            return ["image", "input_image"]
        elif "whisper" in model_lower:
            return ["audio", "file"]
        elif "text-to-speech" in model_lower or "tts" in model_lower:
            return ["text", "prompt"]
        elif "nano-banana" in model_lower:
            return ["image", "prompt", "instruction"]
        else:
            # Default required fields
            return [field for field in self.example_input.keys() 
                   if any(keyword in field.lower() for keyword in ["prompt", "text", "image", "audio"])]
    
    def _analyze_prompt_quality(self, prompt: str) -> List[ValidationIssue]:
        """Analyze prompt quality for specific model types"""
        issues = []
        model_lower = self.model_name.lower()
        
        # Check for image editing models
        if "nano-banana" in model_lower or "image-edit" in model_lower:
            if not any(word in prompt.lower() for word in ["change", "edit", "transform", "make", "add", "remove"]):
                issues.append(ValidationIssue(
                    field="prompt",
                    issue="Prompt should specify what changes to make to the image",
                    severity="warning",
                    suggested_fix="Add specific editing instructions like 'change the color to blue' or 'remove the background'",
                    auto_fixable=False
                ))
        
        # Check for text-to-speech models
        elif "text-to-speech" in model_lower or "tts" in model_lower:
            if len(prompt) > 500:
                issues.append(ValidationIssue(
                    field="prompt",
                    issue="Text is very long for speech synthesis",
                    severity="warning",
                    suggested_fix="Consider breaking into shorter segments for better quality",
                    auto_fixable=False
                ))
        
        return issues
    
    def _check_parameter_conflicts(self) -> List[ValidationIssue]:
        """Check for conflicting parameters"""
        issues = []
        
        # Example: Check for conflicting image parameters
        image_fields = [field for field in self.example_input.keys() 
                       if "image" in field.lower()]
        
        if len(image_fields) > 1 and self.run_input.document_url:
            # Multiple image fields but only one file uploaded
            issues.append(ValidationIssue(
                field="image_parameters",
                issue=f"Model has multiple image fields ({image_fields}) but only one file uploaded",
                severity="warning",
                suggested_fix="Clarify which image field should use the uploaded file",
                auto_fixable=True
            ))
        
        return issues
    
    def _validate_file_accessibility(self) -> List[ValidationIssue]:
        """Validate that uploaded files are accessible"""
        issues = []
        
        if self.run_input.document_url:
            # Basic URL validation
            if not self.run_input.document_url.startswith(("http://", "https://", "data:")):
                issues.append(ValidationIssue(
                    field="document_url",
                    issue="File URL appears to be invalid",
                    severity="error",
                    suggested_fix="Please upload a valid file",
                    auto_fixable=False
                ))
        
        return issues
    
    def _check_model_compatibility(self) -> List[ValidationIssue]:
        """Check if model is compatible with provided inputs"""
        issues = []
        model_lower = self.model_name.lower()
        
        # Check image models with text-only input
        if any(keyword in model_lower for keyword in ["image", "visual", "photo"]):
            if not self.run_input.document_url and self.run_input.prompt:
                issues.append(ValidationIssue(
                    field="model_compatibility",
                    issue="Image model selected but no image provided",
                    severity="warning",
                    suggested_fix="Upload an image file or select a text-based model",
                    auto_fixable=False
                ))
        
        # Check audio models with non-audio input
        if any(keyword in model_lower for keyword in ["whisper", "audio", "speech"]):
            if self.run_input.document_url and not any(ext in self.run_input.document_url.lower() 
                                                      for ext in [".mp3", ".wav", ".m4a", ".flac"]):
                issues.append(ValidationIssue(
                    field="model_compatibility",
                    issue="Audio model selected but uploaded file may not be audio",
                    severity="warning",
                    suggested_fix="Upload an audio file (.mp3, .wav, .m4a, .flac)",
                    auto_fixable=False
                ))
        
        return issues


def create_hitl_validation_summary(checkpoints: List[ValidationCheckpoint]) -> Dict[str, Any]:
    """Create a summary of validation results for HITL UI"""
    total_issues = sum(len(cp.validation_issues) for cp in checkpoints)
    blocking_issues = sum(len([issue for issue in cp.validation_issues if issue.severity == "error"]) 
                         for cp in checkpoints)
    
    return {
        "total_checkpoints": len(checkpoints),
        "passed_checkpoints": len([cp for cp in checkpoints if not cp.validation_issues]),
        "failed_checkpoints": len([cp for cp in checkpoints if cp.validation_issues]),
        "blocking_checkpoints": len([cp for cp in checkpoints if cp.is_blocking()]),
        "total_issues": total_issues,
        "blocking_issues": blocking_issues,
        "ready_for_execution": blocking_issues == 0,
        "checkpoints": [
            {
                "type": cp.checkpoint_type.value,
                "title": cp.title,
                "description": cp.description,
                "required": cp.required,
                "passed": len(cp.validation_issues) == 0,
                "blocking": cp.is_blocking(),
                "user_input_required": cp.user_input_required,
                "issues": [
                    {
                        "field": issue.field,
                        "issue": issue.issue,
                        "severity": issue.severity,
                        "suggested_fix": issue.suggested_fix,
                        "auto_fixable": issue.auto_fixable
                    }
                    for issue in cp.validation_issues
                ]
            }
            for cp in checkpoints
        ]
    }
