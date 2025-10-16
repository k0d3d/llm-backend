"""
HITL Validation System - Pre-execution checkpoints and parameter validation
"""
from typing import Dict, List, Any, Optional, Tuple
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
        # Extract model name from 'name' field (not 'model_name')
        self.model_name = tool_config.get("name", "") or tool_config.get("model_name", "")
        self.example_input = tool_config.get("example_input", {})
        self.description = tool_config.get("description", "")
        self.hitl_alias_metadata = tool_config.get("hitl_alias_metadata", {}) or {}
        self.field_metadata = tool_config.get("field_metadata", {}) or {}
        print(f"🔍 HITLValidator init: model_name='{self.model_name}' from tool_config keys: {list(tool_config.keys())}")
    
    def validate_pre_execution(self) -> List[ValidationCheckpoint]:
        """Run all pre-execution validation checkpoints"""
        print(f"🔍 HITLValidator: Starting validation for model: {self.model_name}")
        print(f"📝 Prompt: {self.run_input.prompt}")
        # Note: document_url removed; attachments will be discovered by orchestrator if needed
        print(f"⚙️ Tool config: {self.tool_config}")
        
        checkpoints = []
        
        # 1. Parameter Review Checkpoint
        print("🔍 Running Parameter Review checkpoint...")
        param_checkpoint = self._validate_parameters()
        print(f"📊 Parameter checkpoint issues: {len(param_checkpoint.validation_issues)}")
        for issue in param_checkpoint.validation_issues:
            print(f"  ❌ {issue.severity}: {issue.issue}")
        checkpoints.append(param_checkpoint)
        
        # 2. Input Validation Checkpoint
        print("🔍 Running Input Validation checkpoint...")
        input_checkpoint = self._validate_inputs()
        print(f"📊 Input checkpoint issues: {len(input_checkpoint.validation_issues)}")
        for issue in input_checkpoint.validation_issues:
            print(f"  ❌ {issue.severity}: {issue.issue}")
        checkpoints.append(input_checkpoint)
        
        # 3. File Validation Checkpoint
        print("🔍 Running File Validation checkpoint...")
        file_checkpoint = self._validate_files()
        print(f"📊 File checkpoint issues: {len(file_checkpoint.validation_issues)}")
        for issue in file_checkpoint.validation_issues:
            print(f"  ❌ {issue.severity}: {issue.issue}")
        print(f"🔍 File checkpoint requires file input: {self._requires_file_input()}")
        print(f"🔍 Model name for file check: '{self.model_name}'")
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
        
        # Validate example_input completeness (textual parameters)
        required_fields = self._get_required_fields()
        for field in required_fields:
            if not self._has_any_field_value([field]):
                issues.append(ValidationIssue(
                    field=field,
                    issue=f"Required parameter '{field}' is missing or empty",
                    severity="error",
                    suggested_fix=f"Please provide a value for {field}",
                    auto_fixable=False
                ))

        # Provide guidance (non-blocking) for media inputs that will be auto-resolved later
        media_requirement = self._get_media_requirement()
        if media_requirement:
            canonical_field, alias_fields = media_requirement
            if not self._has_any_field_value(alias_fields):
                issues.append(ValidationIssue(
                    field=canonical_field,
                    issue="Media input will be auto-discovered from recent attachments unless you upload one.",
                    severity="warning",
                    suggested_fix="Attach the desired asset in chat or provide a direct URL if you want to override the default.",
                    auto_fixable=True
                ))

        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.PARAMETER_REVIEW,
            title="Parameter Review",
            description="Verify all required parameters are present and valid",
            required=True,
            validation_issues=issues,
            user_input_required=any(issue.severity == "error" for issue in issues)
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
            # No explicit document_url anymore; warn and let orchestrator attempt discovery
            # Do not block execution: orchestrator will attempt to gather attachments from chat history
            file_type = self._get_required_file_type()
            issues.append(ValidationIssue(
                field="file_input",
                issue=f"No {file_type} file explicitly provided; will attempt to discover from chat history",
                severity="warning",
                suggested_fix=f"Optionally attach a {file_type} file or select a recent asset from the thread",
                auto_fixable=False
            ))
        
        return ValidationCheckpoint(
            checkpoint_type=CheckpointType.FILE_VALIDATION,
            title="File Validation",
            description="Verify files are available and accessible (auto-discovers recent assets if not provided)",
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
        
        if any(keyword in model_lower for keyword in ["text-to-speech", "tts"]):
            return ["text", "prompt"]

        if "nano-banana" in model_lower:
            return ["prompt", "instruction"]

        # Inspect alias metadata for textual requirements
        alias_text_fields: List[str] = []
        for canonical_name, meta in (self.hitl_alias_metadata or {}).items():
            name_lower = canonical_name.lower()
            if any(keyword in name_lower for keyword in ["prompt", "text", "instruction"]):
                alias_text_fields.append(canonical_name)

        if alias_text_fields:
            # Preserve order while removing duplicates
            seen = set()
            unique_fields = []
            for field in alias_text_fields:
                if field not in seen:
                    seen.add(field)
                    unique_fields.append(field)
            return unique_fields

        # Fallback to example input heuristics, ignoring media fields
        textual_fields = [
            field for field in self.example_input.keys()
            if any(keyword in field.lower() for keyword in ["prompt", "text", "instruction"])
            and not self._is_media_field(field)
        ]
        return textual_fields
    
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
        # Without explicit file input, we cannot determine conflicts at this stage
        
        return issues
    
    def _validate_file_accessibility(self) -> List[ValidationIssue]:
        """Validate that uploaded files are accessible"""
        issues = []
        # No explicit file URL is part of RunInput anymore; accessibility checks happen later when attachments are resolved
        return issues
    
    def _check_model_compatibility(self) -> List[ValidationIssue]:
        """Check if model is compatible with provided inputs"""
        issues = []
        model_lower = self.model_name.lower()
        
        # Check image models with text-only input
        if any(keyword in model_lower for keyword in ["image", "visual", "photo", "remove-bg", "nano-banana"]):
            media_requirement = self._get_media_requirement()
            if media_requirement:
                canonical_field, alias_fields = media_requirement
                if not self._has_any_field_value(alias_fields):
                    issues.append(ValidationIssue(
                        field=canonical_field,
                        issue="Image input will be auto-selected from chat history unless you upload a replacement.",
                        severity="warning",
                        suggested_fix="Attach the desired image or keep the referenced URL in your prompt.",
                        auto_fixable=True
                    ))

        # Check audio models with non-audio input
        if any(keyword in model_lower for keyword in ["whisper", "audio", "speech"]):
            media_requirement = self._get_media_requirement(media_type="audio")
            if media_requirement:
                canonical_field, alias_fields = media_requirement
                if not self._has_any_field_value(alias_fields):
                    issues.append(ValidationIssue(
                        field=canonical_field,
                        issue="Audio input will be auto-selected from chat history unless overridden.",
                        severity="warning",
                        suggested_fix="Attach an audio clip or provide an accessible URL.",
                        auto_fixable=True
                    ))

        return issues

    def _has_any_field_value(self, candidate_fields: List[str]) -> bool:
        """Check example input for any non-empty value across alias variants"""
        if not candidate_fields:
            return False

        fields_to_check = set(candidate_fields)

        for field in candidate_fields:
            alias_meta = (self.hitl_alias_metadata or {}).get(field, {})
            for alias in alias_meta.get("targets", []):
                fields_to_check.add(alias)

        for field in fields_to_check:
            value = self.example_input.get(field)
            if value not in (None, "", [], {}):
                return True

        return False

    def _get_media_requirement(self, media_type: str = "image") -> Optional[Tuple[str, List[str]]]:
        """Determine canonical media field and its aliases from provider metadata"""
        alias_metadata = self.hitl_alias_metadata or {}

        if media_type == "image":
            keywords = ["image", "photo", "frame"]
        elif media_type == "audio":
            keywords = ["audio", "sound", "voice"]
        else:
            keywords = [media_type]

        for canonical_name, meta in alias_metadata.items():
            name_lower = canonical_name.lower()
            if any(keyword in name_lower for keyword in keywords):
                targets = meta.get("targets") or [canonical_name]
                return canonical_name, targets

        # Fallback: inspect example input keys when alias metadata is unavailable
        for field in self.example_input.keys():
            lower_field = field.lower()
            if any(keyword in lower_field for keyword in keywords):
                return field, [field]

        return None

    def _is_media_field(self, field_name: str) -> bool:
        """Determine if a field represents media input"""
        if not field_name:
            return False

        lowered = field_name.lower()
        return any(token in lowered for token in ["image", "photo", "audio", "sound", "video", "frame", "clip"])


def create_hitl_validation_summary(checkpoints: List[ValidationCheckpoint]) -> Dict[str, Any]:
    """Create a summary of validation results for HITL UI"""
    total_issues = sum(len(cp.validation_issues) for cp in checkpoints)
    blocking_issues = sum(len([issue for issue in cp.validation_issues if issue.severity == "error"])
                         for cp in checkpoints)
    
    # Generate user-friendly message based on issues
    user_message = "Ready to proceed"
    if blocking_issues > 0:
        # Check what type of files are needed
        needs_image = any(issue.field == "input_image" for cp in checkpoints for issue in cp.validation_issues)
        needs_audio = any(issue.field == "input_audio" for cp in checkpoints for issue in cp.validation_issues)
        
        if needs_image and needs_audio:
            user_message = "This model needs both an image and audio file to work. Please upload the required files."
        elif needs_image:
            user_message = "This model needs an image to work. Please upload an image file."
        elif needs_audio:
            user_message = "This model needs an audio file to work. Please upload an audio file."
        else:
            user_message = "There are some issues that need your attention before we can continue."
    
    return {
        "total_checkpoints": len(checkpoints),
        "passed_checkpoints": len([cp for cp in checkpoints if not cp.validation_issues]),
        "failed_checkpoints": len([cp for cp in checkpoints if cp.validation_issues]),
        "blocking_checkpoints": len([cp for cp in checkpoints if cp.is_blocking()]),
        "total_issues": total_issues,
        "blocking_issues": blocking_issues,
        "ready_for_execution": blocking_issues == 0,
        "user_friendly_message": user_message,
        "checkpoints": [
            {
                "type": cp.checkpoint_type.value,
                "title": cp.title,
                "description": cp.description,
                "required": cp.required,
                "passed": len(cp.validation_issues) == 0,
                "blocking": cp.is_blocking(),
                "user_input_required": cp.user_input_required,
                "checkpoint_type": cp.checkpoint_type.value,
                "issues": [
                    {
                        "field": issue.field,
                        "issue": issue.issue,
                        "severity": issue.severity,
                        "message": issue.issue,  # Add message field for UI compatibility
                        "suggestion": issue.suggested_fix,  # Add suggestion field for UI compatibility
                        "suggested_fix": issue.suggested_fix,
                        "auto_fixable": issue.auto_fixable
                    }
                    for issue in cp.validation_issues
                ]
            }
            for cp in checkpoints
        ]
    }


def validate_form_completeness(
    form_data: Dict[str, Any],
    classification: Dict[str, Any]
) -> List[ValidationIssue]:
    """
    Validate that form has all required fields filled

    Args:
        form_data: Current form values
        classification: AI classification with field requirements

    Returns:
        List of validation issues for missing/invalid fields
    """
    issues = []
    field_classifications = classification.get("field_classifications", {})

    for field_name, field_class in field_classifications.items():
        if not field_class.get("required"):
            continue

        value = form_data.get(field_name)

        # Check required field is not empty
        if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
            issues.append(ValidationIssue(
                field=field_name,
                issue=f"Required field '{field_name}' is empty",
                severity="error",
                suggested_fix=field_class.get("user_prompt", f"Please provide {field_name}"),
                auto_fixable=False
            ))

    return issues


def validate_form_field_types(
    form_data: Dict[str, Any],
    classification: Dict[str, Any]
) -> List[ValidationIssue]:
    """
    Validate that form field values match expected types

    Args:
        form_data: Current form values
        classification: AI classification with field types

    Returns:
        List of validation issues for type mismatches
    """
    issues = []
    field_classifications = classification.get("field_classifications", {})

    for field_name, value in form_data.items():
        if field_name not in field_classifications:
            continue

        field_class = field_classifications[field_name]
        expected_type = field_class.get("value_type")

        # Skip validation for None/empty values (handled by completeness check)
        if value is None or value == "":
            continue

        # Type validation
        type_valid = True
        if expected_type == "array" and not isinstance(value, list):
            type_valid = False
        elif expected_type == "object" and not isinstance(value, dict):
            type_valid = False
        elif expected_type in ["integer", "number"] and not isinstance(value, (int, float)):
            # Try to convert if it's a string number
            if isinstance(value, str):
                try:
                    float(value) if expected_type == "number" else int(value)
                except ValueError:
                    type_valid = False
            else:
                type_valid = False
        elif expected_type == "boolean" and not isinstance(value, bool):
            type_valid = False

        if not type_valid:
            issues.append(ValidationIssue(
                field=field_name,
                issue=f"Field '{field_name}' has wrong type (expected {expected_type}, got {type(value).__name__})",
                severity="warning",
                suggested_fix=f"Please provide a valid {expected_type} value",
                auto_fixable=False
            ))

    return issues


def create_form_validation_summary(form_data: Dict[str, Any], classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive validation summary for form data

    Args:
        form_data: Current form values
        classification: AI classification with field requirements

    Returns:
        Validation summary with blocking issues count and details
    """
    completeness_issues = validate_form_completeness(form_data, classification)
    type_issues = validate_form_field_types(form_data, classification)

    all_issues = completeness_issues + type_issues
    blocking_issues = [issue for issue in all_issues if issue.severity == "error"]

    return {
        "blocking_issues": len(blocking_issues),
        "total_issues": len(all_issues),
        "completeness_issues": [issue.model_dump() for issue in completeness_issues],
        "type_issues": [issue.model_dump() for issue in type_issues],
        "all_issues": [issue.model_dump() for issue in all_issues],
        "is_valid": len(blocking_issues) == 0,
        "user_friendly_message": (
            "Form is complete and valid"
            if len(blocking_issues) == 0
            else f"{len(blocking_issues)} required field(s) need attention"
        )
    }
