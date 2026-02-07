from typing import List, Any, Dict
from pydantic import BaseModel
from .schema_extractor import ModelSchema, FieldType

class ValidationIssue(BaseModel):
    field: str
    issue: str
    severity: str  # "error" (blocking) or "warning"
    suggested_fix: str = ""

class Validator:
    @staticmethod
    def validate(payload_input: Dict[str, Any], schema: ModelSchema) -> List[ValidationIssue]:
        issues = []
        
        # 1. Check Required Fields
        for field_name, field_def in schema.fields.items():
            val = payload_input.get(field_name)
            
            # Check if missing
            # Note: 0 or False are valid values. None or "" or [] are considered "missing" for required checks.
            is_empty = val is None or val == "" or (isinstance(val, list) and len(val) == 0)
            
            if field_def.is_required and is_empty:
                # Special case: If schema has a default, it's technically not "missing" if we are allowed to inject the default.
                # But here we assume the payload_input SHOULD have had it populated by the Builder.
                # If the Builder left it empty, it means it couldn't find a source.
                
                # However, if there IS a default value in the schema, we might accept it (but Builder should have put it there).
                # Let's be strict: If it's required and missing in payload, it's an error.
                issues.append(ValidationIssue(
                    field=field_name,
                    issue="Field is required but missing",
                    severity="error",
                    suggested_fix=f"Provide a value for {field_name}"
                ))
                continue
            
            # 2. Type Checking (Basic)
            if val is not None:
                if field_def.type == FieldType.INTEGER and not isinstance(val, int):
                     issues.append(ValidationIssue(
                        field=field_name,
                        issue=f"Expected integer, got {type(val).__name__}",
                        severity="error"
                    ))
                # Add more type checks as needed
                
        return issues
