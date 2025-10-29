# Validation System Migration Guide

## Overview

The HITL validation system has been migrated from a checkpoint-based validation approach to a **Natural Language + Form-Based** approach. This document maps the old system to the new system.

---

## Old System → New System Mapping

### 1. HITLValidator Class → Form Field Classifier + NL Agents

**Old Approach:**
```python
from llm_backend.core.hitl.validation import HITLValidator, create_hitl_validation_summary

validator = HITLValidator(run_input=run_input, tool_config=tool_config)
checkpoints = validator.validate_pre_execution()
summary = create_hitl_validation_summary(checkpoints)
```

**New Approach:**
```python
from llm_backend.agents.form_field_classifier import classify_model_fields
from llm_backend.agents.nl_prompt_generator import generate_nl_prompt

# Classify fields from example_input
classification = await classify_model_fields(
    example_input=example_input,
    model_description=model_description
)

# Generate natural language prompt for missing fields
if missing_fields:
    nl_message = await generate_nl_prompt(
        classification=classification,
        current_values=current_values,
        missing_fields=missing_fields
    )
```

---

### 2. Validation Checkpoints → Form Validation

**Old Checkpoints:**
- `PARAMETER_REVIEW` - Checked if required parameters present
- `INPUT_VALIDATION` - Validated input quality
- `FILE_VALIDATION` - Checked for required files
- `MODEL_SELECTION` - Validated model compatibility
- `EXECUTION_APPROVAL` - Final approval gate

**New Validation:**
```python
from llm_backend.core.hitl.validation import validate_form_completeness

# Validate form has all required fields
validation_issues = validate_form_completeness(
    form_data=current_values,
    classification=classification
)

# Check for blocking issues
blocking_issues = [issue for issue in validation_issues if issue.severity == "error"]
```

---

### 3. Validation Messages → Natural Language Prompts

**Old System:**
- Generated structured validation messages
- Form-based UI with specific field errors
- Required understanding of technical field names

**Example Old Message:**
```json
{
  "checkpoint_type": "parameter_review",
  "validation_issues": [
    {
      "field": "input_image",
      "issue": "Required parameter 'input_image' is missing or empty",
      "severity": "error",
      "suggested_fix": "Please provide a value for input_image"
    }
  ]
}
```

**New System:**
- Natural language conversation
- User-friendly messages
- No technical jargon

**Example New Message:**
```
"I need a prompt describing the image you'd like to create. You can also adjust the aspect
ratio (currently set to 16:9) and choose the number of outputs (currently 1). What kind of
image do you have in mind?"
```

---

## Components Still in Use

### ✅ Form Validation Functions

These functions are **still actively used** and should NOT be removed:

```python
from llm_backend.core.hitl.validation import (
    validate_form_completeness,      # Validates all required fields filled
    validate_form_field_types,       # Validates field types are correct
    create_form_validation_summary   # Creates validation summary
)
```

**Usage Example:**
```python
# After user submits form
validation_issues = validate_form_completeness(form_data, classification)

if validation_issues:
    # Re-prompt user with natural language message
    nl_message = await generate_nl_prompt(
        classification=classification,
        current_values=form_data,
        missing_fields=[issue.field for issue in validation_issues]
    )
```

---

## Deprecated Components

### ❌ DO NOT USE (Legacy Fallback Only)

```python
# DEPRECATED - Only used in _legacy_information_review()
from llm_backend.core.hitl.validation import (
    HITLValidator,                    # Use form_field_classifier instead
    create_hitl_validation_summary,   # Use nl_prompt_generator instead
    CheckpointType,                   # Not used in new system
    ValidationCheckpoint              # Not used in new system
)
```

---

## Migration Steps

### For Existing Code Using HITLValidator

**Before:**
```python
validator = HITLValidator(run_input, tool_config)
checkpoints = validator.validate_pre_execution()
summary = create_hitl_validation_summary(checkpoints)

if summary['blocking_issues'] > 0:
    # Pause for human input
    await pause_for_approval(summary)
```

**After:**
```python
# 1. Initialize form from example_input
classification = await classify_model_fields(
    example_input=example_input,
    model_description=model_description
)

# 2. Check if all required fields are filled
validation_issues = validate_form_completeness(form_data, classification)

# 3. Generate natural language prompt if needed
if validation_issues:
    nl_message = await generate_nl_prompt(
        classification=classification,
        current_values=form_data,
        missing_fields=[issue.field for issue in validation_issues]
    )

    # 4. Pause with NL conversation mode
    await pause_for_nl_conversation(nl_message, form_data, classification)
```

---

## Key Differences

| Aspect | Old System | New System |
|--------|-----------|------------|
| **User Interface** | Technical forms with field names | Natural language conversation |
| **Validation** | Multiple checkpoint layers | Single form validation |
| **Messages** | Structured error objects | Human-friendly text |
| **Field Discovery** | Hardcoded per model | AI-powered classification |
| **User Input** | Fill specific fields | Conversational response |
| **Auto-Skip Logic** | Complex threshold system | Simple: all required fields present |

---

## Performance Improvements

**Old System:**
- ⏱️ 48+ seconds execution time
- 🔄 7+ OpenAI API calls per request
- 💰 ~$0.0035 per request
- ⏸️ Always paused even with blocking_issues=0

**New System:**
- ⚡ 5-10 seconds execution time
- 🔄 1-2 OpenAI API calls per request
- 💰 ~$0.0005 per request
- ✅ Auto-skips when all fields present

**Improvement:** 80% faster, 85% cheaper

---

## AI Agents in New System

### 1. Form Field Classifier
**File:** `src/llm_backend/agents/form_field_classifier.py`

**Purpose:** Analyzes `example_input` and classifies each field as CONTENT, CONFIG, or HYBRID.

**Usage:**
```python
classification = await classify_model_fields(
    example_input={"prompt": "", "aspect_ratio": "16:9", "num_outputs": 1},
    model_description="Text-to-image generation"
)

# Returns:
# {
#   "field_classifications": {
#     "prompt": {
#       "field_type": "CONTENT",
#       "required": True,
#       "user_prompt": "What would you like to generate?"
#     },
#     "aspect_ratio": {
#       "field_type": "CONFIG",
#       "required": False,
#       "default_value": "16:9"
#     }
#   }
# }
```

### 2. NL Prompt Generator
**File:** `src/llm_backend/agents/nl_prompt_generator.py`

**Purpose:** Generates natural language messages asking for missing information.

**Usage:**
```python
nl_message = await generate_nl_prompt(
    classification=classification,
    current_values={"aspect_ratio": "16:9"},
    missing_fields=["prompt"]
)

# Returns:
# "I need a prompt describing what you'd like to generate. The aspect ratio
#  is set to 16:9. What would you like me to create?"
```

### 3. NL Response Parser
**File:** `src/llm_backend/agents/nl_response_parser.py`

**Purpose:** Parses user's natural language response and extracts field values.

**Usage:**
```python
parsed_response = await parse_nl_response(
    user_response="Create a sunset over mountains in 1:1 format",
    classification=classification,
    current_values={}
)

# Returns:
# {
#   "prompt": "sunset over mountains",
#   "aspect_ratio": "1:1"
# }
```

### 4. Attachment Mapper
**File:** `src/llm_backend/agents/attachment_mapper.py`

**Purpose:** Maps user attachments to appropriate fields semantically.

**Features:**
- Understands field equivalence: "image" = "input_image" = "img" = "photo"
- Detects file types from URLs
- Handles both single fields and arrays
- Prioritizes CONTENT fields over CONFIG fields

---

## Testing

### Test Natural Language HITL

```bash
# Run comprehensive test suite
python test_nl_hitl_request.py

# Run quick bash tests
./test_nl_hitl_simple.sh
```

### Expected Results

**Scenario 1: Complete Request**
- ✅ Auto-skips information_review
- ✅ No pause
- ✅ ~5-10 seconds execution

**Scenario 2: Missing Fields**
- ✅ Pauses with NL message
- ✅ User responds in natural language
- ✅ System extracts field values

---

## Troubleshooting

### Issue: Still seeing old validation checkpoints

**Solution:** Ensure `form_data` is being initialized. The legacy validator only runs when `form_data` is missing:

```python
# In orchestrator.py
if not self.state.form_data:
    print("⚠️ No form_data found - falling back to legacy validation")
    return await self._legacy_information_review()  # ← Old system
```

### Issue: Natural language not activating

**Solution:** Check that `use_natural_language_hitl=True` in HITLConfig:

```python
hitl_config = HITLConfig(
    policy="auto_with_thresholds",
    use_natural_language_hitl=True  # ← Must be True
)
```

---

## References

- **Implementation Plan:** `docs/HITL_NATURAL_LANGUAGE_IMPLEMENTATION.md`
- **Testing Guide:** `TESTING_NL_HITL.md`
- **Form-Based HITL:** `docs/FORM_BASED_HITL.md`
- **Architecture:** `docs/ARCHITECTURE.md`
