# Form-Based HITL Workflow

## Overview

The HITL system has been enhanced with a form-based workflow that treats `example_input` as a form definition rather than just a validation template. This document describes the new flow and implementation.

## Key Concept

**Before:** Validate against example_input → Find issues → User fixes issues
**After:** Create form from example_input → Prompt for required fields → User fills form → Use form as payload

## Architecture Changes

### 1. New AI Agent: FormFieldClassifierAgent

**File:** `src/llm_backend/agents/form_field_classifier.py`

The agent classifies each field in `example_input` into categories:
- **CONTENT** - User must provide (images, prompts, etc.) → Reset to empty
- **CONFIG** - Optional parameters with sensible defaults → Keep default values
- **HYBRID** - Optional content → Reset but not required

**Key Features:**
- Uses gpt-4.1-mini-mini for intelligent classification
- Fallback heuristic classification if AI fails
- Handles nested objects recursively
- Generates user-friendly prompts for each field

### 2. New AI Agent: AttachmentMappingAgent

**File:** `src/llm_backend/agents/attachment_mapper.py`

The agent intelligently maps user-provided attachments (files/URLs) to form fields using semantic understanding:

**Key Features:**
- Uses gpt-4.1-mini-mini for semantic field name matching
- Understands equivalence: "image" = "input_image" = "img" = "photo"
- Detects file types from URLs: `.jpg` → image fields, `.mp3` → audio fields
- Handles both single string fields AND array fields (not just arrays)
- Prioritizes CONTENT category fields over CONFIG fields
- Provides confidence scores and reasoning for each mapping
- Falls back to improved heuristic matching if AI fails

**Problem Solved:**
Previous hardcoded logic only checked array fields, missing single fields like `{"image": "..."}`. Now handles all field types with AI-powered semantic matching.

**Example Mapping:**
```python
# Input
user_attachments = ["https://serve.com/photo.jpg"]
field_classifications = {
    "image": {"category": "CONTENT", "collection": False}
}

# AI Output
{
    "field_name": "image",
    "attachment": "https://serve.com/photo.jpg",
    "confidence": 0.95,
    "file_type": "image",
    "reasoning": "Image file matches CONTENT image field"
}
```

### 2b. Schema-Aware Attachment Fallback

**File:** `src/llm_backend/providers/replicate_provider.py` and `src/llm_backend/agents/attachment_resolver.py`

When AI agents fail (e.g., "Exceeded maximum retries") or produce payloads missing attachment fields, a schema-aware fallback cascade ensures user attachments are still mapped correctly:

**Fallback Cascade:**
1. **Replace existing placeholders**: If payload has fields with `replicate.delivery` URLs, replace with user attachment
2. **Schema field matching**: Check if common fields (`input_image`, `image`, `image_input`, etc.) exist in `example_input`
3. **Heuristic field detection**: Find any schema field with attachment-like name (`image`, `photo`, `file`, `source`, `media`)
4. **URL value detection**: Find any schema field containing a URL value (likely an attachment placeholder)
5. **Warning**: Log if no suitable field found, preventing silent failures

**Critical Design Decision:**
- Fields are only added if they exist in the model's `example_input` schema
- This prevents `_filter_payload_to_schema()` from removing the attachment field
- Avoids hardcoded field names that may not exist in a specific model's schema

**Example Fallback Flow:**
```python
# User provides image, but AI agent outputs:
{'prompt': 'change the room'}  # Missing image_input!

# Fallback checks example_input schema:
example_input = {'image_input': 'https://replicate.delivery/...', 'prompt': '...'}

# Finds 'image_input' in schema → adds user attachment:
{'prompt': 'change the room', 'image_input': 'https://user-photo.jpg'}
```

**Log Messages:**
- `🔧 Manual replacement: {field} -> {url}` - Replaced placeholder
- `🔧 Manual addition (schema match): {field} = {url}` - Added to known field
- `🔧 Manual addition (schema heuristic): {field} = {url}` - Added via heuristic
- `🔧 Manual addition (URL field): {field} = {url}` - Added to URL-value field
- `⚠️ Manual fallback: Could not find attachment field in schema` - No field found

### 3. New HITL Step: FORM_INITIALIZATION

**File:** `src/llm_backend/core/hitl/types.py`

Added new step to the workflow pipeline:
```python
class HITLStep(str, Enum):
    CREATED = "created"
    FORM_INITIALIZATION = "form_initialization"  # NEW
    INFORMATION_REVIEW = "information_review"
    PAYLOAD_REVIEW = "payload_review"
    API_CALL = "api_call"
    RESPONSE_REVIEW = "response_review"
    COMPLETED = "completed"
```

### 3. Enhanced HITLState

**File:** `src/llm_backend/core/hitl/types.py`

Added `form_data` field to track form state:
```python
form_data: Optional[Dict[str, Any]] = None
"""
Form data structure:
{
    "schema": {...},  # Original example_input
    "classification": {...},  # AI agent field classifications
    "defaults": {...},  # Default values per field
    "current_values": {...},  # Current form state (reset/defaults applied)
    "user_edits": {...}  # User-provided values
}
"""
```

## Workflow Steps

### Step 1: Form Initialization

**Method:** `orchestrator._step_form_initialization()`

1. Extract `example_input` from provider
2. Extract URLs from user prompt and gather explicit attachments
3. Call `FormFieldClassifierAgent` to classify fields (CONTENT/CONFIG/HYBRID)
4. Call `AttachmentMappingAgent` to map user attachments to appropriate fields
5. Build form applying reset logic with user-provided values:
   - **ALL arrays → empty []** (unless pre-populated from attachments)
   - **CONTENT fields → null or ""** (unless pre-populated from attachments)
   - **CONFIG fields → keep defaults**
6. Store form data in state

**Key Enhancement:** Attachments are now intelligently mapped before form initialization, so fields are pre-populated when the user has already provided the required content.

**Example:**
```python
# example_input
{
  "image": "https://example.com/demo.jpg",
  "negative_prompts": ["blurry", "low quality"],
  "guidance_scale": 7.5,
  "num_steps": 50
}

# User provides
user_attachments = ["https://serve.com/photo.jpg"]

# After AttachmentMappingAgent
attachment_mapping = {"image": "https://serve.com/photo.jpg"}  # AI mapped!

# After form initialization (WITH pre-population)
{
  "image": "https://serve.com/photo.jpg",  # PRE-POPULATED from attachment!
  "negative_prompts": [],  # RESET (always empty arrays)
  "guidance_scale": 7.5,  # KEEP (config field)
  "num_steps": 50  # KEEP (config field)
}

# Result: No prompt for image field - already filled!
```

### Step 2: Information Review (Form Prompting)

**Method:** `orchestrator._step_information_review()`

1. Check form completeness
2. Identify missing required fields
3. Build form field definitions for UI
4. Send form to user via WebSocket
5. Wait for user to submit form
6. Apply form submissions via `_apply_form_submission()`

**WebSocket Message Format:**
```json
{
  "type": "hitl_approval_request",
  "data": {
    "checkpoint_type": "form_requirements",
    "form": {
      "title": "Configure Model Parameters",
      "fields": [
        {
          "name": "input_image",
          "label": "Input Image",
          "type": "file",
          "required": true,
          "current_value": null,
          "prompt": "Upload an image to process",
          "collection": false
        },
        {
          "name": "negative_prompts",
          "label": "Negative Prompts",
          "type": "array",
          "required": false,
          "current_value": [],
          "prompt": "Optionally provide negative prompts",
          "collection": true,
          "hint": "You can add multiple items"
        }
      ]
    },
    "required_fields": ["input_image"],
    "optional_fields": ["negative_prompts"],
    "missing_required_fields": ["input_image"]
  }
}
```

### Step 3: Form Submission Handling

**Method:** `orchestrator._apply_form_submission()`

Handles user form submissions:
- **For single values:** Direct assignment
- **For arrays:** Append items or replace entire array
- Stores in `form_data.user_edits` and `form_data.current_values`
- Persists to database

**Example:**
```python
# User submits
{
  "input_image": "https://user-upload.com/my-image.jpg",
  "negative_prompts": ["blurry"]
}

# Result in form_data.current_values
{
  "input_image": "https://user-upload.com/my-image.jpg",  # Updated
  "negative_prompts": ["blurry"],  # Updated
  "guidance_scale": 7.5,  # Unchanged (default kept)
  "num_steps": 50  # Unchanged (default kept)
}
```

### Step 4: Direct Payload Creation

**Method:** `replicate_provider._create_payload_from_form()`

Bypasses intelligent agent mapping:
1. Take `form_data.current_values` directly
2. Apply schema coercion (ensure correct types)
3. Filter to only include fields from `example_input`
4. Return as payload

**Flow:**
```
Form Values → Schema Coercion → Schema Filtering → Payload
```

No AI agent field mapping needed - form values are used directly!

## Array Handling

### Reset Behavior
**ALL arrays are emptied on form initialization**, regardless of category:
```python
# example_input
{
  "input_images": ["demo1.jpg", "demo2.jpg"],
  "config_list": [{"key": "value"}]
}

# After reset
{
  "input_images": [],  # EMPTY - user must provide
  "config_list": []  # EMPTY - even config arrays reset
}
```

### Submission Behavior
When user submits array values:
- **List provided → Replace entire array**
- **Single item provided → Append to array**

```python
# User submits: {"input_images": ["my-image.jpg"]}
# Result: input_images = ["my-image.jpg"]

# User submits: {"input_images": "another-image.jpg"}
# Result: input_images = ["my-image.jpg", "another-image.jpg"]
```

## Nested Objects

Classification and reset logic applied recursively:
```python
# example_input
{
  "input": {
    "image": "demo.jpg",
    "scale": 1.0
  }
}

# Classified as
{
  "input": {
    "image": null,  # RESET (content)
    "scale": 1.0  # KEEP (config)
  }
}
```

## Backward Compatibility

### Legacy Fallback
If form initialization fails or `form_data` is not available:
- Falls back to `_legacy_information_review()`
- Uses validation-based approach
- Maintains existing behavior

### Provider Support
Providers without `set_orchestrator()` method:
- Use intelligent agent for payload creation
- No changes to existing behavior

## Validation

### Form Completeness Validation

**File:** `src/llm_backend/core/hitl/validation.py`

New functions added:
- `validate_form_completeness()` - Check required fields filled
- `validate_form_field_types()` - Check field types match
- `create_form_validation_summary()` - Comprehensive validation

**Usage:**
```python
from llm_backend.core.hitl.validation import create_form_validation_summary

summary = create_form_validation_summary(
    form_data={"input_image": None, "scale": 7.5},
    classification=classification
)

# Returns
{
  "blocking_issues": 1,
  "total_issues": 1,
  "is_valid": False,
  "user_friendly_message": "1 required field(s) need attention",
  "all_issues": [
    {
      "field": "input_image",
      "issue": "Required field 'input_image' is empty",
      "severity": "error",
      "suggested_fix": "Upload an image to process"
    }
  ]
}
```

## WebSocket Integration

The WebSocket bridge already supports form-based messages through flexible `checkpoint_type` and `context` parameters.

**New checkpoint type:**
- `form_requirements` - Form-based field collection

**Message structure automatically includes:**
- Form field definitions
- Current values
- Required vs optional fields
- User prompts

## File Changes Summary

### New Files
1. `src/llm_backend/agents/form_field_classifier.py` - AI field classifier agent

### Modified Files
1. `src/llm_backend/core/hitl/types.py`
   - Added `HITLStep.FORM_INITIALIZATION`
   - Added `form_data` to `HITLState`

2. `src/llm_backend/core/hitl/orchestrator.py`
   - Added `_step_form_initialization()`
   - Added `_build_form_from_classification()`
   - Added `_extract_defaults_from_classification()`
   - Modified `_step_information_review()` for form prompting
   - Added `_legacy_information_review()` fallback
   - Added `_apply_form_submission()`
   - Added `_infer_ui_field_type()`
   - Linked orchestrator to provider

3. `src/llm_backend/providers/replicate_provider.py`
   - Added `_orchestrator` attribute
   - Added `set_orchestrator()` method
   - Added `_create_payload_from_form()` method
   - Modified `create_payload()` to check form data first

4. `src/llm_backend/core/hitl/validation.py`
   - Added `validate_form_completeness()`
   - Added `validate_form_field_types()`
   - Added `create_form_validation_summary()`

5. `src/llm_backend/core/hitl/websocket_bridge.py`
   - Updated documentation for `form_requirements` checkpoint type

## Testing Recommendations

1. **Unit Tests**
   - Test field classification with various `example_input` structures
   - Test array reset logic
   - Test nested object handling
   - Test form submission handling

2. **Integration Tests**
   - Test full form workflow end-to-end
   - Test fallback to legacy validation
   - Test form validation logic

3. **Edge Cases**
   - Empty `example_input`
   - Deeply nested objects
   - Mixed array types
   - Missing required fields

## Migration Notes

### For Existing Workflows
- Form-based workflow activates automatically when `example_input` is present
- Falls back to legacy validation if initialization fails
- No breaking changes to existing code

### For Frontend
- Watch for `checkpoint_type: "form_requirements"` messages
- Render form UI based on field definitions
- Submit form values via approval response `edits` field
- Handle array fields with "can add multiple" hint

## Benefits

1. **Clearer UX** - "Fill this form" vs "Fix these errors"
2. **Simpler Logic** - Direct form-to-payload mapping
3. **Better Defaults** - Keep sensible config values
4. **Array Clarity** - Always start with empty arrays
5. **Type Safety** - AI classifies field types upfront
6. **Nested Support** - Recursive handling of complex schemas
