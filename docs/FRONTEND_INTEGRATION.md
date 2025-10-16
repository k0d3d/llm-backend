# HITL Frontend Integration Guide

This guide provides comprehensive instructions for integrating the Human-in-the-Loop (HITL) system with React-based frontend applications.

## Table of Contents

1. [Overview](#overview)
2. [Form-Based HITL Workflow](#form-based-hitl-workflow) **NEW**
3. [API Integration](#api-integration)
4. [WebSocket Integration](#websocket-integration)
5. [Session Resume & State Management](#session-resume--state-management)
6. [React Components](#react-components)
7. [State Management](#state-management)
8. [Error Handling](#error-handling)
9. [User Experience Patterns](#user-experience-patterns)
10. [Complete Examples](#complete-examples)

## Overview

The HITL system provides enhanced AI workflow control with human oversight through:
- **Form-based parameter collection** **NEW** - Interactive forms built from model schemas
- **Pre-execution validation checkpoints** - Legacy validation-based approach
- **Real-time WebSocket communication**
- **Interactive approval workflows**
- **Comprehensive error handling**
- **Parameter validation and guidance**

### HITL Workflow Modes

The system supports two modes:

1. **Form-Based Mode (NEW)** - Recommended for new integrations
   - Treats `example_input` as a form definition
   - Prompts users for required fields upfront
   - Direct form-to-payload mapping (no AI guessing)
   - Clear distinction between content fields and config parameters

2. **Validation-Based Mode (Legacy)** - Backward compatible
   - Validates against `example_input` schema
   - Shows errors when parameters are missing
   - User fixes issues after validation
   - Falls back automatically if form initialization fails

## Form-Based HITL Workflow

The new form-based workflow provides a superior UX by treating model parameters as an interactive form rather than validation errors to fix.

### How It Works

1. **Backend**: AI classifies each field in `example_input` as:
   - **CONTENT** - User must provide (images, prompts) → Reset to empty
   - **CONFIG** - Optional parameters with defaults (steps, scale) → Keep defaults
   - **HYBRID** - Optional content → Empty but not required

2. **Frontend**: Receives form definition with field metadata
3. **User**: Fills required fields, optionally adjusts configs
4. **Submission**: Form values used directly as payload (no AI field mapping)

### WebSocket Message Format

#### Form Requirements Checkpoint

When the backend pauses for form input, you'll receive:

```typescript
interface FormRequirementsMessage {
  type: "hitl_approval_request";
  data: {
    approval_id: string;
    run_id: string;
    checkpoint_type: "form_requirements"; // NEW checkpoint type
    context: {
      message: string;
      checkpoint_type: "form_requirements";
      form: {
        title: string;
        fields: FormField[];
      };
      required_fields: string[];
      optional_fields: string[];
      missing_required_fields: string[];
    };
  };
}

interface FormField {
  name: string;                    // Field identifier
  label: string;                   // Display label
  type: "text" | "number" | "file" | "checkbox" | "array"; // UI field type
  category: "CONTENT" | "CONFIG" | "HYBRID"; // Field classification
  required: boolean;               // Is field required?
  current_value: any;              // Current value (null for reset fields)
  default?: any;                   // Default value if applicable
  prompt: string;                  // User-friendly prompt
  collection: boolean;             // Is this an array field?
  hint?: string;                   // Additional guidance (e.g., "You can add multiple items")
}
```

#### Example Form Requirements Message

```json
{
  "type": "hitl_approval_request",
  "data": {
    "approval_id": "approval-123",
    "run_id": "run-456",
    "checkpoint_type": "form_requirements",
    "context": {
      "message": "Please provide required information to continue",
      "checkpoint_type": "form_requirements",
      "form": {
        "title": "Configure Model Parameters",
        "fields": [
          {
            "name": "input_image",
            "label": "Input Image",
            "type": "file",
            "category": "CONTENT",
            "required": true,
            "current_value": null,
            "prompt": "Upload an image to process",
            "collection": false
          },
          {
            "name": "negative_prompts",
            "label": "Negative Prompts",
            "type": "array",
            "category": "CONTENT",
            "required": false,
            "current_value": [],
            "prompt": "Optionally provide negative prompts",
            "collection": true,
            "hint": "You can add multiple items"
          },
          {
            "name": "guidance_scale",
            "label": "Guidance Scale",
            "type": "number",
            "category": "CONFIG",
            "required": false,
            "current_value": 7.5,
            "default": 7.5,
            "prompt": "Adjust guidance scale (optional)",
            "collection": false
          }
        ]
      },
      "required_fields": ["input_image"],
      "optional_fields": ["negative_prompts", "guidance_scale"],
      "missing_required_fields": ["input_image"]
    }
  }
}
```

### Form Submission

Submit user-provided values via the standard approval endpoint:

```typescript
// Submit form values
await submitApproval(
  runId,
  approvalId,
  'approve', // or 'edit'
  {
    // Form field values provided by user
    input_image: "https://user-upload.com/image.jpg",
    negative_prompts: ["blurry", "low quality"],
    guidance_scale: 8.0 // User changed from default 7.5
  },
  undefined, // reason (optional)
  userId
);
```

### React Component for Form-Based HITL

```typescript
import React, { useState } from 'react';

interface FormBasedHITLCardProps {
  hitlData: {
    run_id: string;
    context: {
      message: string;
      checkpoint_type: string;
      form?: {
        title: string;
        fields: FormField[];
      };
      required_fields: string[];
      optional_fields: string[];
      missing_required_fields: string[];
    };
  };
  onApprovalComplete?: (approved: boolean, runId: string) => void;
}

export const FormBasedHITLCard: React.FC<FormBasedHITLCardProps> = ({
  hitlData,
  onApprovalComplete
}) => {
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Check if this is a form requirements checkpoint
  const isFormCheckpoint = hitlData.context.checkpoint_type === 'form_requirements';
  const form = hitlData.context.form;

  if (!isFormCheckpoint || !form) {
    // Fall back to legacy validation card
    return <LegacyHITLValidationCard hitlData={hitlData} onApprovalComplete={onApprovalComplete} />;
  }

  const handleFieldChange = (fieldName: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [fieldName]: value
    }));

    // Clear error for this field
    if (errors[fieldName]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[fieldName];
        return newErrors;
      });
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Check required fields
    hitlData.context.required_fields.forEach(fieldName => {
      const value = formValues[fieldName];
      if (value === undefined || value === null || value === '' ||
          (Array.isArray(value) && value.length === 0)) {
        newErrors[fieldName] = `${fieldName} is required`;
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    try {
      await submitApproval(
        hitlData.run_id,
        hitlData.run_id, // approval_id
        'approve',
        formValues, // Form data as edits
        undefined,
        currentUserId
      );
      onApprovalComplete?.(true, hitlData.run_id);
    } catch (error) {
      console.error('Failed to submit form:', error);
      setErrors({ _form: 'Failed to submit form. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderField = (field: FormField) => {
    const value = formValues[field.name] ?? field.current_value;
    const hasError = errors[field.name];

    switch (field.type) {
      case 'file':
        return (
          <div key={field.name} className="mb-3">
            <label className="form-label">
              {field.label}
              {field.required && <span className="text-danger"> *</span>}
            </label>
            <input
              type="file"
              className={`form-control ${hasError ? 'is-invalid' : ''}`}
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  // Handle file upload - convert to URL or upload to server
                  const reader = new FileReader();
                  reader.onload = () => {
                    handleFieldChange(field.name, reader.result);
                  };
                  reader.readAsDataURL(file);
                }
              }}
            />
            <div className="form-text text-muted">{field.prompt}</div>
            {hasError && <div className="invalid-feedback d-block">{hasError}</div>}
          </div>
        );

      case 'array':
        return (
          <div key={field.name} className="mb-3">
            <label className="form-label">
              {field.label}
              {field.required && <span className="text-danger"> *</span>}
            </label>
            <div className="input-group">
              <input
                type="text"
                className={`form-control ${hasError ? 'is-invalid' : ''}`}
                placeholder={field.prompt}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    const input = e.currentTarget;
                    if (input.value.trim()) {
                      const currentArray = Array.isArray(value) ? value : [];
                      handleFieldChange(field.name, [...currentArray, input.value.trim()]);
                      input.value = '';
                    }
                  }
                }}
              />
              <button
                className="btn btn-outline-secondary btn-sm"
                type="button"
                onClick={(e) => {
                  const input = e.currentTarget.previousElementSibling as HTMLInputElement;
                  if (input.value.trim()) {
                    const currentArray = Array.isArray(value) ? value : [];
                    handleFieldChange(field.name, [...currentArray, input.value.trim()]);
                    input.value = '';
                  }
                }}
              >
                Add
              </button>
            </div>
            {field.hint && <div className="form-text text-muted">{field.hint}</div>}
            {Array.isArray(value) && value.length > 0 && (
              <div className="mt-2">
                {value.map((item, idx) => (
                  <span key={idx} className="badge bg-primary me-1">
                    {item}
                    <button
                      type="button"
                      className="btn-close btn-close-white ms-1"
                      style={{ fontSize: '10px' }}
                      onClick={() => {
                        const newArray = value.filter((_, i) => i !== idx);
                        handleFieldChange(field.name, newArray);
                      }}
                    />
                  </span>
                ))}
              </div>
            )}
            {hasError && <div className="invalid-feedback d-block">{hasError}</div>}
          </div>
        );

      case 'number':
        return (
          <div key={field.name} className="mb-3">
            <label className="form-label">
              {field.label}
              {field.required && <span className="text-danger"> *</span>}
              {field.default !== undefined && (
                <span className="text-muted ms-2">(default: {field.default})</span>
              )}
            </label>
            <input
              type="number"
              className={`form-control ${hasError ? 'is-invalid' : ''}`}
              value={value ?? ''}
              onChange={(e) => handleFieldChange(field.name, parseFloat(e.target.value))}
              placeholder={field.prompt}
            />
            <div className="form-text text-muted">{field.prompt}</div>
            {hasError && <div className="invalid-feedback d-block">{hasError}</div>}
          </div>
        );

      case 'checkbox':
        return (
          <div key={field.name} className="mb-3">
            <div className="form-check">
              <input
                type="checkbox"
                className="form-check-input"
                checked={value || false}
                onChange={(e) => handleFieldChange(field.name, e.target.checked)}
              />
              <label className="form-check-label">
                {field.label}
                {field.required && <span className="text-danger"> *</span>}
              </label>
            </div>
            <div className="form-text text-muted">{field.prompt}</div>
            {hasError && <div className="invalid-feedback d-block">{hasError}</div>}
          </div>
        );

      default: // text
        return (
          <div key={field.name} className="mb-3">
            <label className="form-label">
              {field.label}
              {field.required && <span className="text-danger"> *</span>}
            </label>
            <input
              type="text"
              className={`form-control ${hasError ? 'is-invalid' : ''}`}
              value={value ?? ''}
              onChange={(e) => handleFieldChange(field.name, e.target.value)}
              placeholder={field.prompt}
            />
            <div className="form-text text-muted">{field.prompt}</div>
            {hasError && <div className="invalid-feedback d-block">{hasError}</div>}
          </div>
        );
    }
  };

  // Separate required and optional fields
  const requiredFields = form.fields.filter(f => f.required);
  const optionalFields = form.fields.filter(f => !f.required);

  return (
    <div className="hitl-form-card bg-light border rounded-3 p-4 mb-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3">
        <i className="bi bi-ui-checks text-primary me-2" style={{ fontSize: '24px' }}></i>
        <div>
          <h5 className="mb-0">{form.title}</h5>
          <p className="text-muted small mb-0">{hitlData.context.message}</p>
        </div>
      </div>

      {/* Form Errors */}
      {errors._form && (
        <div className="alert alert-danger" role="alert">
          {errors._form}
        </div>
      )}

      {/* Required Fields Section */}
      {requiredFields.length > 0 && (
        <div className="mb-4">
          <h6 className="text-danger mb-3">
            <i className="bi bi-asterisk me-1"></i>
            Required Fields
          </h6>
          {requiredFields.map(renderField)}
        </div>
      )}

      {/* Optional Fields Section */}
      {optionalFields.length > 0 && (
        <div className="mb-4">
          <h6 className="text-muted mb-3">
            <i className="bi bi-sliders me-1"></i>
            Optional Configuration
          </h6>
          {optionalFields.map(renderField)}
        </div>
      )}

      {/* Action Buttons */}
      <div className="d-flex gap-2 mt-4">
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <span className="spinner-border spinner-border-sm me-2" />
              Submitting...
            </>
          ) : (
            <>
              <i className="bi bi-check-circle me-2"></i>
              Submit & Continue
            </>
          )}
        </button>

        <button
          className="btn btn-outline-secondary"
          onClick={() => onApprovalComplete?.(false, hitlData.run_id)}
          disabled={isSubmitting}
        >
          <i className="bi bi-x-circle me-2"></i>
          Cancel
        </button>
      </div>

      {/* Run Info */}
      <div className="mt-3 pt-3 border-top">
        <small className="text-muted">
          Run ID: {hitlData.run_id} |
          Missing: {hitlData.context.missing_required_fields.join(', ') || 'None'}
        </small>
      </div>
    </div>
  );
};
```

### Key Differences from Legacy Mode

| Aspect | Form-Based Mode | Legacy Validation Mode |
|--------|----------------|------------------------|
| **Checkpoint Type** | `form_requirements` | `information_review` |
| **Message Structure** | Contains `form` object with field definitions | Contains `validation_summary` with issues |
| **UX** | Fill form fields | Fix validation errors |
| **Field Classification** | AI classifies upfront (CONTENT/CONFIG) | No classification |
| **Default Values** | Config fields keep defaults | No distinction |
| **Array Handling** | Always start empty | May contain example values |
| **Submission** | Form values as edits | Fixes to validation issues |

### Migration from Legacy to Form-Based

Your existing HITL cards can support both modes:

```typescript
const HITLCard = ({ hitlData, onApprovalComplete }) => {
  // Detect checkpoint type
  const checkpointType = hitlData.context?.checkpoint_type || hitlData.context?.current_step;

  if (checkpointType === 'form_requirements') {
    return <FormBasedHITLCard hitlData={hitlData} onApprovalComplete={onApprovalComplete} />;
  }

  // Fall back to legacy validation card
  return <LegacyHITLValidationCard hitlData={hitlData} onApprovalComplete={onApprovalComplete} />;
};
```

## API Integration

### Basic HITL Endpoint Usage

```typescript
// Types for HITL API responses
interface HITLRunResponse {
  run_id: string;
  status: 'queued' | 'running' | 'awaiting_human' | 'completed' | 'failed' | 'cancelled';
  message?: string;
  websocket_url?: string;
  hitl_enabled: boolean;
  current_step?: string;
  actions_required?: string[];
  approval_token?: string;
  expires_at?: string;
  events_url?: string;
}

interface ValidationCheckpoint {
  type: string;
  title: string;
  description: string;
  required: boolean;
  passed: boolean;
  blocking: boolean;
  user_input_required: boolean;
  issues: ValidationIssue[];
}

interface ValidationIssue {
  field: string;
  issue: string;
  severity: 'error' | 'warning';
  suggested_fix: string;
  auto_fixable: boolean;
}

// Start HITL-enabled run
async function startHITLRun(runInput: RunInput): Promise<HITLRunResponse> {
  const response = await fetch('/api/hitl/run', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      run_input: runInput,
      user_id: currentUserId,
      session_id: currentSessionId
    })
  });
  
  if (!response.ok) {
    throw new Error(`HITL run failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Get run status with validation details
async function getRunStatus(runId: string): Promise<HITLStatusResponse> {
  const response = await fetch(`/api/hitl/run/${runId}/status`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Status check failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Get complete run state for session resume
async function getRunState(runId: string) {
  const response = await fetch(`/api/hitl/run/${runId}/state`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get run state: ${response.statusText}`);
  }
  
  return response.json();
}

// Get active runs for session resume
async function getActiveRuns(userId?: string, sessionId?: string, status?: string) {
  const params = new URLSearchParams();
  if (userId) params.append('user_id', userId);
  if (sessionId) params.append('session_id', sessionId);
  if (status) params.append('status', status);
  
  const response = await fetch(`/api/hitl/runs/active?${params}`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get active runs: ${response.statusText}`);
  }
  
  return response.json();
}

// Get session-specific active runs
async function getSessionActiveRuns(sessionId: string, userId?: string) {
  const params = new URLSearchParams();
  if (userId) params.append('user_id', userId);
  
  const response = await fetch(`/api/hitl/sessions/${sessionId}/active?${params}`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get session runs: ${response.statusText}`);
  }
  
  return response.json();
}

// Submit approval for pending checkpoint
async function submitApproval(
  runId: string, 
  approvalId: string, 
  action: 'approve' | 'edit' | 'reject',
  edits?: Record<string, any>,
  reason?: string,
  approvedBy?: string
): Promise<void> {
  const response = await fetch(`/api/hitl/run/${runId}/approve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      approval_id: approvalId,
      action,
      edits,
      reason,
      approved_by: approvedBy
    })
  });
  
  if (!response.ok) {
    throw new Error(`Approval failed: ${response.statusText}`);
  }
}

// Pause a running HITL workflow
async function pauseRun(runId: string): Promise<void> {
  const response = await fetch(`/api/hitl/run/${runId}/pause`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Pause failed: ${response.statusText}`);
  }
}

// Resume a paused HITL workflow
async function resumeRun(runId: string): Promise<void> {
  const response = await fetch(`/api/hitl/run/${runId}/resume`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (!response.ok) {
    throw new Error(`Resume failed: ${response.statusText}`);
  }
}

// Enhanced Run Input with HITL Support

interface EnhancedRunInput {
  prompt: string;
  agent_tool_config: {
    REPLICATETOOL: {
      data: {
        model_name: string;
        description: string;
        example_input: Record<string, any>;
        latest_version: string;
      }
    }
  };
  // HITL-specific options
  hitl_config?: {
    require_approval: boolean;
    policy: 'auto' | 'require_human' | 'auto_with_thresholds';
    allowed_steps: string[];
    review_thresholds?: Record<string, number>;
  };
}
```

Note: Do not include file/image URLs in `EnhancedRunInput`. If the selected model requires assets, the backend will attempt to auto-discover recent attachments from the user's chat history for the given `session_id`. If none are found, the HITL flow will pause with a validation checkpoint requesting the asset.

> 💡 **Tip for resumable sessions**: When rendering validation cards, inspect `hitlData.context.validation_summary.blocking_issues` and `pending_actions`. If the backend pauses due to a missing attachment, surface uploader controls so reviewers can attach the required file directly from the UI.

## WebSocket Integration

### HITL WebSocket Message Handling

The HITL system sends real-time notifications via WebSocket when human approval is required:

```typescript
// WebSocket message structure for HITL requests
interface HITLWebSocketMessage {
  sessionId: string;
  userId: string;
  status: "hitl_request";
  action: "approval_required";
  data: HITLApprovalRequest;
}

interface HITLApprovalRequest {
  run_id: string;
  user_id: string;
  session_id: string;
  created_at: string;
  context: {
    message: string;
    current_step: string;
    confidence_score: number;
    validation_summary: HITLValidationSummary;
    checkpoints: HITLValidationCheckpoint[];
    // NEW: Field schema metadata for editable fields
    schema?: {
      required_fields: string[];
      optional_fields: Record<string, FieldMetadata>;
      editable_fields: string[];
    };
  };
}

interface FieldMetadata {
  type: 'string' | 'number' | 'boolean' | 'float' | 'integer';
  default?: any;
  description?: string;
  range?: [number, number];
  min?: number;
  max?: number;
  options?: string[];
}

// Enhanced WebSocket hook for HITL detection
const useRagWebSocket = ({ sessionId, authToken, onHITLRequest }) => {
  // ... existing WebSocket setup

  useEffect(() => {
    if (lastMessage?.data) {
      const parsedData = JSON.parse(lastMessage.data);
      
      // Detect HITL approval requests
      if (parsedData.status === "hitl_request" || parsedData.action === "approval_required") {
        const hitlData = {
          run_id: parsedData.run_id || parsedData.data?.run_id,
          user_id: parsedData.userId || parsedData.user_id,
          session_id: parsedData.sessionId || parsedData.session_id,
          created_at: parsedData.created_at || new Date().toISOString(),
          context: parsedData.data?.context || parsedData.context || {
            message: parsedData.data?.message || "Human review required",
            validation_summary: parsedData.data?.validation_summary || { checkpoints: [] },
            // Extract schema metadata if present
            schema: parsedData.data?.context?.schema || parsedData.context?.schema
          }
        };

        // Add HITL message to chat thread as inline validation card
        addMessage({
          id: `hitl-${hitlData.run_id || Date.now()}`,
          content: hitlData.context.message,
          sender: "system",
          destination: sessionId || "",
          createdAt: new Date().toISOString(),
          agentSessionId: sessionId || "",
          props: {
            messageType: TMessageEnum.HITL_REQUEST,
            hitlData: hitlData,
          },
          hitlData: hitlData,
          type: TMessageEnum.HITL_REQUEST,
          isFresh: true,
        });

        if (onHITLRequest) {
          onHITLRequest(hitlData, sessionId || "");
        }
        return;
      }
    }
  }, [lastMessage]);
};
```

## Field Schema Metadata for Editable Fields

### Overview

The enhanced HITL system now provides field schema metadata to help frontends understand which fields can be edited, their types, defaults, and validation constraints. This enables better UX for the "edit" action in approval workflows.

### Schema Structure

When a HITL approval request includes editable fields, the `context.schema` object provides:

```typescript
interface HITLFieldSchema {
  required_fields: string[];           // Fields that must be present
  optional_fields: Record<string, FieldMetadata>;  // Fields with defaults
  editable_fields: string[];          // Fields humans can modify
}

interface FieldMetadata {
  type: 'string' | 'number' | 'boolean' | 'float' | 'integer';
  default?: any;                      // Default value if not provided
  description?: string;               // Human-readable description
  range?: [number, number];           // Min/max for numeric fields
  min?: number;                       // Minimum value
  max?: number;                       // Maximum value
  options?: string[];                 // Valid options for enum fields
}
```

### Example Schema Usage

```typescript
// Example HITL approval request with schema
const hitlRequest: HITLApprovalRequest = {
  run_id: "abc-123",
  user_id: "user-456",
  session_id: "session-789",
  created_at: "2024-01-15T10:30:00Z",
  context: {
    message: "Review AI model parameters before execution",
    current_step: "payload_validation",
    confidence_score: 0.85,
    validation_summary: { checkpoints: [] },
    schema: {
      required_fields: ["prompt", "model"],
      optional_fields: {
        "temperature": {
          type: "float",
          default: 0.7,
          description: "Controls randomness in generation",
          range: [0, 2]
        },
        "max_tokens": {
          type: "integer", 
          default: 1000,
          description: "Maximum tokens to generate",
          min: 1,
          max: 4000
        },
        "top_p": {
          type: "float",
          default: 1.0,
          description: "Nucleus sampling parameter",
          range: [0, 1]
        }
      },
      editable_fields: ["prompt", "temperature", "max_tokens", "top_p"]
    }
  }
};
```

### Frontend Implementation

```typescript
// Enhanced HITL validation card with field editing
const HITLValidationCardWithEditing: React.FC<HITLValidationCardProps> = ({
  hitlData,
  onApprovalComplete
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedFields, setEditedFields] = useState<Record<string, any>>({});
  const schema = hitlData.context.schema;

  const handleFieldEdit = (fieldName: string, value: any) => {
    setEditedFields(prev => ({
      ...prev,
      [fieldName]: value
    }));
  };

  const handleEditSubmit = async () => {
    try {
      await submitApproval(
        hitlData.run_id,
        hitlData.run_id, // approval_id
        'edit',
        editedFields, // Pass edited fields as edits parameter
        undefined,
        currentUserId
      );
      onApprovalComplete?.(true, hitlData.run_id);
    } catch (error) {
      console.error('Failed to submit edits:', error);
    }
  };

  const renderFieldEditor = (fieldName: string, metadata: FieldMetadata) => {
    const currentValue = editedFields[fieldName] ?? metadata.default;

    switch (metadata.type) {
      case 'string':
        return (
          <input
            type="text"
            className="form-control form-control-sm"
            value={currentValue || ''}
            onChange={(e) => handleFieldEdit(fieldName, e.target.value)}
            placeholder={metadata.description}
          />
        );
      
      case 'float':
      case 'number':
        return (
          <input
            type="number"
            className="form-control form-control-sm"
            value={currentValue || ''}
            onChange={(e) => handleFieldEdit(fieldName, parseFloat(e.target.value))}
            min={metadata.min || metadata.range?.[0]}
            max={metadata.max || metadata.range?.[1]}
            step={metadata.type === 'float' ? 0.1 : 1}
            placeholder={metadata.description}
          />
        );
      
      case 'boolean':
        return (
          <div className="form-check">
            <input
              type="checkbox"
              className="form-check-input"
              checked={currentValue || false}
              onChange={(e) => handleFieldEdit(fieldName, e.target.checked)}
            />
            <label className="form-check-label text-muted small">
              {metadata.description}
            </label>
          </div>
        );
      
      default:
        return (
          <input
            type="text"
            className="form-control form-control-sm"
            value={currentValue || ''}
            onChange={(e) => handleFieldEdit(fieldName, e.target.value)}
            placeholder={metadata.description}
          />
        );
    }
  };

  return (
    <div className="hitl-validation-card">
      {/* Standard validation display */}
      <div className="validation-summary">
        <p>{hitlData.context.message}</p>
      </div>

      {/* Field editing interface */}
      {schema && editMode && (
        <div className="field-editor mt-3">
          <h6 className="text-muted mb-2">Edit Parameters:</h6>
          {schema.editable_fields.map(fieldName => {
            const metadata = schema.optional_fields[fieldName];
            const isRequired = schema.required_fields.includes(fieldName);
            
            return (
              <div key={fieldName} className="mb-2">
                <label className="form-label small">
                  {fieldName}
                  {isRequired && <span className="text-danger">*</span>}
                  {metadata?.default !== undefined && (
                    <span className="text-muted"> (default: {String(metadata.default)})</span>
                  )}
                </label>
                {metadata ? renderFieldEditor(fieldName, metadata) : (
                  <input
                    type="text"
                    className="form-control form-control-sm"
                    onChange={(e) => handleFieldEdit(fieldName, e.target.value)}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Action buttons */}
      <div className="d-flex gap-2 mt-3">
        <button
          className="btn btn-success btn-sm"
          onClick={() => handleApprove()}
        >
          Approve
        </button>
        
        {schema && schema.editable_fields.length > 0 && (
          <>
            {!editMode ? (
              <button
                className="btn btn-warning btn-sm"
                onClick={() => setEditMode(true)}
              >
                Edit Parameters
              </button>
            ) : (
              <>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={handleEditSubmit}
                >
                  Submit Edits
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setEditMode(false)}
                >
                  Cancel
                </button>
              </>
            )}
          </>
        )}
        
        <button
          className="btn btn-danger btn-sm"
          onClick={() => handleReject()}
        >
          Reject
        </button>
      </div>
    </div>
  );
};
```

### Backend Integration

The backend automatically includes schema metadata when calling `request_human_approval()`:

```python
# Example usage in HITL orchestrator
schema = websocket_bridge.create_field_schema(
    required_fields=["prompt", "model"],
    optional_fields={
        "temperature": {"default": 0.7, "type": "float", "range": [0, 2]},
        "max_tokens": {"default": 1000, "type": "int", "min": 1, "max": 4000}
    },
    editable_fields=["prompt", "temperature", "max_tokens"]
)

approval_response = await websocket_bridge.request_human_approval(
    run_id=run_id,
    checkpoint_type="payload_review",
    context={"message": "Review parameters", "payload": current_payload},
    user_id=user_id,
    session_id=session_id,
    schema=schema  # Include schema metadata
)
```

## Session Resume & State Management

### Overview

The HITL system provides comprehensive session resume capabilities, allowing users to leave and return to continue their workflows seamlessly. This is critical for long-running AI tasks that require human oversight.

### Database Schema for Session Resume

The HITL system persists complete state across three main tables:

```sql
-- Main HITL runs with complete state
CREATE TABLE hitl_runs (
    run_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    current_step VARCHAR(50) NOT NULL,
    provider_name VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    original_input JSONB NOT NULL,
    hitl_config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    
    -- Step artifacts for resume
    capabilities JSONB,
    suggested_payload JSONB,
    validation_issues JSONB,
    raw_response JSONB,
    processed_response TEXT,
    final_result TEXT,
    
    -- Human interaction state
    pending_actions JSONB,
    approval_token VARCHAR(255),
    
    -- Metrics
    total_execution_time_ms INTEGER DEFAULT 0,
    human_review_time_ms INTEGER DEFAULT 0
);

-- Complete audit trail of all steps
CREATE TABLE hitl_step_events (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    actor VARCHAR(255) NOT NULL, -- 'system' or user_id
    message TEXT,
    metadata JSONB
);

-- Individual approval tracking
CREATE TABLE hitl_approvals (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    approval_id VARCHAR(255) UNIQUE NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    response JSONB,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP,
    expires_at TIMESTAMP
);

### API Endpoints for Session Resume

```typescript
// Get all active HITL runs for a user/session
interface GetActiveRunsRequest {
  user_id?: string;
  session_id?: string;
  status?: 'awaiting_human' | 'running' | 'all';
}

interface ActiveRunSummary {
  run_id: string;
  status: string;
  current_step: string;
  created_at: string;
  expires_at?: string;
  pending_actions: string[];
  context_summary: string;
}

// GET /api/hitl/runs/active
async function getActiveHITLRuns(params: GetActiveRunsRequest): Promise<ActiveRunSummary[]> {
  const query = new URLSearchParams(params as any).toString();
  const response = await fetch(`/api/hitl/runs/active?${query}`);
  return response.json();
}

// Get complete run state for resume
interface RunStateResponse {
  run_id: string;
  status: string;
  current_step: string;
  original_input: any;
  hitl_config: any;
  
  // Current state artifacts
  capabilities?: any;
  suggested_payload?: any;
  validation_issues?: ValidationIssue[];
  
  // Pending human actions
  pending_actions: string[];
  approval_token?: string;
  expires_at?: string;
  
  // Step history for context
  step_history: StepEvent[];
  
  // Validation summary for current step
  validation_summary?: HITLValidationSummary;
}

// GET /api/hitl/runs/{run_id}/state
async function getRunState(runId: string): Promise<RunStateResponse> {
  const response = await fetch(`/api/hitl/runs/${runId}/state`);
  return response.json();
}

// Resume execution from current step
interface ResumeRunRequest {
  run_id: string;
  action: 'continue' | 'approve' | 'edit' | 'cancel';
  modifications?: Record<string, any>;
  approval_token?: string;
}

// POST /api/hitl/runs/{run_id}/resume
async function resumeRun(request: ResumeRunRequest): Promise<HITLRunResponse> {
  const response = await fetch(`/api/hitl/runs/${request.run_id}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  return response.json();
}
```

### Frontend Session Resume Implementation

```typescript
// Session store enhancement for HITL resume
interface SessionState {
  activeHITLRuns: ActiveRunSummary[];
  pendingApprovals: Map<string, HITLApprovalRequest>;
  resumeInProgress: Set<string>;
}

// Enhanced session store
const useSessionStore = create<SessionState>((set, get) => ({
  activeHITLRuns: [],
  pendingApprovals: new Map(),
  resumeInProgress: new Set(),

  // Load active HITL runs when session starts
  loadActiveHITLRuns: async (sessionId: string, userId: string) => {
    try {
      const runs = await getActiveHITLRuns({ session_id: sessionId, user_id: userId });
      set({ activeHITLRuns: runs });
      
      // Load detailed state for awaiting_human runs
      const awaitingRuns = runs.filter(r => r.status === 'awaiting_human');
      const pendingApprovals = new Map();
      
      for (const run of awaitingRuns) {
        const state = await getRunState(run.run_id);
        if (state.pending_actions.length > 0) {
          pendingApprovals.set(run.run_id, {
            run_id: run.run_id,
            user_id: userId,
            session_id: sessionId,
            created_at: run.created_at,
            context: {
              message: state.context_summary || 'Human review required',
              current_step: state.current_step,
              confidence_score: 0.8,
              validation_summary: state.validation_summary || { checkpoints: [] }
            }
          });
        }
      }
      
      set({ pendingApprovals });
    } catch (error) {
      console.error('Failed to load active HITL runs:', error);
    }
  },

  // Resume a specific HITL run
  resumeHITLRun: async (runId: string, action: string, modifications?: any) => {
    const { resumeInProgress } = get();
    if (resumeInProgress.has(runId)) return;

    set({ resumeInProgress: new Set([...resumeInProgress, runId]) });
    
    try {
      const result = await resumeRun({
        run_id: runId,
        action: action as any,
        modifications
      });
      
      // Update local state
      const { activeHITLRuns, pendingApprovals } = get();
      const updatedRuns = activeHITLRuns.map(run => 
        run.run_id === runId ? { ...run, status: result.status } : run
      );
      
      const updatedApprovals = new Map(pendingApprovals);
      if (result.status !== 'awaiting_human') {
        updatedApprovals.delete(runId);
      }
      
      set({ 
        activeHITLRuns: updatedRuns,
        pendingApprovals: updatedApprovals,
        resumeInProgress: new Set([...resumeInProgress].filter(id => id !== runId))
      });
      
      return result;
    } catch (error) {
      console.error('Failed to resume HITL run:', error);
      set({ 
        resumeInProgress: new Set([...resumeInProgress].filter(id => id !== runId))
      });
      throw error;
    }
  }
}));

// Session initialization with HITL resume
const ChatSession = ({ sessionId, userId }) => {
  const { loadActiveHITLRuns, pendingApprovals } = useSessionStore();
  const addMessage = useThreadStore(state => state.addMessage);

  useEffect(() => {
    // Load active HITL runs when session starts
    loadActiveHITLRuns(sessionId, userId);
  }, [sessionId, userId]);

  useEffect(() => {
    // Add pending approvals as inline messages in chat
    pendingApprovals.forEach((hitlData, runId) => {
      addMessage({
        id: `hitl-resume-${runId}`,
        content: hitlData.context.message,
        sender: "system",
        destination: sessionId,
        createdAt: hitlData.created_at,
        agentSessionId: sessionId,
        props: {
          messageType: TMessageEnum.HITL_REQUEST,
          hitlData: hitlData,
        },
        hitlData: hitlData,
        type: TMessageEnum.HITL_REQUEST,
        isFresh: false, // Not fresh since it's resumed
      });
    });
  }, [pendingApprovals]);

  return <ChatThread sessionId={sessionId} />;
};
```

### Resume UX Patterns

```typescript
// Resume indicator component
const HITLResumeIndicator = ({ runs }: { runs: ActiveRunSummary[] }) => {
  if (runs.length === 0) return null;

  return (
    <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
      <div className="flex">
        <div className="flex-shrink-0">
          <ClockIcon className="h-5 w-5 text-blue-400" />
        </div>
        <div className="ml-3">
          <p className="text-sm text-blue-700">
            You have {runs.length} pending HITL approval{runs.length > 1 ? 's' : ''} from previous sessions.
          </p>
          <div className="mt-2 space-y-1">
            {runs.map(run => (
              <div key={run.run_id} className="text-xs text-blue-600">
                • {run.context_summary} (Step: {run.current_step})
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Enhanced HITL validation card with resume context
const HITLValidationCard = ({ hitlData, isResumed = false, onApprovalComplete }) => {
  return (
    <div className="hitl-validation-inline mb-3">
      {/* Resume indicator */}
      {isResumed && (
        <div className="bg-amber-50 border border-amber-200 rounded-md p-2 mb-2">
          <div className="flex items-center">
            <RefreshIcon className="h-4 w-4 text-amber-600 mr-2" />
            <span className="text-sm text-amber-800">Resumed from previous session</span>
          </div>
        </div>
      )}
      
      {/* System message style header */}
      <div className="d-flex align-items-center mb-2">
        <div className="bg-warning rounded-circle p-1 me-2" style={{ width: '24px', height: '24px' }}>
          <i className="bi bi-exclamation-triangle-fill text-white" style={{ fontSize: '12px' }}></i>
        </div>
        <span className="text-muted small">System</span>
        <span className="text-muted small ms-auto">
          {new Date(hitlData.created_at).toLocaleTimeString()}
          {isResumed && <span className="text-amber-600 ml-2">(Resumed)</span>}
        </span>
      </div>

      {/* Rest of validation card... */}
    </div>
  );
};
```

### Key Resume Scenarios

1. **Browser Refresh**: Complete state restoration from database
2. **Session Switch**: Load pending approvals across different chat sessions  
3. **Timeout Recovery**: Handle expired approvals with re-authentication
4. **Multi-device Access**: Same HITL state accessible from different devices
5. **Collaborative Workflows**: Multiple users can view and act on shared HITL requests

### Implementation Checklist

- [ ] Database tables for state persistence (`hitl_runs`, `hitl_step_events`, `hitl_approvals`)
- [ ] API endpoints for active runs and state retrieval
- [ ] Session store enhancement for HITL resume
- [ ] WebSocket reconnection for resumed sessions
- [ ] UI indicators for resumed vs new HITL requests
- [ ] Timeout and expiration handling
- [ ] Multi-user collaboration support
- [ ] Error recovery for failed resumes

## Local Testing

Run the backend HITL regression suites before releasing any frontend changes that touch validation flows or session resume logic:

```bash
poetry run pytest tests/test_hitl_session_resumability.py -v
poetry run pytest tests/test_hitl_database_integrity.py -v
poetry run pytest tests/test_hitl_edge_cases.py -v

# Aggregated report with resumability verification
poetry run python tests/run_hitl_tests.py
```

The aggregated runner outputs a human-readable summary of each suite and confirms the database state still reflects resumability requirements—useful when validating new UI affordances for paused runs.

## React Components

### HITL Validation Card Component

```typescript
import React, { useState } from "react";
import { HITLApprovalRequest } from "@/lib/types";
import { approveHITLRequest, rejectHITLRequest } from "../services/session-req";

interface HITLValidationCardProps {
  hitlData: HITLApprovalRequest;
  onApprovalComplete?: (approved: boolean, runId: string) => void;
  isResumed?: boolean;
}

export const HITLValidationCard: React.FC<HITLValidationCardProps> = ({
  hitlData,
  onApprovalComplete,
  isResumed = false
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [rejectionReason, setRejectionReason] = useState("");
  const [expandedCheckpoints, setExpandedCheckpoints] = useState<Set<number>>(new Set());

  const handleApprove = async () => {
    setIsProcessing(true);
    try {
      await approveHITLRequest(hitlData.run_id, hitlData.user_id);
      onApprovalComplete?.(true, hitlData.run_id);
    } catch (error) {
      console.error("Failed to approve HITL request:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReject = async () => {
    if (!rejectionReason.trim()) {
      alert("Please provide a reason for rejection");
      return;
    }
    
    setIsProcessing(true);
    try {
      await rejectHITLRequest(hitlData.run_id, hitlData.user_id, rejectionReason);
      onApprovalComplete?.(false, hitlData.run_id);
    } catch (error) {
      console.error("Failed to reject HITL request:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleCheckpoint = (index: number) => {
    const newExpanded = new Set(expandedCheckpoints);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedCheckpoints(newExpanded);
  };

  return (
    <div className="hitl-validation-inline mb-3">
      {/* Resume indicator */}
      {isResumed && (
        <div className="bg-amber-50 border border-amber-200 rounded-md p-2 mb-2">
          <div className="flex items-center">
            <RefreshIcon className="h-4 w-4 text-amber-600 mr-2" />
            <span className="text-sm text-amber-800">Resumed from previous session</span>
          </div>
        </div>
      )}
      
      {/* System message style header */}
      <div className="d-flex align-items-center mb-2">
        <div className="bg-warning rounded-circle p-1 me-2" style={{ width: '24px', height: '24px' }}>
          <i className="bi bi-exclamation-triangle-fill text-white" style={{ fontSize: '12px' }}></i>
        </div>
        <span className="text-muted small">System</span>
        <span className="text-muted small ms-auto">
          {new Date(hitlData.created_at).toLocaleTimeString()}
          {isResumed && <span className="text-amber-600 ml-2">(Resumed)</span>}
        </span>
      </div>

      {/* Message bubble style card */}
      <div className="bg-light border rounded-3 p-3" style={{ maxWidth: '85%' }}>
        <div className="d-flex align-items-center mb-2">
          <i className="bi bi-exclamation-triangle-fill text-warning me-2"></i>
          <strong className="text-warning">Human Review Required</strong>
        </div>
        
        <p className="mb-3 text-muted">{hitlData.context.message}</p>

        {/* Validation Summary Badges */}
        {hitlData.context.validation_summary && (
          <div className="mb-3">
            <div className="row g-2">
              <div className="col-6 col-md-3">
                <span className="badge bg-success">Passed: {hitlData.context.validation_summary.total_checkpoints || 0}</span>
              </div>
              <div className="col-6 col-md-3">
                <span className="badge bg-danger">Failed: {hitlData.context.validation_summary.total_issues || 0}</span>
              </div>
              <div className="col-6 col-md-3">
                <span className="badge bg-secondary">Issues: {hitlData.context.validation_summary.total_issues || 0}</span>
              </div>
              <div className="col-6 col-md-3">
                <span className="badge bg-warning">Blocking: {hitlData.context.validation_summary.blocking_issues || 0}</span>
              </div>
            </div>
          </div>
        )}

        {/* Expandable Validation Checkpoints */}
        {hitlData.context.validation_summary?.checkpoints && hitlData.context.validation_summary.checkpoints.length > 0 && (
          <div className="mb-3">
            <h6 className="small text-muted mb-2">Validation Checkpoints:</h6>
            {hitlData.context.validation_summary.checkpoints.map((checkpoint, index) => (
              <div key={index} className="border rounded p-2 mb-2 bg-white">
                <div 
                  className="d-flex justify-content-between align-items-center cursor-pointer"
                  onClick={() => toggleCheckpoint(index)}
                >
                  <span className="small fw-medium">{checkpoint.checkpoint_type || `Checkpoint ${index + 1}`}</span>
                  <div className="d-flex align-items-center">
                    <span className={`badge badge-sm me-2 ${checkpoint.passed ? 'bg-success' : 'bg-danger'}`}>
                      {checkpoint.passed ? 'passed' : 'failed'}
                    </span>
                    <i className={`bi bi-chevron-${expandedCheckpoints.has(index) ? 'up' : 'down'} small`}></i>
                  </div>
                </div>
                
                {expandedCheckpoints.has(index) && (
                  <div className="mt-2 pt-2 border-top">
                    <p className="mb-2 text-muted small">{checkpoint.description || 'No description available'}</p>
                    {checkpoint.issues && checkpoint.issues.length > 0 && (
                      <div>
                        <strong className="small">Issues:</strong>
                        <ul className="mt-1 mb-0 small">
                          {checkpoint.issues.map((issue, issueIndex) => (
                            <li key={issueIndex} className="text-danger">
                              <strong>{issue.severity || 'Issue'}:</strong> {issue.message || 'No details available'}
                              {issue.suggestion && (
                                <div className="text-primary mt-1">
                                  <strong>Suggestion:</strong> {issue.suggestion}
                                </div>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Action buttons */}
        <div className="d-flex gap-2 align-items-center mt-3">
          <button
            className="btn btn-success btn-sm"
            onClick={handleApprove}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <span className="spinner-border spinner-border-sm me-1" role="status"></span>
                Processing...
              </>
            ) : (
              <>
                <i className="bi bi-check-circle me-1"></i>
                Approve
              </>
            )}
          </button>
          
          <div className="flex-grow-1">
            <input
              type="text"
              className="form-control form-control-sm"
              placeholder="Reason for rejection (required)"
              value={rejectionReason}
              onChange={(e) => setRejectionReason(e.target.value)}
              disabled={isProcessing}
            />
          </div>
          
          <button
            className="btn btn-danger btn-sm"
            onClick={handleReject}
            disabled={isProcessing || !rejectionReason.trim()}
          >
            {isProcessing ? (
              <>
                <span className="spinner-border spinner-border-sm me-1" role="status"></span>
                Processing...
              </>
            ) : (
              <>
                <i className="bi bi-x-circle me-1"></i>
                Reject
              </>
            )}
          </button>
        </div>

        <div className="mt-2">
          <small className="text-muted">Run ID: {hitlData.run_id}</small>
        </div>
      </div>
    </div>
  );
};
```

### Message Component Integration

```typescript
// In your Message.tsx component
import { HITLValidationCard } from './HITLValidationCard';
import { TMessageEnum } from '@/lib/types';

const Message = ({ message, ...props }) => {
  // Handle HITL validation cards as inline messages
  if (message.type === TMessageEnum.HITL_REQUEST && message.hitlData) {
    return (
      <HITLValidationCard 
        hitlData={message.hitlData}
        isResumed={!message.isFresh} // Resumed if not fresh
        onApprovalComplete={(approved, runId) => {
          console.log(`HITL ${approved ? 'approved' : 'rejected'} for run ${runId}`);
          // Handle approval completion - remove from UI, update state, etc.
        }}
      />
    );
  }

  // Regular message rendering...
  return <div className="message">...</div>;
};
```

### WebSocket Hook for HITL Communication

```typescript
import { useEffect, useState, useCallback } from 'react';
import useWebSocket from 'react-use-websocket';
import { useThreadStore } from '@/common/context/thread.store';
import { TMessageEnum, HITLApprovalRequest } from '@/lib/types';

interface HITLWebSocketMessage {
  sessionId: string;
  userId: string;
  status: "hitl_request";
  action: "approval_required";
  data: HITLApprovalRequest;
}

interface UseRagWebSocketProps {
  sessionId: string | null;
  authToken: string | null;
  onResponseReceived?: (response: string, sessionId: string) => void;
  onError?: (error: string, sessionId: string) => void;
  onHITLRequest?: (request: HITLApprovalRequest, sessionId: string) => void;
}

export const useRagWebSocket = ({
  sessionId,
  authToken,
  onResponseReceived,
  onError,
  onHITLRequest,
}: UseRagWebSocketProps) => {
  const addMessage = useThreadStore((state) => state.addMessage);
  const setTyping = useThreadStore((state) => state.setTyping);

  const { lastMessage, sendMessage, connectionStatus } = useWebSocket(
    sessionId ? `${WS_BASE_URL}/ws` : null,
    {
      shouldReconnect: () => true,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
    }
  );

  useEffect(() => {
    if (lastMessage?.data) {
      try {
        const parsedData = JSON.parse(lastMessage.data);
        
        // Detect HITL approval requests
        if (parsedData.status === "hitl_request" || parsedData.action === "approval_required") {
          console.log("🔍 HITL request detected:", parsedData);
          
          const hitlData: HITLApprovalRequest = {
            run_id: parsedData.run_id || parsedData.data?.run_id || `hitl-${Date.now()}`,
            user_id: parsedData.userId || parsedData.user_id || "",
            session_id: parsedData.sessionId || parsedData.session_id || sessionId || "",
            created_at: parsedData.created_at || new Date().toISOString(),
            context: parsedData.data?.context || parsedData.context || {
              message: parsedData.data?.message || "Model capabilities and parameters require human review",
              current_step: parsedData.data?.current_step || "information_review",
              confidence_score: parsedData.data?.confidence_score || 0.8,
              validation_summary: parsedData.data?.validation_summary || {
                total_checkpoints: 0,
                passed_checkpoints: 0,
                failed_checkpoints: 0,
                blocking_checkpoints: 0,
                total_issues: 0,
                blocking_issues: 0,
                checkpoints: []
              }
            }
          };

          console.log("🔍 Final HITL data structure:", hitlData);

          // Add HITL validation card as inline message in chat thread
          addMessage({
            id: `hitl-${hitlData.run_id || Date.now()}`,
            content: hitlData.context.message || "Model capabilities and parameters require human review",
            sender: "system",
            destination: sessionId || "",
            createdAt: new Date().toISOString(),
            agentSessionId: sessionId || "",
            props: {
              messageType: TMessageEnum.HITL_REQUEST,
              hitlData: hitlData,
            },
            hitlData: hitlData,
            type: TMessageEnum.HITL_REQUEST,
            isFresh: true,
          });

          if (onHITLRequest) {
            onHITLRequest(hitlData, sessionId || "");
          }
          
          setTyping(false);
          return;
        }

        // Handle other message types...
        
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    }
  }, [lastMessage, sessionId, addMessage, onHITLRequest, setTyping]);

  return {
    connectionStatus,
    sendMessage,
    isConnected: connectionStatus === 'Open'
  };
};
}

export function useHITLWebSocket(sessionId: string, onMessage?: (message: HITLWebSocketMessage) => void) {
  const [lastHITLMessage, setLastHITLMessage] = useState<HITLWebSocketMessage | null>(null);
  
  const { lastMessage, sendMessage } = useWebSocket(
    `wss://ws.tohju.com`,
    {
      onOpen: () => {
        // Subscribe to HITL events for this session
        sendMessage(JSON.stringify({
          type: 'subscribe',
          channel: `hitl_${sessionId}`
        }));
      },
      onMessage: (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type?.startsWith('hitl_')) {
            setLastHITLMessage(message);
            onMessage?.(message);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      },
      shouldReconnect: () => true,
    }
  );

  const sendHITLMessage = useCallback((message: any) => {
    sendMessage(JSON.stringify({
      type: 'hitl_message',
      session_id: sessionId,
      ...message
    }));
  }, [sendMessage, sessionId]);

  return {
    lastHITLMessage,
    sendHITLMessage
  };
}
```

## React Components

### Validation Checkpoint Display Component

```tsx
import React from 'react';
import { AlertTriangle, CheckCircle, XCircle, Upload, Edit } from 'lucide-react';

interface ValidationCheckpointProps {
  checkpoint: ValidationCheckpoint;
  onFixIssue?: (field: string, value: any) => void;
}

export function ValidationCheckpointCard({ checkpoint, onFixIssue }: ValidationCheckpointProps) {
  const getStatusIcon = () => {
    if (checkpoint.passed) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (checkpoint.blocking) return <XCircle className="w-5 h-5 text-red-500" />;
    return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
  };

  const getStatusColor = () => {
    if (checkpoint.passed) return 'border-green-200 bg-green-50';
    if (checkpoint.blocking) return 'border-red-200 bg-red-50';
    return 'border-yellow-200 bg-yellow-50';
  };

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor()}`}>
      <div className="flex items-center gap-3 mb-3">
        {getStatusIcon()}
        <div>
          <h3 className="font-semibold text-gray-900">{checkpoint.title}</h3>
          <p className="text-sm text-gray-600">{checkpoint.description}</p>
        </div>
      </div>

      {checkpoint.issues.length > 0 && (
        <div className="space-y-2">
          {checkpoint.issues.map((issue, index) => (
            <ValidationIssueItem
              key={index}
              issue={issue}
              onFix={onFixIssue}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ValidationIssueItem({ 
  issue, 
  onFix 
}: { 
  issue: ValidationIssue; 
  onFix?: (field: string, value: any) => void;
}) {
  const getSeverityColor = () => {
    return issue.severity === 'error' ? 'text-red-600' : 'text-yellow-600';
  };

  const handleQuickFix = () => {
    if (issue.field === 'prompt' || issue.field.includes('text')) {
      const newValue = prompt('Enter the required text:');
      if (newValue) onFix?.(issue.field, newValue);
    } else if (issue.field.includes('image') || issue.field.includes('file')) {
      // Trigger file upload
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = issue.field.includes('image') ? 'image/*' : '*/*';
      input.onchange = (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) onFix?.(issue.field, file);
      };
      input.click();
    }
  };

  return (
    <div className="flex items-start gap-3 p-3 bg-white rounded border">
      <div className="flex-1">
        <p className={`font-medium ${getSeverityColor()}`}>
          {issue.issue}
        </p>
        <p className="text-sm text-gray-600 mt-1">
          {issue.suggested_fix}
        </p>
      </div>
      
      {!issue.auto_fixable && onFix && (
        <button
          onClick={handleQuickFix}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
        >
          {issue.field.includes('file') || issue.field.includes('image') ? (
            <Upload className="w-4 h-4" />
          ) : (
            <Edit className="w-4 h-4" />
          )}
          Fix
        </button>
      )}
    </div>
  );
}
```

### HITL Approval Interface Component

```tsx
import React, { useState } from 'react';
import { Play, Pause, Edit, X } from 'lucide-react';

interface HITLApprovalProps {
  runId: string;
  approvalToken: string;
  message: string;
  actionsRequired: string[];
  validationSummary?: {
    checkpoints: ValidationCheckpoint[];
    blocking_issues: number;
    ready_for_execution: boolean;
  };
  onApproval: (approved: boolean, modifications?: Record<string, any>) => void;
}

export function HITLApprovalInterface({
  runId,
  approvalToken,
  message,
  actionsRequired,
  validationSummary,
  onApproval
}: HITLApprovalProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [modifications, setModifications] = useState<Record<string, any>>({});

  const handleApprove = () => {
    onApproval(true, Object.keys(modifications).length > 0 ? modifications : undefined);
  };

  const handleReject = () => {
    onApproval(false);
  };

  const handleFixIssue = (field: string, value: any) => {
    setModifications(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <Pause className="w-5 h-5 text-yellow-500" />
          <div>
            <h3 className="font-semibold text-gray-900">Human Approval Required</h3>
            <p className="text-sm text-gray-600">{message}</p>
          </div>
        </div>
      </div>

      {/* Validation Summary */}
      {validationSummary && (
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-gray-900">Validation Results</h4>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {validationSummary.checkpoints.filter(cp => cp.passed).length}
              </div>
              <div className="text-gray-600">Passed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {validationSummary.blocking_issues}
              </div>
              <div className="text-gray-600">Blocking Issues</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${validationSummary.ready_for_execution ? 'text-green-600' : 'text-red-600'}`}>
                {validationSummary.ready_for_execution ? 'Ready' : 'Not Ready'}
              </div>
              <div className="text-gray-600">For Execution</div>
            </div>
          </div>

          {showDetails && (
            <div className="mt-4 space-y-3">
              {validationSummary.checkpoints.map((checkpoint, index) => (
                <ValidationCheckpointCard
                  key={index}
                  checkpoint={checkpoint}
                  onFixIssue={handleFixIssue}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Required Actions */}
      <div className="p-4 border-b border-gray-200">
        <h4 className="font-medium text-gray-900 mb-2">Required Actions</h4>
        <ul className="space-y-1">
          {actionsRequired.map((action, index) => (
            <li key={index} className="flex items-center gap-2 text-sm text-gray-600">
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full" />
              {action.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </li>
          ))}
        </ul>
      </div>

      {/* Modifications Preview */}
      {Object.keys(modifications).length > 0 && (
        <div className="p-4 border-b border-gray-200 bg-blue-50">
          <h4 className="font-medium text-gray-900 mb-2">Pending Modifications</h4>
          <div className="space-y-1">
            {Object.entries(modifications).map(([field, value]) => (
              <div key={field} className="text-sm">
                <span className="font-medium">{field}:</span>{' '}
                <span className="text-gray-600">
                  {typeof value === 'string' ? value : JSON.stringify(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="p-4 flex gap-3">
        <button
          onClick={handleApprove}
          disabled={validationSummary && !validationSummary.ready_for_execution}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          <Play className="w-4 h-4" />
          Approve & Continue
        </button>
        
        <button
          onClick={handleReject}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          <X className="w-4 h-4" />
          Cancel
        </button>
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50"
        >
          <Edit className="w-4 h-4" />
          Edit Parameters
        </button>
      </div>
    </div>
  );
}
```

## State Management

### HITL State Management Hook

```tsx
import { useState, useEffect, useCallback } from 'react';

interface HITLState {
  runId: string | null;
  status: string;
  currentStep: string | null;
  validationSummary: any | null;
  approvalPending: boolean;
  approvalToken: string | null;
  error: string | null;
}

export function useHITLState(sessionId: string) {
  const [state, setState] = useState<HITLState>({
    runId: null,
    status: 'idle',
    currentStep: null,
    validationSummary: null,
    approvalPending: false,
    approvalToken: null,
    error: null
  });

  const { lastHITLMessage, sendHITLMessage } = useHITLWebSocket(
    sessionId,
    useCallback((message) => {
      switch (message.type) {
        case 'hitl_approval_request':
          setState(prev => ({
            ...prev,
            status: 'awaiting_human',
            approvalPending: true,
            approvalToken: message.data.approval_token,
            validationSummary: message.data.validation_summary,
            currentStep: message.data.current_step
          }));
          break;

        case 'hitl_status_update':
          setState(prev => ({
            ...prev,
            status: message.data.status,
            currentStep: message.data.current_step,
            approvalPending: false
          }));
          break;

        case 'hitl_error':
          setState(prev => ({
            ...prev,
            status: 'error',
            error: message.data.error,
            approvalPending: false
          }));
          break;

        case 'hitl_completion':
          setState(prev => ({
            ...prev,
            status: 'completed',
            approvalPending: false,
            currentStep: 'completed'
          }));
          break;
      }
    }, [])
  );

  const startHITLRun = useCallback(async (runInput: EnhancedRunInput) => {
    try {
      setState(prev => ({ ...prev, error: null }));
      const response = await startHITLRun(runInput);
      
      setState(prev => ({
        ...prev,
        runId: response.run_id,
        status: response.status
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'error'
      }));
      throw error;
    }
  }, []);

  const submitApproval = useCallback(async (
    approved: boolean,
    modifications?: Record<string, any>
  ) => {
    if (!state.runId || !state.approvalToken) return;

    try {
      await submitApproval(state.runId, state.approvalToken, approved, modifications);
      
      setState(prev => ({
        ...prev,
        approvalPending: false,
        approvalToken: null
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Approval failed'
      }));
    }
  }, [state.runId, state.approvalToken]);

  const resetState = useCallback(() => {
    setState({
      runId: null,
      status: 'idle',
      currentStep: null,
      validationSummary: null,
      approvalPending: false,
      approvalToken: null,
      error: null
    });
  }, []);

  return {
    state,
    startHITLRun,
    submitApproval,
    resetState
  };
}
```

## Error Handling

### Error Boundary for HITL Components

```tsx
import React, { Component, ReactNode } from 'react';

interface HITLErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class HITLErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  HITLErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): HITLErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('HITL Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 border border-red-200 rounded-lg bg-red-50">
          <h3 className="font-semibold text-red-800 mb-2">HITL System Error</h3>
          <p className="text-red-600 text-sm">
            {this.state.error?.message || 'An unexpected error occurred in the HITL system.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-3 px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## User Experience Patterns

### Progressive Enhancement Pattern

```tsx
import React, { useState } from 'react';

interface AIWorkflowProps {
  onSubmit: (input: EnhancedRunInput) => void;
  hitlEnabled?: boolean;
}

export function AIWorkflowForm({ onSubmit, hitlEnabled = false }: AIWorkflowProps) {
  const [input, setInput] = useState<EnhancedRunInput>({
    prompt: '',
    agent_tool_config: {
      REPLICATETOOL: {
        data: {
          model_name: '',
          description: '',
          example_input: {},
          latest_version: ''
        }
      }
    }
  });

  const [validationMode, setValidationMode] = useState<'basic' | 'enhanced'>('basic');

  useEffect(() => {
    if (hitlEnabled) {
      setValidationMode('enhanced');
    }
  }, [hitlEnabled]);

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={validationMode === 'enhanced'}
            onChange={(e) => setValidationMode(e.target.checked ? 'enhanced' : 'basic')}
          />
          <span className="text-sm font-medium">Enhanced validation (HITL)</span>
        </label>
      </div>

      {/* Enhanced mode benefits */}
      {validationMode === 'enhanced' && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-1">Enhanced Mode Benefits</h4>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Pre-execution parameter validation</li>
            <li>• Human approval checkpoints</li>
            <li>• Detailed error guidance</li>
            <li>• Auto-fix suggestions</li>
          </ul>
        </div>
      )}

      {/* Form fields */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Prompt *
          </label>
          <textarea
            value={input.prompt}
            onChange={(e) => setInput(prev => ({ ...prev, prompt: e.target.value }))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={4}
            placeholder="Describe what you want the AI to do..."
          />
        </div>

        {/* File upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Upload File (Optional)
          </label>
          <input
            type="file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                // Handle file upload
                setInput(prev => ({ ...prev, document_url: URL.createObjectURL(file) }));
              }
            }}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>
      </div>

      {/* Submit button */}
      <button
        onClick={() => onSubmit({
          ...input,
          hitl_config: validationMode === 'enhanced' ? {
            require_approval: true,
            policy: 'auto_with_thresholds',
            allowed_steps: ['information_review', 'payload_review', 'response_review']
          } : undefined
        })}
        disabled={!input.prompt.trim()}
        className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
      >
        {validationMode === 'enhanced' ? 'Start Enhanced AI Workflow' : 'Start AI Workflow'}
      </button>
    </div>
  );
}
```

## Complete Examples

### Full HITL Integration Example

```tsx
import React from 'react';
import { HITLErrorBoundary } from './HITLErrorBoundary';
import { useHITLState } from './useHITLState';
import { HITLApprovalInterface } from './HITLApprovalInterface';
import { AIWorkflowForm } from './AIWorkflowForm';

export function HITLWorkflowContainer({ sessionId }: { sessionId: string }) {
  const { state, startHITLRun, submitApproval, resetState } = useHITLState(sessionId);

  const handleSubmit = async (input: EnhancedRunInput) => {
    try {
      await startHITLRun(input);
    } catch (error) {
      console.error('Failed to start HITL run:', error);
    }
  };

  const handleApproval = async (approved: boolean, modifications?: Record<string, any>) => {
    await submitApproval(approved, modifications);
  };

  return (
    <HITLErrorBoundary>
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <h1 className="text-2xl font-bold text-gray-900">AI Workflow with HITL</h1>

        {/* Status Display */}
        {state.status !== 'idle' && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-600">Status:</span>
                <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                  state.status === 'completed' ? 'bg-green-100 text-green-800' :
                  state.status === 'error' ? 'bg-red-100 text-red-800' :
                  state.status === 'awaiting_human' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-blue-100 text-blue-800'
                }`}>
                  {state.status.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <button
                onClick={resetState}
                className="text-sm text-gray-600 hover:text-gray-800"
              >
                Reset
              </button>
            </div>
            
            {state.currentStep && (
              <div className="mt-2">
                <span className="text-sm text-gray-600">Current Step: {state.currentStep}</span>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {state.error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <h3 className="font-medium text-red-800">Error</h3>
            <p className="text-red-600 text-sm mt-1">{state.error}</p>
          </div>
        )}

        {/* Approval Interface */}
        {state.approvalPending && state.approvalToken && (
          <HITLApprovalInterface
            runId={state.runId!}
            approvalToken={state.approvalToken}
            message="Please review the validation results and approve to continue."
            actionsRequired={['review_validation', 'approve_execution']}
            validationSummary={state.validationSummary}
            onApproval={handleApproval}
          />
        )}

        {/* Workflow Form */}
        {state.status === 'idle' && (
          <AIWorkflowForm
            onSubmit={handleSubmit}
            hitlEnabled={true}
          />
        )}

        {/* Results Display */}
        {state.status === 'completed' && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="font-medium text-green-800">Workflow Completed Successfully</h3>
            <p className="text-green-600 text-sm mt-1">
              Your AI workflow has been completed with human oversight.
            </p>
          </div>
        )}
      </div>
    </HITLErrorBoundary>
  );
}
```

## Best Practices

### 1. Progressive Enhancement
- Start with basic AI workflows
- Add HITL features as optional enhancements
- Provide clear benefits explanation

### 2. User Feedback
- Show validation progress clearly
- Provide actionable error messages
- Guide users through fixing issues

### 3. Performance
- Use WebSocket connections efficiently
- Cache validation results when possible
- Implement proper loading states

### 4. Accessibility
- Ensure keyboard navigation works
- Provide screen reader support
- Use semantic HTML elements

### 5. Error Recovery
- Implement retry mechanisms
- Provide fallback options
- Save user progress when possible

This integration guide provides everything needed to implement HITL functionality in React applications, from basic API integration to complete user interfaces with real-time WebSocket communication.
