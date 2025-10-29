# Error Recovery Implementation Plan

## Overview

This document provides a complete implementation plan for adding natural language error recovery to the HITL system. It integrates structured error responses, conversation history tracking, and NL-first user interactions to allow users to fix API errors conversationally.

**Key Principle:** Natural language first - no dropdowns, no technical UI, all errors explained conversationally.

---

## Prerequisites

✅ **Already Implemented:**
- Resumability architecture (persistence.py)
- Natural language conversation mode (orchestrator.py:1010-1130)
- Form-based HITL with field classification
- State persistence (Redis/PostgreSQL)

❌ **Missing (To Be Implemented):**
- Structured error responses from API providers
- Conversation history tracking in state
- Error recovery checkpoint with NL mode
- Chat history integration
- NL parser with conversation context

---

## Implementation Phases

### Phase 1: Structured Error Responses
**Priority:** HIGH
**Estimated Time:** 2-3 hours

#### Changes Required

**1.1 Update replicate_tool.py**
**File:** `src/llm_backend/tools/replicate_tool.py`

**Current (lines 63-80):**
```python
if response.status_code not in [200, 201]:
    print(f"❌ API Error: {response.text}")

# Check if the request was successful
if response.status_code == 201 or response.status_code == 200:
    prediction = response.json()
    # ... success handling ...

return response.json(), response.status_code
```

**New Implementation:**
```python
# Check if the request was successful
if response.status_code in [200, 201]:
    prediction = response.json()
    message_type = MessageType["REPLICATE_PREDICTION"]
    send_data_to_url(
        data={
            "prediction": prediction,
            "operation_type": operation_type,
        },
        url=f"{TOHJU_NODE_API}/api/webhooks/onReplicateStarted",
        crew_input=run_input,
        message_type=message_type,
    )
    return response.json(), response.status_code

# Handle error responses
print(f"❌ API Error ({response.status_code}): {response.text}")

try:
    error_json = response.json() if response.text else {}
except json.JSONDecodeError:
    error_json = {"detail": response.text}

# Return structured error
error_response = {
    "error": True,
    "status_code": response.status_code,
    "error_message": error_json.get("detail", "") or error_json.get("error", ""),
    "raw_error": error_json
}

return error_response, response.status_code
```

**1.2 Create Error Parser**
**File:** `src/llm_backend/providers/replicate_error_parser.py` (NEW)

```python
"""
Replicate API Error Parser

Parses Replicate API error responses into structured format for error recovery.
"""

from typing import Dict, Optional, List, Any
import re

class ReplicateErrorParser:
    """Parse Replicate API errors into structured format"""

    @staticmethod
    def parse_validation_error(error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse 422 validation error into structured format.

        Example Replicate error:
        {
            "detail": "aspect_ratio: aspect_ratio must be one of the following: '16:9', '1:1', '21:9', ..."
        }

        Returns:
        {
            "error_type": "validation",
            "field": "aspect_ratio",
            "message": "aspect_ratio must be one of ...",
            "current_value": None,  # Extracted from context if available
            "valid_values": ["16:9", "1:1", "21:9", ...],
            "raw_error": {...}
        }
        """
        detail = error_data.get("detail", "")

        # Try to extract field name and validation message
        match = re.match(r"(\w+):\s*(.+)", detail)
        if not match:
            return {
                "error_type": "validation",
                "field": None,
                "message": detail,
                "valid_values": [],
                "raw_error": error_data
            }

        field_name = match.group(1)
        message = match.group(2)

        # Extract valid enum values if present
        valid_values = []
        enum_match = re.search(r"must be one of the following:\s*(.+)", message)
        if enum_match:
            # Parse: '16:9', '1:1', '21:9', ...
            values_str = enum_match.group(1)
            valid_values = re.findall(r"'([^']+)'", values_str)

        return {
            "error_type": "validation",
            "field": field_name,
            "message": message,
            "valid_values": valid_values,
            "raw_error": error_data
        }

    @staticmethod
    def parse_error(status_code: int, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse any Replicate API error.

        Returns:
        {
            "error_type": "validation" | "authentication" | "not_found" | "rate_limit" | "server_error",
            "recoverable": bool,
            "field": str | None,
            "message": str,
            "valid_values": List[str],
            "raw_error": dict
        }
        """
        # Map status codes to error types
        error_type_map = {
            401: "authentication",
            403: "authentication",
            404: "not_found",
            422: "validation",
            429: "rate_limit",
            500: "server_error",
            502: "server_error",
            503: "server_error"
        }

        error_type = error_type_map.get(status_code, "unknown")

        # Validation errors are recoverable
        recoverable = error_type == "validation"

        # Parse based on error type
        if error_type == "validation":
            parsed = ReplicateErrorParser.parse_validation_error(error_data)
            parsed["recoverable"] = True
            return parsed

        # Non-validation errors
        detail = error_data.get("detail", "") or error_data.get("error", "")
        return {
            "error_type": error_type,
            "recoverable": False,
            "field": None,
            "message": detail or f"HTTP {status_code} error",
            "valid_values": [],
            "raw_error": error_data
        }
```

**1.3 Update replicate_provider.py**
**File:** `src/llm_backend/providers/replicate_provider.py`

Add error parsing to `_execute_via_tool()` method (around line 803):

```python
from llm_backend.providers.replicate_error_parser import ReplicateErrorParser

async def _execute_via_tool(self, payload_input: PayloadInput, ...) -> ProviderResponse:
    """Execute via replicate tool"""
    try:
        # ... existing code ...

        run, status_code = run_replicate(
            run_input=self.run_input,
            model_params=self.config,
            input=payload_input,
            operation_type=operation_type,
        )

        # Check for errors
        if isinstance(run, dict) and run.get("error"):
            print(f"❌ Replicate API error: {run.get('error_message')}")

            # Parse error into structured format
            error_details = ReplicateErrorParser.parse_error(
                status_code=run.get("status_code", 500),
                error_data=run.get("raw_error", {})
            )

            return ProviderResponse(
                error=error_details["message"],
                status_code=run.get("status_code"),
                metadata={
                    "error_details": error_details,
                    "recoverable": error_details["recoverable"]
                }
            )

        # Success path
        return ProviderResponse(
            success=True,
            data=run,
            metadata={"execution_time_ms": execution_time_ms}
        )

    except Exception as e:
        print(f"❌ Unexpected error in _execute_via_tool: {e}")
        return ProviderResponse(
            error=str(e),
            status_code=500,
            metadata={
                "error_details": {
                    "error_type": "unknown",
                    "recoverable": False,
                    "message": str(e)
                }
            }
        )
```

---

### Phase 2: Conversation History Tracking
**Priority:** HIGH
**Estimated Time:** 2-3 hours

#### Changes Required

**2.1 Add conversation_history to HITLState**
**File:** `src/llm_backend/core/hitl/types.py`

Add to `HITLState` class (around line 50):

```python
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class HITLState:
    # ... existing fields ...

    # NEW: Conversation history tracking
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    """
    Full conversation history including:
    - User's original messages
    - System's NL questions
    - User's NL responses
    - Error messages and explanations
    - Clarifications

    Format:
    [
        {"role": "user", "message": "...", "timestamp": "..."},
        {"role": "assistant", "message": "...", "timestamp": "..."},
        {"role": "system", "message": "...", "timestamp": "..."}
    ]
    """
```

**2.2 Add conversation tracking to orchestrator**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Add method to track conversation (around line 200):

```python
from datetime import datetime

def _add_to_conversation(self, role: str, message: str, metadata: Optional[Dict] = None):
    """
    Add a message to the conversation history.

    Args:
        role: "user", "assistant", or "system"
        message: The message content
        metadata: Optional metadata (e.g., step, checkpoint_type)
    """
    if not hasattr(self.state, 'conversation_history'):
        self.state.conversation_history = []

    entry = {
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    if metadata:
        entry["metadata"] = metadata

    self.state.conversation_history.append(entry)

    print(f"💬 Added to conversation ({role}): {message[:100]}...")
```

Update existing NL conversation code (lines 1050-1130) to track messages:

```python
# Line ~1061 - After generating NL prompt
nl_prompt = await generate_natural_language_prompt(...)

# Track the assistant's question
self._add_to_conversation(
    role="assistant",
    message=nl_prompt.message,
    metadata={"step": "information_review", "missing_fields": nl_prompt.missing_field_names}
)

# ... pause and wait for user ...

# Line ~1094 - After receiving user response
user_message = approval.edits.get("user_message", "")

# Track the user's response
self._add_to_conversation(
    role="user",
    message=user_message,
    metadata={"step": "information_review"}
)

# Parse and continue...
```

**2.3 Create Chat History Client**
**File:** `src/llm_backend/core/hitl/chat_history_client.py` (NEW)

See full implementation in `docs/CHAT_HISTORY_INTEGRATION.md` - copy the `ChatHistoryClient` class.

**2.4 Add chat history fetching to orchestrator**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Add to `__init__` (around line 100):

```python
from llm_backend.core.hitl.chat_history_client import ChatHistoryClient

def __init__(self, ...):
    # ... existing init ...
    self.chat_history_client = ChatHistoryClient()
```

Add method to fetch full conversation (around line 250):

```python
async def _get_conversation_history(self) -> List[Dict[str, str]]:
    """
    Get full conversation history including:
    1. Database chat history (from D1 via CF Workers)
    2. Current HITL conversation (from state)

    Returns combined and sorted conversation.
    """
    # Fetch from database
    db_history = []
    if hasattr(self, 'session_id') and self.session_id:
        try:
            db_history = await self.chat_history_client.get_session_history(
                self.session_id
            )
            print(f"📚 Fetched {len(db_history)} messages from database")
        except Exception as e:
            print(f"⚠️ Failed to fetch chat history: {e}")
            db_history = []

    # Get HITL conversation from state
    hitl_history = getattr(self.state, 'conversation_history', [])
    print(f"💬 HITL conversation has {len(hitl_history)} messages")

    # Combine (database history is older, HITL history is recent)
    combined = db_history + hitl_history

    # Sort by timestamp
    combined.sort(key=lambda x: x.get('timestamp', ''))

    return combined
```

---

### Phase 3: Error Recovery Checkpoint
**Priority:** HIGH
**Estimated Time:** 3-4 hours

#### Changes Required

**3.1 Create Error Recovery NL Agent**
**File:** `src/llm_backend/agents/error_recovery_nl_agent.py` (NEW)

```python
"""
Error Recovery Natural Language Agent

Generates natural language error messages for HITL error recovery.
Uses AI to explain technical errors in user-friendly terms.
"""

import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_error_recovery_message(
    error_type: str,
    field: Optional[str],
    current_value: Optional[str],
    valid_values: List[str],
    conversation_history: List[Dict[str, str]],
    error_message: str
) -> str:
    """
    Generate a natural language error recovery message.

    Args:
        error_type: "validation", "authentication", etc.
        field: The field that caused the error
        current_value: The invalid value that was sent
        valid_values: List of valid values for the field
        conversation_history: Full conversation context
        error_message: Raw error message from API

    Returns:
        Natural language error explanation and recovery prompt
    """

    # Build conversation context
    history_context = "\n".join([
        f"{msg.get('role', 'user')}: {msg.get('message', msg.get('content', ''))[:200]}"
        for msg in conversation_history[-5:]  # Last 5 messages
    ])

    # Build valid values list for prompt
    valid_values_str = ""
    if valid_values:
        if len(valid_values) <= 10:
            valid_values_str = f"Valid options: {', '.join(valid_values)}"
        else:
            valid_values_str = f"Valid options include: {', '.join(valid_values[:10])} (and {len(valid_values)-10} more)"

    prompt = f"""You are helping a user fix an API validation error. Generate a friendly, conversational error message that:

1. Explains what went wrong in simple terms (no technical jargon)
2. Mentions the specific field if applicable
3. Provides the valid options clearly
4. Asks the user to provide a corrected value in natural language

Context:
- Error type: {error_type}
- Field: {field or "unknown"}
- Current value: {current_value or "not specified"}
- {valid_values_str}
- Technical error: {error_message}

Recent conversation:
{history_context}

Generate a natural language error message (2-3 sentences max) that explains the issue and asks the user for the correct value. Be conversational and helpful, not technical.

Example good message:
"Oops! The aspect ratio you chose isn't supported by this model. You can use 16:9 (widescreen), 1:1 (square), or 21:9 (ultra-wide). Which aspect ratio would you like?"

Your error message:"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains API errors in simple, friendly language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        nl_message = response.choices[0].message.content.strip()
        print(f"🤖 Generated error recovery message: {nl_message[:100]}...")
        return nl_message

    except Exception as e:
        print(f"❌ Failed to generate NL error message: {e}")

        # Fallback to template-based message
        if field and valid_values:
            return f"I encountered an issue with the {field} parameter. Please choose from: {', '.join(valid_values[:5])}. What would you like to use?"
        else:
            return f"I encountered an error: {error_message[:100]}. Can you help me fix this?"
```

**3.2 Add Error Recovery Handler to Orchestrator**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Add new method (around line 1400):

```python
from llm_backend.agents.error_recovery_nl_agent import generate_error_recovery_message

async def _handle_validation_error_nl(
    self,
    response: ProviderResponse,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle API validation error with natural language recovery.

    This pauses execution and asks the user to fix the error conversationally.
    """
    print("🔧 Handling validation error with NL recovery...")

    # Extract error details
    error_details = response.metadata.get("error_details", {})
    field = error_details.get("field")
    message = error_details.get("message", "")
    valid_values = error_details.get("valid_values", [])

    # Get current value from payload
    current_value = None
    if field and isinstance(payload, dict):
        current_value = payload.get("input", {}).get(field)

    # Fetch full conversation history
    conversation_history = await self._get_conversation_history()

    # Generate natural language error message using AI
    nl_error = await generate_error_recovery_message(
        error_type="validation",
        field=field,
        current_value=current_value,
        valid_values=valid_values,
        conversation_history=conversation_history,
        error_message=message
    )

    # Add error to conversation
    self._add_to_conversation(
        role="system",
        message=f"API validation error: {message}",
        metadata={"error_type": "validation", "field": field}
    )

    self._add_to_conversation(
        role="assistant",
        message=nl_error,
        metadata={"step": "error_recovery", "field": field}
    )

    # Save state before pausing
    await self.state_manager.save_state(self.run_id, self.state)

    # Pause with natural language conversation mode
    return self._create_pause_response(
        step=HITLStep.API_CALL,
        message=nl_error,
        checkpoint_type="error_recovery",
        conversation_mode=True,  # ← Enable NL mode
        data={
            "error_type": "validation",
            "error_field": field,
            "current_value": current_value,
            "valid_values": valid_values,
            "conversation_history": self.state.conversation_history,
            "failed_payload": payload  # Store for retry
        }
    )
```

**3.3 Update API Execution Step to Catch Errors**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Update `_step_api_execution()` method (around line 1300):

```python
async def _step_api_execution(self, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """Execute API call with error recovery"""

    # ... existing payload creation code ...

    print(f"🚀 Executing API call...")

    # Execute via provider
    response = await self.provider.execute(
        payload=final_payload,
        run_input=self.run_input
    )

    # Check for recoverable errors
    if response.error:
        error_details = response.metadata.get("error_details", {})
        recoverable = response.metadata.get("recoverable", False)

        print(f"❌ API error: {response.error}")
        print(f"   Recoverable: {recoverable}")
        print(f"   Error type: {error_details.get('error_type')}")

        # Handle validation errors with NL recovery
        if recoverable and error_details.get("error_type") == "validation":
            return await self._handle_validation_error_nl(response, final_payload)

        # Non-recoverable errors fail immediately
        return await self._handle_error(Exception(response.error))

    # Success path
    self._transition_to_step(HITLStep.API_CALL, HITLStatus.COMPLETED)
    self._add_step_event(
        HITLStep.API_CALL,
        HITLStatus.COMPLETED,
        "system",
        f"Executed in {response.metadata.get('execution_time_ms', 0)}ms"
    )

    # ... continue to response review ...
```

---

### Phase 4: Parse Error Recovery with History
**Priority:** MEDIUM
**Estimated Time:** 1-2 hours

#### Changes Required

**4.1 Update NL Response Parser**
**File:** `src/llm_backend/agents/nl_response_parser.py`

Add conversation_history parameter (around line 20):

```python
async def parse_natural_language_response(
    user_message: str,
    expected_schema: Dict,
    current_values: Dict,
    conversation_history: Optional[List[Dict]] = None  # ← NEW
) -> ParsedNLResponse:
    """
    Parse user's natural language response with conversation context.

    Args:
        user_message: The user's latest message
        expected_schema: Field classifications and schema
        current_values: Current form values
        conversation_history: Full conversation context for better parsing
    """

    # Build conversation context if provided
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-10:]:  # Last 10 messages
            role = msg.get("role", "user")
            content = msg.get("message") or msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content[:200]}")

        if history_lines:
            history_context = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    # Create parsing prompt with history
    prompt = f"""{history_context}Latest user message: "{user_message}"

Current field values:
{json.dumps(current_values, indent=2)}

Expected fields (classified):
{json.dumps(expected_schema.get("field_classifications", {}), indent=2)}

Task: Parse the user's message and extract field values they mentioned. Consider the conversation history, especially if they're fixing a validation error or clarifying a previous response.

Return JSON with extracted fields."""

    # ... rest of parsing logic ...
```

**4.2 Update Orchestrator to Pass History to Parser**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Update NL parsing call (around line 1094):

```python
# Fetch conversation history
conversation_history = await self._get_conversation_history()

# Parse user's natural language response WITH history
parsed_values = await parse_natural_language_response(
    user_message=user_message,
    expected_schema=classification,
    current_values=current_values,
    conversation_history=conversation_history  # ← Pass history!
)
```

---

### Phase 5: Auto-Retry After Fix
**Priority:** MEDIUM
**Estimated Time:** 1 hour

#### Changes Required

**5.1 Update Resume Logic for Error Recovery**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Update `resume_from_state()` method (around line 500):

```python
async def resume_from_state(self) -> Dict[str, Any]:
    """Resume execution from saved state, handling error recovery"""

    print(f"🔄 Resuming from state: checkpoint_type={getattr(self.state, 'checkpoint_type', 'unknown')}")

    # Special handling for error recovery
    if getattr(self.state, 'checkpoint_type', None) == "error_recovery":
        print("🔧 Resuming from error recovery - user provided fix")

        # Refresh conversation history from database
        self.state.conversation_history = await self._get_conversation_history()

        # The approval handler already updated current_values with parsed fields
        # Now we retry the API call with corrected values

        # Rebuild payload with corrected values
        payload = await self._create_payload_from_form_or_intelligent(
            self.state.form_data
        )

        print(f"🔄 Retrying API call with corrected values...")

        # Re-execute API call (this will call _step_api_execution)
        return await self._step_api_execution(payload=payload)

    # Normal resume logic
    resume_index = self._determine_resume_index()
    return await self._run_pipeline(start_index=resume_index)
```

**5.2 Update Approval Handler for Error Recovery**
**File:** `src/llm_backend/core/hitl/orchestrator.py`

Update approval handler (around line 600):

```python
async def handle_approval(self, approval: HITLApproval) -> Dict[str, Any]:
    """Handle approval with error recovery support"""

    print(f"✅ Received approval: approved={approval.approved}, checkpoint_type={getattr(self.state, 'checkpoint_type', 'unknown')}")

    if not approval.approved:
        return await self._handle_rejection(approval.edits.get("reason", "User rejected"))

    # Extract user message for NL conversation mode
    user_message = approval.edits.get("user_message", "")

    # For error recovery, parse the user's fix
    if getattr(self.state, 'checkpoint_type', None) == "error_recovery":
        print("🔧 Processing error recovery response...")

        # Track user's response in conversation
        self._add_to_conversation(
            role="user",
            message=user_message,
            metadata={"step": "error_recovery"}
        )

        # Get error field from pause data
        pause_data = getattr(self.state, 'pause_data', {})
        error_field = pause_data.get("error_field")

        # Fetch conversation history for context
        conversation_history = await self._get_conversation_history()

        # Parse user's fix with full conversation context
        classification = self.state.form_data.get("classification", {})
        current_values = self.state.form_data.get("current_values", {})

        parsed_response = await parse_natural_language_response(
            user_message=user_message,
            expected_schema=classification,
            current_values=current_values,
            conversation_history=conversation_history
        )

        # Update form with corrected values
        extracted_fields = parsed_response.extracted_fields
        print(f"🔧 Extracted corrected values: {extracted_fields}")

        self.state.form_data["current_values"].update(extracted_fields)

        # Save state
        await self.state_manager.save_state(self.run_id, self.state)

        # Resume will retry API call
        return await self.resume_from_state()

    # Normal approval logic
    # ... existing code ...
```

---

## Testing Strategy

### Test Case 1: Validation Error Recovery

**File:** `tests/test_error_recovery.py` (NEW)

```python
import pytest
from llm_backend.core.hitl.orchestrator import HITLOrchestrator
from llm_backend.providers.replicate_provider import ReplicateProvider

@pytest.mark.asyncio
async def test_validation_error_recovery():
    """Test that validation errors trigger NL error recovery"""

    # Setup orchestrator with mock provider that returns 422 error
    # ...

    # Run pipeline
    result = await orchestrator.execute()

    # Should pause with error recovery checkpoint
    assert result["status"] == "awaiting_human"
    assert result["checkpoint_type"] == "error_recovery"
    assert result["conversation_mode"] is True

    # Error message should be in natural language
    assert "aspect ratio" in result["message"].lower()
    assert "choose from" in result["message"].lower()

    # Simulate user fixing error
    approval = HITLApproval(
        approved=True,
        edits={"user_message": "Use 1:1 aspect ratio"}
    )

    # Resume should parse and retry
    result = await orchestrator.handle_approval(approval)

    # Should complete successfully
    assert result["status"] == "completed"
```

### Test Case 2: Chat History Context

```python
@pytest.mark.asyncio
async def test_error_recovery_with_chat_history():
    """Test that parser uses chat history for context"""

    # Mock chat history with previous "1:1" mention
    mock_history = [
        {"role": "user", "message": "Create a 1:1 image of sunset", "timestamp": "..."},
        {"role": "assistant", "message": "I need more details...", "timestamp": "..."}
    ]

    # Mock error: aspect_ratio invalid
    # User responds: "Like I said before, square"

    # Parser should extract "1:1" by understanding:
    # - User originally said "1:1"
    # - "square" = "1:1"

    # ... test implementation ...
```

### Manual Testing

**Test Script:** `test_error_recovery.sh`

```bash
#!/bin/bash

# Test error recovery with invalid aspect_ratio

curl -X POST http://localhost:8000/api/teams/run?enable_hitl=true \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create an image of a sunset",
    "user_email": "test@example.com",
    "user_id": "user123",
    "agent_email": "agent@example.com",
    "session_id": "test-session-error-recovery",
    "message_type": "REPLICATE_PREDICTION",
    "agent_tool_config": {
      "replicate-agent-tool": {
        "data": {
          "model_name": "black-forest-labs/flux-1.1-pro",
          "example_input": {
            "prompt": "",
            "aspect_ratio": "16:9",
            "num_outputs": 1
          }
        }
      }
    },
    "hitl_config": {
      "policy": "auto_with_thresholds",
      "use_natural_language_hitl": true
    }
  }'

# Expected flow:
# 1. System asks for prompt
# 2. User provides: "sunset in square format"
# 3. AI misinterprets as aspect_ratio="match_input_img"
# 4. API returns 422 error
# 5. System explains error in NL: "The aspect ratio isn't supported. Try 16:9, 1:1..."
# 6. User responds: "Use 1:1"
# 7. System parses, retries, succeeds
```

---

## Deployment Checklist

- [ ] Phase 1: Structured error responses implemented and tested
- [ ] Phase 2: Conversation history tracking implemented and tested
- [ ] Phase 3: Error recovery checkpoint implemented and tested
- [ ] Phase 4: NL parser with history context implemented and tested
- [ ] Phase 5: Auto-retry after fix implemented and tested
- [ ] CF Workers endpoint added for chat history fetching
- [ ] Integration tests passing
- [ ] Manual testing completed
- [ ] Documentation updated
- [ ] Error logging and monitoring configured

---

## Success Metrics

✅ **Users never see technical error messages**
✅ **Recoverable errors don't cause complete failures**
✅ **Conversation history improves parsing accuracy**
✅ **Users can fix errors without restarting the entire flow**
✅ **System explains what went wrong and how to fix it**
✅ **Error recovery rate: >80% of validation errors resolved conversationally**

---

## Rollback Plan

If issues arise:

1. **Disable error recovery:** Set environment variable `ENABLE_ERROR_RECOVERY=false`
2. **Fall back to old behavior:** Errors cause immediate failure with technical message
3. **Monitor logs:** Check for parsing failures or infinite retry loops
4. **Gradual rollout:** Enable for specific users/sessions first

---

## Future Enhancements

1. **Multi-turn clarification:** Ask follow-up questions if user's fix is ambiguous
2. **Error prediction:** Detect likely validation errors before API call
3. **Batch error recovery:** Handle multiple validation errors in one conversation
4. **Visual error feedback:** Show field highlighting in UI (while keeping NL primary)
5. **Error analytics:** Track common validation errors to improve AI prompts
