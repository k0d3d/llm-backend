# Error Recovery System Analysis

## Current State Review

### Resumability Architecture ✅

**Status:** FULLY IMPLEMENTED

The HITL system has comprehensive resumability:

#### 1. State Persistence
**File:** `src/llm_backend/core/hitl/persistence.py`

- ✅ States saved to Redis and/or PostgreSQL
- ✅ `HITLStateStore` interface with save/load/resume operations
- ✅ Supports both sync and async storage backends
- ✅ State includes: run_id, status, current_step, form_data, human_edits, step_events

#### 2. Pipeline Resumption
**File:** `src/llm_backend/core/hitl/orchestrator.py`

```python
async def resume_from_state(self) -> Dict[str, Any]:
    """Resume execution from the current step recorded in state."""
    resume_index = self._determine_resume_index()
    return await self._run_pipeline(start_index=resume_index)
```

- ✅ `_determine_resume_index()` finds where to continue
- ✅ Pipeline steps: FORM_INITIALIZATION → INFORMATION_REVIEW → PAYLOAD_REVIEW → API_CALL → RESPONSE_REVIEW
- ✅ Each step can pause with `status="awaiting_human"`

#### 3. Human Edits Tracking
**File:** `orchestrator.py` lines 53-54, 85-109

```python
# Track human edits across checkpoints
self.state.human_edits = {}

# Collected from multiple sources:
# 1. state.human_edits
# 2. state.last_approval.edits
# 3. state.suggested_payload (backward compat)
```

- ✅ Edits persisted across pause/resume cycles
- ✅ Applied during API payload creation

---

## Natural Language Conversation Mode ✅

**Status:** IMPLEMENTED (lines 1010-1130)

### Current NL Flow

1. **Generate NL Prompt** (line 1022):
   ```python
   nl_prompt = await generate_natural_language_prompt(
       classification=classification,
       current_values=current_values,
       missing_fields=missing_fields
   )
   ```

2. **Pause with Conversation Mode** (line 1050-1061):
   ```python
   pause_response = {
       "checkpoint_type": "information_request",
       "conversation_mode": True,  # ← Signals NL mode
       "nl_prompt": nl_prompt.message,
       "missing_fields": nl_prompt.missing_field_names
   }
   ```

3. **Wait for User Response** (line 1066-1072):
   - WebSocket waits for user message
   - User sends natural language text

4. **Parse User Response** (line 1094-1099):
   ```python
   parsed_values = await parse_natural_language_response(
       user_message=user_message,
       expected_schema=classification,
       current_values=current_values
   )
   ```

5. **Update Form & Continue** (line 1114):
   ```python
   self.state.form_data["current_values"].update(parsed_values.extracted_fields)
   ```

### What's Missing for Error Recovery

❌ **No chat history included in NL parsing**
- Current: Only sends single user message to parser
- Needed: Full conversation context (original prompt + error + user response)

❌ **No error recovery checkpoint**
- Current: API errors cause complete FAILURE via `_handle_error()`
- Needed: Pause at API_CALL step with error context for recovery

❌ **No multi-turn clarification**
- Line 1107-1111: `# TODO: Implement multi-turn clarification`
- Can't ask follow-up questions if response is ambiguous

---

## Error Handling Status

### Current Error Flow ❌

**File:** `orchestrator.py` line 1728-1737

```python
async def _handle_error(self, error: Exception) -> Dict[str, Any]:
    self._transition_to_step(self.state.current_step, HITLStatus.FAILED)
    self._add_step_event(self.state.current_step, HITLStatus.FAILED, "system", str(error))

    return {
        "run_id": self.run_id,
        "status": "failed",
        "error": str(error),
        "failed_at_step": self.state.current_step
    }
```

**Problems:**
1. ❌ Fails immediately with no recovery attempt
2. ❌ No differentiation between recoverable/non-recoverable errors
3. ❌ No HITL checkpoint for user to fix issues
4. ❌ No natural language error explanation

### API Execution Error Handling ❌

**File:** `replicate_provider.py` line 803-811

```python
except Exception as e:
    return ProviderResponse(
        error=str(e)
    )
```

**Problems:**
1. ❌ Catches all exceptions but doesn't distinguish error types
2. ❌ No structured error data (just string)
3. ❌ Orchestrator can't tell if error is recoverable

**File:** `replicate_tool.py` line 62-64

```python
if response.status_code not in [200, 201]:
    print(f"❌ API Error: {response.text}")
# ... continues anyway!
```

**Problems:**
1. ❌ Logs error but returns malformed response anyway
2. ❌ No structured error parsing
3. ❌ No status_code checking in caller

---

## Gap Analysis: What We Need

### 1. Structured Error Responses ❌

**Current:** API errors return string
**Needed:** Structured error data

```python
# Current
return ProviderResponse(error="API failed")

# Needed
return ProviderResponse(
    error="Validation error: aspect_ratio invalid",
    status_code=422,
    error_type="validation",
    error_details={
        "field": "aspect_ratio",
        "current_value": "match_input_img",
        "valid_values": ["16:9", "1:1", "21:9", ...],
        "raw_error": {...}
    }
)
```

### 2. Error Recovery Checkpoint ❌

**Current:** No checkpoint for API errors
**Needed:** Pause at API_CALL with error context

```python
if response.status_code == 422:
    return self._create_pause_response(
        step=HITLStep.API_CALL,
        message="There was an issue with the request. Let me explain...",
        checkpoint_type="error_recovery",
        conversation_mode=True,  # ← Use NL conversation
        data={
            "error_type": "validation",
            "error_field": "aspect_ratio",
            "nl_error_message": "The aspect ratio you chose isn't supported...",
            "conversation_history": [
                {"role": "user", "message": "Create image 1:1"},
                {"role": "assistant", "message": "I interpreted that as..."},
                {"role": "system", "message": "Error: Invalid value"}
            ]
        }
    )
```

### 3. Chat History in NL Parser ❌

**Current:** Parser only sees single message
**Needed:** Full conversation context

```python
# Current
parsed_values = await parse_natural_language_response(
    user_message=user_message,  # ← Only this
    expected_schema=classification,
    current_values=current_values
)

# Needed
parsed_values = await parse_natural_language_response(
    user_message=user_message,
    expected_schema=classification,
    current_values=current_values,
    conversation_history=[  # ← Add this
        {"role": "user", "message": "Create image with 1:1 ratio"},
        {"role": "assistant", "message": "I need an image description..."},
        {"role": "user", "message": "A sunset over mountains"},
        {"role": "system", "message": "Error: aspect_ratio 'match_input_img' invalid"},
        {"role": "assistant", "message": "The aspect ratio isn't valid. Try 16:9, 1:1..."},
        {"role": "user", "message": "Use 1:1"}  # ← Current message
    ]
)
```

### 4. Conversation History Storage ❌

**Current:** Not stored
**Needed:** Track conversation in state

```python
class HITLState:
    conversation_history: List[Dict[str, str]] = []  # ← Add this

    # Track all messages:
    # - User's original prompt
    # - System's NL questions
    # - User's NL responses
    # - Error messages
    # - Clarifications
```

---

## Error Recovery Plan (Natural Language First)

### Phase 1: Structured Error Responses

**Changes:**

1. **`replicate_tool.py`**: Return structured error data
   ```python
   if response.status_code not in [200, 201]:
       error_json = response.json() if response.text else {}
       return {
           "error": True,
           "status_code": response.status_code,
           "error_message": error_json.get("detail", ""),
           "raw_error": error_json
       }, response.status_code
   ```

2. **`replicate_provider.py`**: Parse and structure errors
   ```python
   if status_code not in [200, 201]:
       error_details = self._parse_api_error(run, status_code)
       return ProviderResponse(
           error=error_details["message"],
           status_code=status_code,
           metadata={"error_details": error_details}
       )
   ```

### Phase 2: Add Conversation History to State

**Changes:**

1. **`types.py`**: Add conversation tracking
   ```python
   class HITLState:
       conversation_history: List[Dict[str, str]] = field(default_factory=list)
   ```

2. **`orchestrator.py`**: Track all messages
   ```python
   def _add_to_conversation(self, role: str, message: str):
       self.state.conversation_history.append({
           "role": role,
           "message": message,
           "timestamp": datetime.utcnow().isoformat()
       })
   ```

### Phase 3: Error Recovery Checkpoint (Natural Language)

**Changes:**

1. **`orchestrator.py._step_api_execution()`**: Catch recoverable errors
   ```python
   response = self.provider.execute(payload)

   # Check for recoverable validation errors
   if response.status_code == 422:
       return await self._handle_validation_error_nl(response)
   ```

2. **New Method:** `_handle_validation_error_nl()`
   ```python
   async def _handle_validation_error_nl(self, response) -> Dict:
       # Parse error
       error_details = response.metadata.get("error_details", {})

       # Generate NL error message (using AI)
       nl_error = await generate_error_recovery_message(
           error_type=error_details.get("error_type"),
           field=error_details.get("field"),
           current_value=error_details.get("current_value"),
           valid_values=error_details.get("valid_values"),
           conversation_history=self.state.conversation_history
       )

       # Add to conversation
       self._add_to_conversation("system", "Error occurred")
       self._add_to_conversation("assistant", nl_error)

       # Pause with NL conversation mode
       return self._create_pause_response(
           step=HITLStep.API_CALL,
           message=nl_error,
           checkpoint_type="error_recovery",
           conversation_mode=True,  # ← Natural language response
           data={
               "error_type": "validation",
               "conversation_history": self.state.conversation_history,
               "current_form_values": self.state.form_data["current_values"]
           }
       )
   ```

### Phase 4: Parse Error Recovery Response with History

**Changes:**

1. **`nl_response_parser.py`**: Accept conversation history
   ```python
   async def parse_natural_language_response(
       user_message: str,
       expected_schema: Dict,
       current_values: Dict,
       conversation_history: List[Dict] = None  # ← Add this
   ):
       # Include history in AI prompt for better context
       history_context = "\n".join([
           f"{msg['role']}: {msg['message']}"
           for msg in (conversation_history or [])
       ])

       prompt = f"""Conversation so far:
   {history_context}

   Latest user message: "{user_message}"

   Parse the message to extract field values, considering the full conversation context."""
   ```

### Phase 5: Retry After Fix

**Changes:**

1. **`orchestrator.py`**: Auto-retry after user fixes error
   ```python
   async def resume_from_state(self):
       # If resuming from error_recovery checkpoint
       if self.state.checkpoint_type == "error_recovery":
           print("🔄 Retrying after error fix...")
           # User's response was parsed and updated current_values
           # Re-run API_CALL step with corrected values
           return await self._step_api_execution()
   ```

---

## Key Principles

### ✅ Natural Language First
- No dropdowns or technical UI
- All errors explained conversationally
- User responds in natural language
- AI parses response with full context

### ✅ Conversation History
- Track entire conversation thread
- Include in all AI prompts
- Helps with ambiguous responses
- Provides context for error recovery

### ✅ Resumability
- Every pause point is resumable
- State fully serialized
- Can retry/recover without restart

### ✅ Graceful Degradation
- Recoverable errors → HITL pause
- Non-recoverable errors → Fail with explanation
- Always inform user what happened

---

## Implementation Priority

1. **HIGH**: Structured error responses (Phase 1)
2. **HIGH**: Conversation history tracking (Phase 2)
3. **HIGH**: Error recovery checkpoint (Phase 3)
4. **MEDIUM**: Parse with history context (Phase 4)
5. **MEDIUM**: Auto-retry mechanism (Phase 5)

---

## Testing Strategy

### Test Cases:

1. **Validation Error Recovery**
   - User: "Create 1:1 image of sunset"
   - System: Sets aspect_ratio="match_input_img" (AI mistake)
   - API: 422 error
   - System: "Oops! That aspect ratio isn't supported. Try 16:9, 1:1, 2:3..."
   - User: "Use 1:1"
   - System: Parses "1:1", retries API
   - Result: Success ✅

2. **Multi-Field Error**
   - Multiple validation errors
   - System explains each in NL
   - User fixes all in one response
   - System parses all, retries

3. **Ambiguous Response**
   - User: "square format"
   - System: "Did you mean 1:1 or 4:5?"
   - User: "1 to 1"
   - System: Parses correctly with history context

4. **Non-Recoverable Error**
   - 401 Unauthorized
   - System: "I can't access the API. Please check credentials."
   - Status: FAILED (no retry)

---

## Files to Create/Modify

### New Files:
1. `src/llm_backend/agents/error_recovery_nl_agent.py` - Generate NL error messages

### Modified Files:
1. `src/llm_backend/tools/replicate_tool.py` - Return structured errors
2. `src/llm_backend/providers/replicate_provider.py` - Parse error responses
3. `src/llm_backend/core/hitl/types.py` - Add conversation_history to HITLState
4. `src/llm_backend/core/hitl/orchestrator.py`:
   - Add `_add_to_conversation()`
   - Add `_handle_validation_error_nl()`
   - Update `_step_api_execution()` to catch errors
   - Track conversation in all NL interactions
5. `src/llm_backend/agents/nl_response_parser.py` - Accept conversation_history parameter

---

## Success Metrics

✅ **User never sees technical error messages**
✅ **Recoverable errors don't cause failures**
✅ **Conversation history improves parsing accuracy**
✅ **Users can fix errors without restarting**
✅ **System explains what went wrong and how to fix it**

