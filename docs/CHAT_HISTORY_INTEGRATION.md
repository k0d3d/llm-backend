# Chat History Integration for HITL Error Recovery

## Overview

This document outlines how to integrate chat history with the HITL system, specifically for error recovery scenarios where users provide feedback in natural language. The goal is to maintain conversation context across HITL pause/resume cycles.

---

## Current State Analysis

### Chat History Flow (Initial Request)

**File:** `cf-workers/core-api-d1kvr2/src/routes/messages.ts`

1. **Client sends message** to `/to-llm` endpoint (lines 37-98)
2. **Fetch chat history** from D1 database:
   ```typescript
   const chatHistory = await D1Service.getSessionChatHistory(
     c.env.DB,
     data.sessionId
   )
   ```
3. **Send to queue** with chat history included:
   ```typescript
   await sendToQueue(c.env.RAG_RESULTS_QUEUE, {
     ...data,
     chatHistory: chatHistory || []
   })
   ```

**File:** `cf-workers/core-api-d1kvr2/src/lib/D1Service.ts` (lines 264-290)

```typescript
static async getSessionChatHistory(db: D1Database, sessionId: string) {
  const messages = await db
    .prepare(`
      SELECT * FROM messages
      WHERE session_id = ?
      AND (pinned = 1 OR created_at > datetime('now', '-1 hour'))
      ORDER BY created_at ASC
    `)
    .bind(sessionId)
    .all();

  return messages.results;
}
```

### HITL API Flow

**File:** `cf-workers/core-api-d1kvr2/src/routes/hitl.ts`

1. **Create HITL run** via `/run` endpoint (lines 37-69):
   ```typescript
   // Line 12: chatHistory is optional parameter
   chatHistory?: Array<{role: string, content: string}>

   // Lines 44-59: Forward to llm-backend
   const response = await fetch(`${LLM_BACKEND_URL}/api/teams/run`, {
     body: JSON.stringify({
       ...body,
       chatHistory  // ← Currently not passed!
     })
   })
   ```

2. **Approve HITL checkpoint** via `/approve` endpoint (lines 124-154):
   ```typescript
   // User provides edits/feedback
   const approval = await fetch(`${LLM_BACKEND_URL}/api/hitl/${runId}/approve`, {
     body: JSON.stringify({
       approved: true,
       edits: body.edits,  // ← User's feedback
       step: body.step
     })
   })
   ```

### Gap: HITL Feedback Not in Chat History

**Problem:**
1. User responds to HITL pause with natural language feedback
2. Feedback goes to `/approve` endpoint as `edits` parameter
3. **Feedback is NOT stored as a message in the database**
4. When orchestrator resumes, it has NO chat history context
5. NL parser only sees current user message, loses conversation thread

---

## Three Integration Approaches

### Approach A: llm-backend Fetches Chat History ⭐ RECOMMENDED

**Concept:** llm-backend directly queries the messages database when needed.

#### Architecture

```
┌─────────────────┐
│  CF Workers     │
│  (messages.ts)  │
└────────┬────────┘
         │ 1. Initial request with sessionId
         ▼
┌─────────────────────────────────────┐
│  llm-backend (orchestrator.py)      │
│                                     │
│  2. On pause (error recovery):      │
│     - Return to client              │
│                                     │
│  3. On resume (user responds):      │
│     - Fetch chat history via API    │
│     - Include in NL parser          │
└─────────────────────────────────────┘
         │
         │ 4. GET /api/sessions/{sessionId}/messages
         ▼
┌─────────────────┐
│  CF Workers     │
│  (D1 Database)  │
└─────────────────┘
```

#### Implementation

**Step 1:** Add chat history fetch endpoint to CF Workers

**File:** `cf-workers/core-api-d1kvr2/src/routes/messages.ts` (NEW)

```typescript
// GET /api/sessions/:sessionId/messages
app.get('/api/sessions/:sessionId/messages', async (c) => {
  const { sessionId } = c.req.param();

  try {
    const chatHistory = await D1Service.getSessionChatHistory(
      c.env.DB,
      sessionId
    );

    return c.json({
      success: true,
      messages: chatHistory,
      count: chatHistory.length
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});
```

**Step 2:** Add chat history client to llm-backend

**File:** `src/llm_backend/core/hitl/chat_history_client.py` (NEW)

```python
import os
import httpx
from typing import List, Dict, Optional
from datetime import datetime

CORE_API_URL = os.getenv("CORE_API_URL", "https://core-api-d1kvr2.asyncdev.workers.dev")

class ChatHistoryClient:
    """Client for fetching chat history from Core API"""

    def __init__(self, base_url: str = CORE_API_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_session_history(
        self,
        session_id: str,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Fetch chat history for a session.

        Returns:
            List of messages in format:
            [
                {"role": "user", "content": "...", "timestamp": "..."},
                {"role": "assistant", "content": "...", "timestamp": "..."}
            ]
        """
        try:
            url = f"{self.base_url}/api/sessions/{session_id}/messages"
            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            messages = data.get("messages", [])

            # Convert to standardized format
            formatted_messages = []
            for msg in messages:
                role = self._normalize_role(msg.get("sender", "user"))

                # Skip system messages if requested
                if not include_system and role == "system":
                    continue

                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("created_at", "")
                })

            return formatted_messages

        except httpx.HTTPError as e:
            print(f"⚠️ Failed to fetch chat history: {e}")
            return []

    def _normalize_role(self, sender: str) -> str:
        """Normalize sender to standard roles"""
        sender_lower = sender.lower()
        if "user" in sender_lower:
            return "user"
        elif "assistant" in sender_lower or "agent" in sender_lower:
            return "assistant"
        else:
            return "system"

    async def close(self):
        await self.client.aclose()
```

**Step 3:** Integrate with orchestrator

**File:** `src/llm_backend/core/hitl/orchestrator.py` (MODIFY)

```python
from llm_backend.core.hitl.chat_history_client import ChatHistoryClient

class HITLOrchestrator:
    def __init__(self, ...):
        # ... existing code ...
        self.chat_history_client = ChatHistoryClient()

    async def _handle_validation_error_nl(
        self,
        response: ProviderResponse
    ) -> Dict:
        """Handle API validation error with natural language recovery"""

        # 1. Fetch current chat history
        chat_history = await self._get_conversation_history()

        # 2. Parse error details
        error_details = response.metadata.get("error_details", {})

        # 3. Generate NL error message using AI
        from llm_backend.agents.error_recovery_nl_agent import (
            generate_error_recovery_message
        )

        nl_error = await generate_error_recovery_message(
            error_type=error_details.get("error_type"),
            field=error_details.get("field"),
            current_value=error_details.get("current_value"),
            valid_values=error_details.get("valid_values"),
            conversation_history=chat_history  # ← Include history!
        )

        # 4. Add error to conversation history
        self._add_to_conversation("system", "API validation error occurred")
        self._add_to_conversation("assistant", nl_error)

        # 5. Pause with NL conversation mode
        return self._create_pause_response(
            step=HITLStep.API_CALL,
            message=nl_error,
            checkpoint_type="error_recovery",
            conversation_mode=True,
            data={
                "error_type": "validation",
                "conversation_history": self.state.conversation_history,
                "current_form_values": self.state.form_data["current_values"]
            }
        )

    async def _get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get full conversation history including:
        1. Database chat history (from D1)
        2. Current HITL conversation (from state)
        """
        # Fetch from database
        db_history = []
        if hasattr(self, 'session_id') and self.session_id:
            db_history = await self.chat_history_client.get_session_history(
                self.session_id
            )

        # Combine with HITL conversation
        hitl_history = getattr(self.state, 'conversation_history', [])

        # Merge (deduplicate by timestamp if needed)
        combined = db_history + hitl_history

        # Sort by timestamp
        combined.sort(key=lambda x: x.get('timestamp', ''))

        return combined

    def _add_to_conversation(self, role: str, message: str):
        """Add message to HITL conversation history"""
        if not hasattr(self.state, 'conversation_history'):
            self.state.conversation_history = []

        self.state.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def resume_from_state(self) -> Dict[str, Any]:
        """Resume with fresh chat history"""

        # Refresh conversation history from database
        if self.state.checkpoint_type == "error_recovery":
            print("🔄 Refreshing chat history for error recovery...")
            self.state.conversation_history = await self._get_conversation_history()

        # Continue with normal resume logic
        resume_index = self._determine_resume_index()
        return await self._run_pipeline(start_index=resume_index)
```

**Step 4:** Update NL parser to use history

**File:** `src/llm_backend/agents/nl_response_parser.py` (MODIFY)

```python
async def parse_natural_language_response(
    user_message: str,
    expected_schema: Dict,
    current_values: Dict,
    conversation_history: List[Dict] = None  # ← NEW parameter
) -> ParsedNLResponse:
    """
    Parse user's natural language response with conversation context.
    """

    # Build conversation context for AI
    history_context = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-10:]:  # Last 10 messages
            role = msg.get("role", "user")
            content = msg.get("message") or msg.get("content", "")
            history_lines.append(f"{role}: {content}")

        history_context = "\n".join(history_lines)

    # Create AI prompt with history
    prompt = f"""You are parsing a user's response in a conversation about API parameters.

Conversation history:
{history_context}

Latest user message: "{user_message}"

Current field values:
{json.dumps(current_values, indent=2)}

Expected schema:
{json.dumps(expected_schema, indent=2)}

Parse the user's message and extract any field values they mentioned.
Consider the full conversation context, especially if they're fixing a previous error.

Return JSON with extracted fields.
"""

    # Call AI to parse
    # ... existing parsing logic ...
```

#### Pros
✅ **Full control** - llm-backend owns the conversation state
✅ **Real-time** - Always fetches latest messages
✅ **Simple integration** - Just one API call
✅ **No data duplication** - Single source of truth (D1 database)

#### Cons
❌ **Extra network call** - Adds latency on resume
❌ **Dependency** - llm-backend depends on CF Workers API
❌ **Auth needed** - Must secure the messages endpoint

---

### Approach B: CF Workers Passes Chat History

**Concept:** CF Workers fetches and includes chat history when calling HITL endpoints.

#### Architecture

```
┌─────────────────────────────────────┐
│  CF Workers (hitl.ts)               │
│                                     │
│  1. User approves/resumes HITL      │
│  2. Fetch chat history from D1      │
│  3. Include in request body         │
└────────┬────────────────────────────┘
         │
         │ POST /api/hitl/{runId}/approve
         │ { chatHistory: [...] }
         ▼
┌─────────────────────────────────────┐
│  llm-backend (orchestrator.py)      │
│                                     │
│  4. Receive chat history in request │
│  5. Use in NL parser                │
└─────────────────────────────────────┘
```

#### Implementation

**Step 1:** Modify HITL approve endpoint in CF Workers

**File:** `cf-workers/core-api-d1kvr2/src/routes/hitl.ts` (MODIFY lines 124-154)

```typescript
app.post('/api/hitl/:runId/approve', async (c) => {
  const { runId } = c.req.param();
  const body = await c.req.json();

  // Fetch chat history for this session
  let chatHistory = [];
  if (body.sessionId) {
    chatHistory = await D1Service.getSessionChatHistory(
      c.env.DB,
      body.sessionId
    );
  }

  // Forward to llm-backend with chat history
  const response = await fetch(`${LLM_BACKEND_URL}/api/hitl/${runId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...body,
      chatHistory  // ← Include chat history!
    })
  });

  return response;
});
```

**Step 2:** Update llm-backend approval endpoint

**File:** `src/llm_backend/api/endpoints/hitl.py` (MODIFY)

```python
@router.post("/{run_id}/approve")
async def approve_checkpoint(
    run_id: str,
    approval: HITLApproval,
    chat_history: Optional[List[Dict[str, str]]] = None  # ← NEW parameter
):
    """Approve a HITL checkpoint with optional chat history"""

    try:
        orchestrator = active_orchestrators.get(run_id)

        if not orchestrator:
            raise HTTPException(status_code=404, detail="HITL run not found")

        # Store chat history in orchestrator state
        if chat_history:
            orchestrator.state.conversation_history = chat_history

        # Continue with approval logic
        result = await orchestrator.handle_approval(approval)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Pros
✅ **No extra calls** - Chat history bundled with approval
✅ **CF Workers controls flow** - Centralized message management
✅ **Simple llm-backend** - Just receives history, doesn't fetch

#### Cons
❌ **Larger payloads** - Chat history sent on every approval
❌ **CF Workers complexity** - Must fetch and include history
❌ **Potential staleness** - History fetched at approval time, not parse time

---

### Approach C: Store in HITL State, Sync Periodically

**Concept:** HITL state tracks conversation, syncs to database periodically.

#### Architecture

```
┌─────────────────────────────────────┐
│  llm-backend (orchestrator.py)      │
│                                     │
│  1. Track all messages in state     │
│  2. On pause: sync to database      │
│  3. On resume: load from state      │
└────────┬────────────────────────────┘
         │
         │ Periodic sync
         ▼
┌─────────────────┐
│  Messages DB    │
│  (via API)      │
└─────────────────┘
```

#### Implementation

**Step 1:** Add message sync client

**File:** `src/llm_backend/core/hitl/message_sync.py` (NEW)

```python
import httpx
from typing import List, Dict

class MessageSyncClient:
    """Syncs HITL conversation messages to database"""

    def __init__(self, base_url: str, session_id: str):
        self.base_url = base_url
        self.session_id = session_id
        self.client = httpx.AsyncClient()

    async def sync_messages(self, messages: List[Dict]):
        """Push messages to database"""
        try:
            url = f"{self.base_url}/api/sessions/{self.session_id}/messages/batch"
            response = await self.client.post(url, json={"messages": messages})
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            print(f"⚠️ Failed to sync messages: {e}")
            return False
```

**Step 2:** Integrate with orchestrator

```python
class HITLOrchestrator:
    async def _create_pause_response(self, ...):
        # Before pausing, sync conversation to database
        await self._sync_conversation_to_db()

        # ... existing pause logic ...

    async def _sync_conversation_to_db(self):
        """Sync conversation history to messages database"""
        if not hasattr(self.state, 'conversation_history'):
            return

        syncer = MessageSyncClient(CORE_API_URL, self.session_id)
        await syncer.sync_messages(self.state.conversation_history)
```

#### Pros
✅ **Full conversation in state** - Easy access during execution
✅ **Resumable** - State includes all context
✅ **Batch syncing** - Efficient database writes

#### Cons
❌ **Duplication** - Messages in both state and database
❌ **Sync complexity** - Must handle conflicts and failures
❌ **State size** - Long conversations increase serialization overhead

---

## Recommended Approach: A (llm-backend Fetches)

**Rationale:**

1. **Single source of truth** - Database is authoritative for chat history
2. **Clean separation** - CF Workers manages messages, llm-backend manages HITL
3. **Real-time accuracy** - Always fetches latest messages including user feedback
4. **Minimal payload size** - Only sessionId passed between services
5. **Simpler state** - HITL state stays focused on form/checkpoint data

**Trade-off Accepted:**
- Extra API call adds ~50-100ms latency, but ensures accuracy

---

## Implementation Plan

### Phase 1: Add Chat History Endpoint
**Priority:** HIGH
**Files:** `cf-workers/core-api-d1kvr2/src/routes/messages.ts`

- [ ] Add GET `/api/sessions/:sessionId/messages` endpoint
- [ ] Return formatted chat history from D1
- [ ] Add authentication (API key or session token)

### Phase 2: Add Chat History Client
**Priority:** HIGH
**Files:** `src/llm_backend/core/hitl/chat_history_client.py` (NEW)

- [ ] Create `ChatHistoryClient` class
- [ ] Implement `get_session_history()` method
- [ ] Handle errors gracefully (return empty list on failure)

### Phase 3: Integrate with Orchestrator
**Priority:** HIGH
**Files:** `src/llm_backend/core/hitl/orchestrator.py`, `src/llm_backend/core/hitl/types.py`

- [ ] Add `conversation_history` field to `HITLState`
- [ ] Add `_get_conversation_history()` method to orchestrator
- [ ] Add `_add_to_conversation()` method
- [ ] Update `resume_from_state()` to fetch history

### Phase 4: Update Error Recovery
**Priority:** HIGH
**Files:** `src/llm_backend/core/hitl/orchestrator.py`

- [ ] Modify `_handle_validation_error_nl()` to fetch history
- [ ] Include history in error recovery message generation
- [ ] Store history in pause response

### Phase 5: Update NL Parser
**Priority:** MEDIUM
**Files:** `src/llm_backend/agents/nl_response_parser.py`

- [ ] Add `conversation_history` parameter
- [ ] Include history in AI prompt
- [ ] Test parsing accuracy improvement

### Phase 6: Store HITL Feedback as Messages
**Priority:** MEDIUM
**Files:** `cf-workers/core-api-d1kvr2/src/routes/hitl.ts`

- [ ] When user approves with feedback, store as message in D1
- [ ] Include metadata: `messageType: "hitl_feedback"`
- [ ] Link to HITL run ID

---

## Testing Strategy

### Test Case 1: Error Recovery with Chat History

**Scenario:**
1. User: "Create a 1:1 sunset image"
2. System: Sets aspect_ratio="match_input_img" (AI mistake)
3. API: Returns 422 validation error
4. System: Fetches chat history, generates NL error message
5. User: "Use 1:1 ratio"
6. System: Parses with history context, extracts "1:1", retries API
7. Result: Success ✅

**Validation:**
- Chat history includes original request
- Error message references user's original "1:1" mention
- Parser correctly extracts "1:1" even with ambiguous phrasing

### Test Case 2: Multi-Turn Error Recovery

**Scenario:**
1. User: "square image"
2. System: Sets aspect_ratio="1:1"
3. API: Error (different field issue)
4. User: "Fix it"
5. System: Uses history to understand "Fix it" refers to aspect ratio
6. Result: Success ✅

---

## Success Metrics

✅ **Error recovery works conversationally** - Users fix errors in natural language
✅ **Parser accuracy improves** - Ambiguous responses resolved with history context
✅ **No message loss** - All HITL feedback stored and retrievable
✅ **Resumable across restarts** - Chat history persists beyond orchestrator lifetime

---

## Security Considerations

1. **Authenticate messages endpoint** - Require API key or JWT token
2. **Validate sessionId** - Ensure user owns the session
3. **Rate limiting** - Prevent abuse of history fetching
4. **PII handling** - Consider chat history may contain sensitive data

---

## Future Enhancements

1. **Caching** - Cache chat history in Redis for faster access
2. **Streaming** - Stream long chat histories to reduce memory
3. **Summarization** - Summarize old messages to reduce context size
4. **Multi-modal** - Include images/files in conversation history
