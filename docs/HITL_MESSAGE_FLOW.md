# Message-Based HITL Flow

## Overview

The message-based HITL (Human-in-the-Loop) system integrates checkpoints directly into the conversation history as regular messages. This creates a natural conversational flow where HITL approvals and error recovery happen through natural language interactions, stored permanently in chat history.

## Key Principles

1. **Conversational Integration**: HITL checkpoints appear as assistant messages in the chat
2. **Natural Language Responses**: Users respond with plain text ("Approve", "Cancel", "Use 1:1", etc.)
3. **Persistent History**: All HITL interactions are stored in the database as part of conversation history
4. **No Polling**: System sends checkpoint message and returns; user responds when ready
5. **Context Detection**: Backend automatically detects when user is responding to a HITL checkpoint

## Architecture

### Components

```
┌─────────────────┐
│ LLM Backend     │
│ (orchestrator)  │
└────────┬────────┘
         │ 1. Send HITL checkpoint via /from-llm
         ▼
┌─────────────────┐
│ CF Workers      │
│ (Core API)      │
├─────────────────┤
│ • Stores message│
│ • Sends WS      │
│   notification  │
└────────┬────────┘
         │ 2. WebSocket notification
         ▼
┌─────────────────┐
│ PWA Client      │
│ (useRagWebSocket)│
├─────────────────┤
│ • Displays as   │
│   regular msg   │
└────────┬────────┘
         │ 3. User responds naturally
         ▼
┌─────────────────┐
│ CF Workers      │
│ /to-llm handler │
├─────────────────┤
│ • Detects HITL  │
│   context       │
│ • Adds metadata │
└────────┬────────┘
         │ 4. Resume workflow
         ▼
┌─────────────────┐
│ LLM Backend     │
│ resume_from_state│
└─────────────────┘
```

## Message Structure

### HITL Checkpoint Message (from LLM Backend)

**Endpoint**: `POST /from-llm`

```json
{
  "sessionId": "session-123",
  "userId": "user-456",
  "sender": "assistant",
  "content": "I need to know the aspect ratio for this image. Available options are: 1:1, 16:9, 9:16, 4:3, 3:2. What would you like?",
  "messageType": "hitl_checkpoint",
  "destination": "user",
  "logId": "run-789",
  "props": {
    "checkpoint_type": "error_recovery",
    "awaiting_response": true,
    "run_id": "run-789",
    "error_type": "validation",
    "error_field": "aspect_ratio",
    "current_value": "invalid",
    "valid_values": ["1:1", "16:9", "9:16", "4:3", "3:2"]
  }
}
```

### User Response Message (to LLM Backend)

**Endpoint**: `POST /to-llm`

The `/to-llm` handler automatically detects HITL context:

```typescript
// Check if user is responding to a HITL checkpoint
const lastAssistantMessage = recentMessages
  .reverse()
  .find((msg: any) => msg.sender === "assistant");

if (lastAssistantMessage?.props?.checkpoint_type) {
  messageProps = {
    in_response_to: lastAssistantMessage.props.checkpoint_type,
    resolves_checkpoint: true,
    checkpoint_run_id: lastAssistantMessage.props.run_id,
    checkpoint_message_id: lastAssistantMessage.id,
  };
}
```

User message is stored with context metadata:

```json
{
  "sessionId": "session-123",
  "content": "Use 16:9",
  "sender": "user",
  "props": {
    "in_response_to": "error_recovery",
    "resolves_checkpoint": true,
    "checkpoint_run_id": "run-789",
    "checkpoint_message_id": 12345
  }
}
```

## Checkpoint Types

### 1. Error Recovery (`error_recovery`)

**When**: API validation errors that can be fixed conversationally

**Example Flow**:
```
Assistant: "The aspect ratio 'invalid' is not supported. Available options are: 1:1, 16:9, 9:16, 4:3, 3:2. What would you like?"

User: "Use 16:9"

System: [Parses response, updates current_values, retries API call]
```

**Implementation**: `orchestrator.py:_handle_validation_error_nl()`

### 2. Information Request (`information_request`)

**When**: Missing required fields for model execution

**Example Flow**:
```
Assistant: "To generate the video, I need to know: What audio file should I use for lip sync?"

User: "Use the audio file I uploaded earlier - the speech.mp3"

System: [Parses response, extracts field values, continues workflow]
```

**Implementation**: `orchestrator.py:_natural_language_information_review()`

### 3. Form Requirements (`form_requirements`)

**When**: Structured form data is required (continues to use WebSocket-based UI)

**Note**: This type still uses WebSocket approval mechanism for structured form input.

## Implementation Details

### Backend (LLM Backend)

#### 1. Send HITL Checkpoint

```python
# In orchestrator.py
from .hitl_message_client import HITLMessageClient

# Initialize client
self.hitl_message_client = HITLMessageClient()

# Send checkpoint as message
checkpoint_data = {
    "run_id": self.run_id,
    "error_type": "validation",
    "error_field": field,
    "current_value": current_value,
    "valid_values": valid_values,
}

await self.hitl_message_client.send_hitl_checkpoint(
    session_id=self.session_id,
    user_id=self.user_id or "unknown",
    content=nl_error_message,
    checkpoint_type="error_recovery",
    checkpoint_data=checkpoint_data
)

# Return pause response - user will respond when ready
return self._create_pause_response(
    step=HITLStep.API_CALL,
    message=nl_error_message,
    actions_required=["respond_naturally"],
    checkpoint_type="error_recovery",
    conversation_mode=True,
    data=checkpoint_data
)
```

#### 2. Resume from State

```python
# In orchestrator.py:resume_from_state()
checkpoint_type = checkpoint_context.get('checkpoint_type')

if checkpoint_type == "error_recovery":
    # Refresh conversation history
    self.state.conversation_history = await self._get_conversation_history()

    # Retry API call with corrected values
    return await self._step_api_execution()

elif checkpoint_type == "information_request":
    # Parse user's natural language response
    conversation_history = await self._get_conversation_history()

    # Extract last user message
    user_message = find_last_user_message(conversation_history)

    # Parse with NL parser
    parsed_values = await parse_natural_language_response(
        user_message=user_message,
        expected_schema=classification,
        current_values=current_values,
        conversation_history=conversation_history
    )

    # Update form data
    self.state.form_data["current_values"].update(parsed_values.extracted_fields)

    # Continue to next step
    return await self._run_pipeline(start_index=resume_index)
```

### CF Workers (Core API)

#### 1. Accept HITL Messages (`/from-llm`)

```typescript
// index.ts
app.post("/from-llm", zValidator("json", fromLLMMessageSchema), async (c) => {
  const data = c.req.valid("json");

  const queueMessage = {
    sessionId: data.sessionId,
    content: data.content,
    messageType: data.messageType,
    props: data.props || {},  // Include HITL metadata
    sender: data.sender,
    userId: data.userId,
  };

  const message = await handleCreateMessage(c, queueMessage);

  // Determine WebSocket action based on message type
  const wsAction = data.messageType === "hitl_checkpoint"
    ? WS_ACTION.HITL_CHECKPOINT
    : WS_ACTION.NEW_MESSAGE;

  await WebSocketService.sendNotification(c.env, {
    sessionId: data.sessionId,
    userId: data.userId,
    status: wsAction === WS_ACTION.HITL_CHECKPOINT ? "awaiting_human" : "processing",
    action: wsAction,
    message,
  });

  return c.json({ success: true, messageId: message.id });
});
```

#### 2. Detect HITL Context (`/to-llm`)

```typescript
// messages.ts
let messageProps = {};
try {
  const recentMessages = await D1Service.getSessionMessages(
    c.env.DB,
    data.sessionId
  );

  const lastAssistantMessage = recentMessages
    .reverse()
    .find((msg: any) => msg.sender === "assistant");

  // If last message was a HITL checkpoint, mark this user response
  if (lastAssistantMessage?.props?.checkpoint_type) {
    messageProps = {
      in_response_to: lastAssistantMessage.props.checkpoint_type,
      resolves_checkpoint: true,
      checkpoint_run_id: lastAssistantMessage.props.run_id,
      checkpoint_message_id: lastAssistantMessage.id,
    };
    console.log(`User responding to HITL checkpoint: ${lastAssistantMessage.props.checkpoint_type}`);
  }
} catch (error) {
  console.error("Error detecting HITL context:", error);
}

const queueMessage = {
  // ...
  props: messageProps,  // Include HITL context if present
  // ...
};
```

### PWA Client

#### 1. Handle HITL_CHECKPOINT Action

```typescript
// useRagWebSocket.ts
if (parsedData.action == DO_ACTION.HITL_CHECKPOINT) {
  const message = parsedData.message;
  if (message) {
    updateSessionProps(parsedData.sessionId);
    addMessage({
      ...message,
      isFresh: true,
    });
  } else {
    reload({ sessionId: parsedData.sessionId })
  }
  setIsThinking(false);
}
```

**Key Point**: HITL checkpoint is treated exactly like `NEW_MESSAGE` - no special UI, just regular message display.

#### 2. User Types Response

User simply types natural response in the chat input:
- "Use 16:9"
- "Approve"
- "Cancel"
- "The audio file is speech.mp3"
- "1:1 aspect ratio please"

The response is sent via normal message flow (`/to-llm`), and the backend detects the HITL context automatically.

## Database Schema

### Messages Table (`props` field)

```sql
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  sender TEXT NOT NULL,
  session_id TEXT NOT NULL,
  user_id TEXT,
  message_type TEXT,
  props TEXT,  -- JSON field for HITL metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Example `props` values**:

```json
// HITL checkpoint message
{
  "checkpoint_type": "error_recovery",
  "awaiting_response": true,
  "run_id": "run-789",
  "error_type": "validation",
  "error_field": "aspect_ratio",
  "valid_values": ["1:1", "16:9", "9:16"]
}

// User response message
{
  "in_response_to": "error_recovery",
  "resolves_checkpoint": true,
  "checkpoint_run_id": "run-789",
  "checkpoint_message_id": 12345
}
```

## Natural Language Parsing

### Error Recovery Parser

```python
# In error_recovery_nl_agent.py
async def parse_error_recovery_response(
    user_message: str,
    error_field: str,
    valid_values: List[str],
    current_value: Any,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Parse user's natural language response to error recovery checkpoint.

    Examples:
    - "Use 16:9" → {"aspect_ratio": "16:9"}
    - "1:1 please" → {"aspect_ratio": "1:1"}
    - "Cancel" → {"action": "cancel"}
    """
```

### Information Request Parser

```python
# In nl_response_parser.py
async def parse_natural_language_response(
    user_message: str,
    expected_schema: Dict[str, Any],
    current_values: Dict[str, Any],
    model_description: str,
    conversation_history: List[Dict[str, str]]
) -> ParsedResponse:
    """
    Parse user's natural language response to information request.

    Uses gpt-4o-mini to extract field values from natural conversation.
    Includes full conversation history for context-aware parsing.
    """
```

## Benefits

### 1. Natural Conversation Flow
- HITL appears as regular chat messages
- Users respond naturally without special UI
- Full conversation context preserved

### 2. Persistent History
- All HITL interactions stored in database
- Can review past approvals/rejections
- Better audit trail

### 3. Better Context
- NL parsers have access to full conversation history
- Can reference previous messages ("the audio file I uploaded earlier")
- More intelligent parsing

### 4. No Resource Waste
- No polling loops waiting for user response
- System pauses cleanly and resumes naturally
- User responds at their own pace

### 5. Multi-Turn Conversations
- Can have back-and-forth clarifications
- Multiple HITL checkpoints in sequence
- Natural follow-up questions

## Migration Notes

### What Changed

**Before** (WebSocket-based):
```python
# Old approach - WebSocket approval with polling
approval_response = await websocket_bridge.request_human_approval(
    run_id=self.run_id,
    checkpoint_type="information_request",
    context=pause_response,
    user_id=user_id,
    session_id=session_id
)

# Wait and poll for response
user_message = approval_response.get("message")
```

**After** (Message-based):
```python
# New approach - Send as message, return, wait naturally
await hitl_message_client.send_hitl_checkpoint(
    session_id=session_id,
    user_id=user_id,
    content=checkpoint_message,
    checkpoint_type="information_request",
    checkpoint_data=data
)

# Return pause response - user will respond when ready
return pause_response

# User response triggers resume_from_state() automatically
```

### What Stayed the Same

- Form-based HITL (`form_requirements`) still uses WebSocket approval mechanism
- WebSocket still used for real-time notifications
- Same state machine and step pipeline
- Same approval/rejection logic

## Testing

### Manual Testing Flow

1. **Start a workflow that requires HITL**:
   ```bash
   curl -X POST http://localhost:8000/hitl/runs \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Generate video with invalid aspect ratio",
       "session_id": "test-session",
       "user_id": "test-user",
       "agent_tool_config": {...}
     }'
   ```

2. **Verify checkpoint appears as message**:
   - Check PWA chat interface
   - Should see assistant message with error explanation
   - No special UI, just regular message bubble

3. **Respond naturally**:
   - Type: "Use 16:9"
   - Send message normally

4. **Verify resume**:
   - Check backend logs for "Resuming from error recovery"
   - Should see NL parsing results
   - Should see API retry with corrected value

### Conversation History Verification

```bash
# Fetch session messages
curl http://localhost:8787/api/sessions/{session_id}/messages

# Should show:
# 1. Initial user request
# 2. HITL checkpoint (props.checkpoint_type = "error_recovery")
# 3. User response (props.in_response_to = "error_recovery")
# 4. Final assistant response
```

## Future Enhancements

1. **Multi-turn clarifications**: Support back-and-forth until fields are complete
2. **Confidence thresholds**: Auto-approve if NL parser confidence > 0.9
3. **Suggested responses**: Show quick reply buttons ("16:9", "9:16", "Cancel")
4. **Rich media**: Support image/video previews in HITL checkpoints
5. **Analytics**: Track HITL checkpoint types, response times, approval rates

## Related Documentation

- `HITL_ORCHESTRATOR.md` - Overall HITL architecture
- `ERROR_RECOVERY.md` - Error recovery system details
- `FORM_BASED_HITL.md` - Form-based workflow
- `NATURAL_LANGUAGE_HITL.md` - NL conversation system
