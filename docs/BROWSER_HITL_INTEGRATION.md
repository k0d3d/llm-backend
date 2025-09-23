# Browser Agent HITL Integration

This document outlines how to integrate the HITL orchestrator with browser agents and WebSocket communication for real-time human feedback during browser automation tasks.

## Overview

Based on your existing browser agent architecture using `browser-use` + Celery + WebSocket server, we can extend the HITL system to:

1. **Pause browser agents** at critical decision points
2. **Send real-time notifications** to browsers via WebSocket
3. **Collect human feedback** through browser UI
4. **Resume agent execution** with human input
5. **Persist state** across browser sessions

## Architecture Integration

```
Browser Agent (Celery) ←→ HITL Orchestrator ←→ WebSocket Server ←→ Browser UI
                                ↓
                        Database/Redis Storage
```

### Key Components

1. **Browser Agent Provider**: Extends `AIProvider` for browser automation
2. **WebSocket HITL Bridge**: Connects orchestrator to WebSocket server
3. **Browser UI Components**: Real-time HITL approval interfaces
4. **State Persistence**: Redis/Database for pause/resume capability

## Browser Agent Provider Implementation

```python
# llm_backend/providers/browser_agent_provider.py
from typing import Dict, Any, List, Optional
from llm_backend.core.providers.base import AIProvider, ProviderPayload, ProviderResponse, ProviderCapabilities, ValidationIssue, OperationType
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge
import asyncio
import json

class BrowserAgentPayload(ProviderPayload):
    task_description: str
    target_url: Optional[str] = None
    max_steps: int = 100
    browser_config: Dict[str, Any] = {}
    checkpoint_conditions: List[str] = []  # When to pause for human input

class BrowserAgentProvider(AIProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get("task_type", "web_automation")
        self.websocket_bridge = WebSocketHITLBridge()
        
    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="Browser Agent",
            description="AI-powered browser automation with human oversight",
            version="1.0",
            input_schema={
                "task_description": "string",
                "target_url": "string?",
                "max_steps": "integer",
                "checkpoint_conditions": "array"
            },
            supported_operations=[OperationType.WEB_AUTOMATION],
            safety_features=["human_checkpoints", "url_validation", "action_logging"],
            rate_limits={"concurrent_sessions": 5}
        )
    
    def create_payload(self, prompt: str, attachments: List[str], operation_type: OperationType, config: Dict) -> BrowserAgentPayload:
        return BrowserAgentPayload(
            provider_name="browser_agent",
            task_description=prompt,
            target_url=attachments[0] if attachments else None,
            max_steps=config.get("max_steps", 100),
            browser_config=config.get("browser_config", {}),
            checkpoint_conditions=config.get("checkpoint_conditions", [
                "before_form_submission",
                "before_data_extraction", 
                "before_navigation_to_new_domain",
                "on_error_or_unexpected_content"
            ])
        )
    
    def validate_payload(self, payload: BrowserAgentPayload, prompt: str, attachments: List[str]) -> List[ValidationIssue]:
        issues = []
        
        if not payload.task_description:
            issues.append(ValidationIssue(
                field="task_description",
                issue="Task description is required",
                severity="error",
                suggested_fix="Provide clear task instructions",
                auto_fixable=False
            ))
        
        if payload.target_url and not payload.target_url.startswith(('http://', 'https://')):
            issues.append(ValidationIssue(
                field="target_url",
                issue="Invalid URL format",
                severity="error", 
                suggested_fix="Use full URL with protocol",
                auto_fixable=True
            ))
        
        return issues
    
    async def execute(self, payload: BrowserAgentPayload) -> ProviderResponse:
        """Execute browser agent with HITL checkpoints"""
        start_time = time.time()
        
        try:
            # Create browser agent task
            from your_browser_tasks import create_browser_agent_task
            
            task_result = await create_browser_agent_task.delay(
                task_description=payload.task_description,
                target_url=payload.target_url,
                max_steps=payload.max_steps,
                checkpoint_conditions=payload.checkpoint_conditions,
                hitl_session_id=self.run_input.session_id,
                run_id=getattr(self, 'run_id', None)
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return ProviderResponse(
                raw_response=task_result,
                processed_response=json.dumps(task_result),
                metadata={
                    "task_type": self.task_type,
                    "steps_executed": task_result.get("steps_executed", 0),
                    "checkpoints_triggered": task_result.get("checkpoints_triggered", 0)
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ProviderResponse(
                raw_response=None,
                processed_response="",
                metadata={},
                execution_time_ms=execution_time,
                error=str(e)
            )
```

## WebSocket HITL Bridge

```python
# llm_backend/core/hitl/websocket_bridge.py
import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional
from llm_backend.core.types.common import RunInput

class WebSocketHITLBridge:
    def __init__(self, websocket_server_url: str = "https://ws.tohju.com"):
        self.websocket_server_url = websocket_server_url
        self.pending_approvals: Dict[str, asyncio.Event] = {}
        self.approval_responses: Dict[str, Dict[str, Any]] = {}
    
    async def request_human_approval(
        self, 
        session_id: str, 
        run_id: str,
        checkpoint_type: str,
        context: Dict[str, Any],
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Request human approval via WebSocket and wait for response"""
        
        approval_id = f"{run_id}_{checkpoint_type}_{int(time.time())}"
        
        # Create event for this approval
        approval_event = asyncio.Event()
        self.pending_approvals[approval_id] = approval_event
        
        # Send approval request via WebSocket server
        await self._send_websocket_message(session_id, {
            "type": "hitl_approval_request",
            "data": {
                "approval_id": approval_id,
                "run_id": run_id,
                "checkpoint_type": checkpoint_type,
                "context": context,
                "actions": ["approve", "reject", "edit"],
                "expires_at": (datetime.utcnow() + timedelta(seconds=timeout_seconds)).isoformat()
            }
        })
        
        try:
            # Wait for human response with timeout
            await asyncio.wait_for(approval_event.wait(), timeout=timeout_seconds)
            response = self.approval_responses.pop(approval_id, {"action": "timeout"})
            return response
            
        except asyncio.TimeoutError:
            return {"action": "timeout", "message": "Human approval timed out"}
        finally:
            # Cleanup
            self.pending_approvals.pop(approval_id, None)
    
    async def handle_approval_response(self, approval_id: str, response: Dict[str, Any]):
        """Handle approval response from browser"""
        self.approval_responses[approval_id] = response
        
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id].set()
    
    async def _send_websocket_message(self, session_id: str, message: Dict[str, Any]):
        """Send message via WebSocket server REST API"""
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    f"{self.websocket_server_url}/update-status",
                    json={
                        "sessionId": session_id,
                        "data": message
                    },
                    headers={"Authorization": f"Bearer {os.getenv('WEBSOCKET_API_KEY')}"}
                )
            except Exception as e:
                print(f"Failed to send WebSocket message: {e}")
```

## Enhanced Browser Agent Tasks

```python
# your_browser_tasks.py (enhanced with HITL)
from celery import Celery
from browser_use import Agent, Controller, ActionResult
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge

@celery.task(name="browser.agent.with.hitl", bind=True)
async def create_browser_agent_task(
    self,
    task_description: str,
    target_url: str,
    max_steps: int,
    checkpoint_conditions: List[str],
    hitl_session_id: str,
    run_id: str
):
    websocket_bridge = WebSocketHITLBridge()
    
    # Enhanced controller with HITL checkpoints
    controller = Controller()
    
    @controller.registry.action('Request Human Approval')
    async def request_approval(checkpoint_type: str, context: Dict[str, Any]):
        """Pause execution and request human approval"""
        
        approval_response = await websocket_bridge.request_human_approval(
            session_id=hitl_session_id,
            run_id=run_id,
            checkpoint_type=checkpoint_type,
            context=context
        )
        
        if approval_response["action"] == "approve":
            return ActionResult(extracted_content="approved")
        elif approval_response["action"] == "edit":
            # Apply edits from human
            edits = approval_response.get("edits", {})
            return ActionResult(extracted_content=f"approved_with_edits:{json.dumps(edits)}")
        else:
            # Reject or timeout
            return ActionResult(
                is_done=True, 
                extracted_content=f"rejected:{approval_response.get('reason', 'Unknown')}"
            )
    
    @controller.registry.action('Check Checkpoint Condition')
    async def check_checkpoint(current_action: str, page_context: Dict[str, Any]):
        """Check if current action requires human approval"""
        
        for condition in checkpoint_conditions:
            if condition == "before_form_submission" and "submit" in current_action.lower():
                return await request_approval("form_submission", {
                    "action": current_action,
                    "form_data": page_context.get("form_data", {}),
                    "target_url": page_context.get("url", "")
                })
            
            elif condition == "before_data_extraction" and "extract" in current_action.lower():
                return await request_approval("data_extraction", {
                    "action": current_action,
                    "extraction_target": page_context.get("extraction_target", ""),
                    "page_content_preview": page_context.get("content_preview", "")
                })
            
            elif condition == "before_navigation_to_new_domain":
                current_domain = page_context.get("current_domain", "")
                target_domain = page_context.get("target_domain", "")
                if current_domain != target_domain:
                    return await request_approval("domain_navigation", {
                        "from_domain": current_domain,
                        "to_domain": target_domain,
                        "action": current_action
                    })
        
        return ActionResult(extracted_content="no_checkpoint_required")
    
    # Create agent with HITL-enhanced controller
    agent = Agent(
        task=task_description,
        llm=ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
        controller=controller,
        browser=init_browser()
    )
    
    try:
        history = await agent.run(max_steps=max_steps)
        result = history.final_result()
        
        return {
            "status": "completed",
            "result": result,
            "steps_executed": len(history.history),
            "checkpoints_triggered": len([h for h in history.history if "request_approval" in str(h)])
        }
        
    except Exception as e:
        # Send error notification via WebSocket
        await websocket_bridge._send_websocket_message(hitl_session_id, {
            "type": "browser_agent_error",
            "data": {
                "run_id": run_id,
                "error": str(e),
                "task_description": task_description
            }
        })
        raise
```

## Browser UI Components

```typescript
// Browser-side HITL approval component
interface HITLApprovalRequest {
  approval_id: string;
  run_id: string;
  checkpoint_type: string;
  context: any;
  actions: string[];
  expires_at: string;
}

const HITLApprovalModal: React.FC<{request: HITLApprovalRequest}> = ({ request }) => {
  const [response, setResponse] = useState<string>('');
  const [edits, setEdits] = useState<any>({});
  
  const handleApproval = async (action: string) => {
    // Send approval response back to HITL system
    await fetch('/api/hitl/approval-response', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        approval_id: request.approval_id,
        action,
        edits: action === 'edit' ? edits : undefined,
        response_message: response
      })
    });
  };
  
  return (
    <div className="hitl-approval-modal">
      <h3>Human Approval Required</h3>
      <p><strong>Checkpoint:</strong> {request.checkpoint_type}</p>
      <p><strong>Context:</strong></p>
      <pre>{JSON.stringify(request.context, null, 2)}</pre>
      
      {request.checkpoint_type === 'form_submission' && (
        <div>
          <h4>Form Data to Submit:</h4>
          <pre>{JSON.stringify(request.context.form_data, null, 2)}</pre>
        </div>
      )}
      
      <div className="approval-actions">
        <button onClick={() => handleApproval('approve')}>
          Approve
        </button>
        <button onClick={() => handleApproval('reject')}>
          Reject
        </button>
        <button onClick={() => handleApproval('edit')}>
          Edit & Approve
        </button>
      </div>
      
      {/* Edit interface for modifications */}
      <textarea 
        value={response}
        onChange={(e) => setResponse(e.target.value)}
        placeholder="Add notes or modifications..."
      />
    </div>
  );
};
```

## State Persistence & Resume Capability

Yes, HITL sessions can absolutely be paused and resumed using the `run_id`. Here's the storage strategy:

### Database Schema

```sql
-- PostgreSQL schema for HITL state persistence
CREATE TABLE hitl_runs (
    run_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    current_step VARCHAR(50) NOT NULL,
    provider_name VARCHAR(100) NOT NULL,
    original_input JSONB NOT NULL,
    hitl_config JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Step artifacts
    capabilities JSONB,
    suggested_payload JSONB,
    validation_issues JSONB,
    raw_response JSONB,
    processed_response TEXT,
    final_result TEXT,
    
    -- Human interactions
    pending_actions TEXT[],
    approval_token VARCHAR(255),
    
    -- Metrics
    total_execution_time_ms INTEGER DEFAULT 0,
    human_review_time_ms INTEGER DEFAULT 0,
    provider_execution_time_ms INTEGER DEFAULT 0
);

CREATE TABLE hitl_step_events (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES hitl_runs(run_id),
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    actor VARCHAR(255) NOT NULL,
    message TEXT,
    metadata JSONB
);

CREATE TABLE hitl_approvals (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES hitl_runs(run_id),
    approval_id VARCHAR(255) UNIQUE NOT NULL,
    checkpoint_type VARCHAR(100) NOT NULL,
    context JSONB NOT NULL,
    response JSONB,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Indexes
CREATE INDEX idx_hitl_runs_status ON hitl_runs(status);
CREATE INDEX idx_hitl_runs_created_at ON hitl_runs(created_at);
CREATE INDEX idx_hitl_step_events_run_id ON hitl_step_events(run_id);
CREATE INDEX idx_hitl_approvals_approval_id ON hitl_approvals(approval_id);
```

### Redis for Real-time State

```python
# llm_backend/core/hitl/state_manager.py
import redis
import json
from typing import Optional, Dict, Any

class HITLStateManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.db_session = get_db_session()  # Your DB session
    
    async def save_run_state(self, run_state: HITLState):
        """Save state to both Redis (fast access) and DB (persistence)"""
        
        # Redis for fast access
        await self.redis.setex(
            f"hitl:run:{run_state.run_id}",
            3600,  # 1 hour TTL
            run_state.json()
        )
        
        # Database for persistence
        await self._save_to_database(run_state)
    
    async def load_run_state(self, run_id: str) -> Optional[HITLState]:
        """Load state from Redis first, fallback to DB"""
        
        # Try Redis first
        cached_state = await self.redis.get(f"hitl:run:{run_id}")
        if cached_state:
            return HITLState.parse_raw(cached_state)
        
        # Fallback to database
        return await self._load_from_database(run_id)
    
    async def pause_run(self, run_id: str, checkpoint_type: str, context: Dict[str, Any]):
        """Pause run and save checkpoint context"""
        
        state = await self.load_run_state(run_id)
        if state:
            state.status = HITLStatus.AWAITING_HUMAN
            state.current_checkpoint = checkpoint_type
            state.checkpoint_context = context
            await self.save_run_state(state)
    
    async def resume_run(self, run_id: str, approval_response: Dict[str, Any]):
        """Resume run with human approval/edits"""
        
        state = await self.load_run_state(run_id)
        if state:
            state.status = HITLStatus.RUNNING
            state.last_approval = approval_response
            await self.save_run_state(state)
            
            # Trigger run continuation
            await self._continue_run(state)
```

## Integration with Existing WebSocket Server

Update your WebSocket server to handle HITL messages:

```typescript
// websocket-server enhancement for HITL
interface HITLMessage {
  type: 'hitl_approval_request' | 'hitl_approval_response' | 'browser_agent_error';
  data: any;
}

// In your WebSocket server
ws.on('message', async (message) => {
  const parsed = JSON.parse(message);
  
  if (parsed.type === 'hitl_approval_response') {
    // Forward approval response to HITL system
    await fetch(`${HITL_API_URL}/hitl/approval-response`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parsed.data)
    });
  }
});

// REST endpoint for HITL notifications
app.post('/hitl-notify', (req, res) => {
  const { sessionId, data } = req.body;
  
  // Send to connected WebSocket clients for this session
  const clients = getClientsForSession(sessionId);
  clients.forEach(client => {
    client.send(JSON.stringify({
      type: 'hitl_notification',
      data
    }));
  });
  
  res.json({ success: true });
});
```

## Benefits of This Integration

1. **Real-time Feedback**: Instant notifications to browser when human input needed
2. **Persistent State**: Runs can be paused/resumed across browser sessions
3. **Rich Context**: Full browser state and screenshots available for human review
4. **Flexible Checkpoints**: Configurable conditions for when to pause
5. **Audit Trail**: Complete history of human decisions and agent actions
6. **Error Recovery**: Human can intervene when browser agents encounter issues

This creates a seamless integration between your existing browser automation infrastructure and the new HITL system, enabling sophisticated human-AI collaboration workflows.
