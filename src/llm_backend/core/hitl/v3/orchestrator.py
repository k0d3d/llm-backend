import time
import asyncio
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..types import HITLState, HITLConfig, HITLStep, HITLStatus
from ..persistence import HITLStateStore
from ..websocket_bridge import WebSocketHITLBridge
from ..hitl_message_client import HITLMessageClient
from ..chat_history_client import ChatHistoryClient
from llm_backend.core.providers.base import AIProvider, ProviderResponse
from llm_backend.core.types.common import RunInput

from .schema_extractor import SchemaExtractor
from .context_assembler import ContextAssembler
from .payload_builder import PayloadBuilder
from .validator import Validator

class HITLOrchestratorV3:
    """
    V3 Orchestrator: Schema-Driven Assembly Line.
    
    Pipeline:
    1. Schema Extraction (Blueprint)
    2. Context Assembly (Bundle)
    3. Payload Construction (Fabrication)
    4. Validation (Guard)
    5. Execution (API)
    """

    def __init__(
        self, 
        provider: AIProvider, 
        config: HITLConfig, 
        run_input: RunInput, 
        state_manager: Optional[HITLStateStore] = None, 
        websocket_bridge: Optional[WebSocketHITLBridge] = None
    ):
        self.provider = provider
        self.provider_name = getattr(provider, "model_name", "unknown")
        self.latest_version = getattr(provider, "latest_version", "")
        self.config = config
        self.run_input = run_input
        self.run_id = str(uuid.uuid4())
        self.state_manager = state_manager
        self.websocket_bridge = websocket_bridge
        
        self.chat_history_client = ChatHistoryClient()
        self.hitl_message_client = HITLMessageClient()
        
        # Metadata
        self.session_id = getattr(run_input, "session_id", None)
        self.user_id = getattr(run_input, "user_id", None)
        self.tenant = getattr(run_input, "tenant", "tohju")

        # Link provider to run context
        self.provider.set_run_input(self.run_input)
        if hasattr(self.provider, 'set_orchestrator'):
            self.provider.set_orchestrator(self)

        # Initialize State
        self.state = HITLState(
            run_id=self.run_id,
            config=config,
            original_input=run_input.model_dump() if hasattr(run_input, "model_dump") else dict(run_input)
        )
        
        # Ensure provider_name is in original_input for persistence fallback
        if isinstance(self.state.original_input, dict):
            self.state.original_input["provider"] = self.provider_name
        
        if config.timeout_seconds > 0:
            self.state.expires_at = datetime.utcnow() + timedelta(seconds=config.timeout_seconds)

        self._add_step_event(HITLStep.CREATED, HITLStatus.QUEUED, "system", "Run created")

    def _add_step_event(self, step: HITLStep, status: HITLStatus, actor: str, message: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a step event to the history"""
        from ..types import StepEvent
        event = StepEvent(
            step=step,
            status=status,
            actor=actor,
            message=message,
            metadata=metadata or {}
        )
        self.state.step_history.append(event)
        self.state.updated_at = datetime.utcnow()
        print(f"📍 Step Event: {step.value} - {status.value} ({actor}): {message}")

    def _transition_to_step(self, step: HITLStep, status: HITLStatus = HITLStatus.RUNNING, message: Optional[str] = None):
        """Transition to a new step"""
        self.state.current_step = step
        self.state.status = status
        self._add_step_event(step, status, "system", message or f"Transitioned to {step.value}")

    async def start_run(self, original_input: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Start the HITL run and persist initial state to database"""
        # Update run input with user/session info
        if user_id:
            self.run_input.user_id = user_id
            self.user_id = user_id
        if session_id:
            self.run_input.session_id = session_id
            self.session_id = session_id

        # Sync state with latest run input values
        updated_input = self.run_input.model_dump() if hasattr(self.run_input, 'model_dump') else dict(self.run_input)
        if original_input:
            updated_input.update(original_input)
        self.state.original_input = updated_input

        if self.state_manager:
            await self.state_manager.save_state(self.state)
        return self.run_id

    @classmethod
    async def resume(
        cls,
        session_id: str,
        provider: AIProvider,
        state_manager: HITLStateStore,
        websocket_bridge: Optional[WebSocketHITLBridge] = None
    ) -> Optional['HITLOrchestratorV3']:
        """Resume an active V3 HITL run"""
        active_run_id = await state_manager.get_active_run_id(session_id)
        if not active_run_id:
            return None

        state = await state_manager.load_state(active_run_id)
        if not state:
            return None

        # Reconstruct RunInput
        try:
            run_input = RunInput(**state.original_input)
        except:
            # Fallback for dict-based state
            from llm_backend.core.providers.base import AttributeDict
            run_input = AttributeDict(state.original_input)

        config = state.config
        orchestrator = cls(provider, config, run_input, state_manager, websocket_bridge)
        orchestrator.state = state
        orchestrator.run_id = state.run_id
        
        return orchestrator

    async def apply_edits(self, edits: Dict[str, Any]) -> None:
        """Apply human edits to state and prepare for retry"""
        if not hasattr(self.state, "human_edits") or self.state.human_edits is None:
            self.state.human_edits = {}
        
        self.state.human_edits.update(edits)
        self.state.status = HITLStatus.RUNNING
        self.state.pending_actions = []
        
        if self.state_manager:
            await self.state_manager.save_state(self.state)

    async def execute(self) -> Dict[str, Any]:
        """Runs the V3 Assembly Line pipeline"""
        print(f"🚀 V3 Orchestrator executing run: {self.run_id}")
        start_time = time.time()
        
        try:
            # 1. BLUEPRINT: Extract Schema (Stripped of demo values)
            self._transition_to_step(HITLStep.FORM_INITIALIZATION)
            tool_config = self._get_replicate_config()
            schema = SchemaExtractor.extract(tool_config)
            print(f"📦 Blueprint: Extracted {len(schema.fields)} fields from schema")
            for f_name, f_def in schema.fields.items():
                print(f"   - {f_name}: type={f_def.type.value}, required={f_def.is_required}, content={f_def.is_content}")

            # 2. BUNDLE: Assemble Context (Prompt + History + Attachments)
            self._transition_to_step(HITLStep.INFORMATION_REVIEW)
            # Ensure history is loaded
            if self.run_input and not getattr(self.run_input, "conversation", None) and self.session_id:
                print(f"📡 Fetching chat history for session {self.session_id}")
                self.run_input.conversation = await self.chat_history_client.get_session_history(self.session_id)
            
            context = ContextAssembler.build(self.run_input, self.state.human_edits)
            print(f"🔗 Bundle: Assembled context with {len(context.attachments)} attachments")
            # print(f"🔍 Context View: {context.get_llm_view()}")

            # 3. FABRICATION: Build Candidate Payload
            self._transition_to_step(HITLStep.PAYLOAD_REVIEW)
            candidate = await PayloadBuilder.build(context, schema)
            self.state.suggested_payload = candidate.parameters
            print(f"🛠️ Fabrication: Candidate parameters: {candidate.parameters}")
            print(f"🛠️ Fabrication: Reasoning: {candidate.reasoning[:200]}...")

            # 4. GUARD: Validate Payload
            issues = Validator.validate(candidate.parameters, schema)
            self.state.validation_issues = [i.model_dump() for i in issues]
            
            # Check for blocking errors
            blocking_issues = [i for i in issues if i.severity == "error"]
            if blocking_issues:
                print(f"⚠️ Guard: Found {len(blocking_issues)} blocking issues: {[i.field for i in blocking_issues]}")
                self.state.status = HITLStatus.AWAITING_HUMAN
                return await self._pause_for_information(blocking_issues)

            # 5. EXECUTION: Call Provider
            self._transition_to_step(HITLStep.API_CALL)
            
            # Update provider with the fabricated payload
            # Wrap the raw dictionary from fabrication into a ReplicatePayload object
            from llm_backend.providers.replicate_provider import ReplicatePayload
            
            replicate_payload = ReplicatePayload(
                provider_name="replicate",
                input=candidate.parameters,
                operation_type=self._infer_operation_type(),
                model_version=self.latest_version
            )
            
            print(f"🚀 Calling provider {self.provider_name} with parameters: {candidate.parameters}")
            response = self.provider.execute(replicate_payload)
            
            self.state.raw_response = response.raw_response
            self.state.processed_response = response.processed_response
            
            # 6. COMPLETION
            self._transition_to_step(HITLStep.COMPLETED, HITLStatus.COMPLETED)
            self.state.final_result = self.provider.audit_response(response)
            
            if self.state_manager and self.session_id:
                await self.state_manager.clear_active_run_id(self.session_id)
            
            return {
                "run_id": self.run_id,
                "status": "completed",
                "result": self.state.final_result
            }

        except Exception as e:
            print(f"❌ V3 Orchestrator Failed: {e}")
            import traceback
            traceback.print_exc()
            self._add_step_event(self.state.current_step, HITLStatus.FAILED, "system", str(e))
            self.state.status = HITLStatus.FAILED
            if self.state_manager:
                await self.state_manager.save_state(self.state)
            raise e
        finally:
            self.state.total_execution_time_ms = int((time.time() - start_time) * 1000)
            if self.state_manager:
                await self.state_manager.save_state(self.state)

    async def _pause_for_information(self, issues: List[Any]) -> Dict[str, Any]:
        """Pauses execution and requests information from the user"""
        self.state.status = HITLStatus.AWAITING_HUMAN
        
        # Build a friendly message
        missing_fields = [i.field for i in issues]
        message = f"I need a bit more information to continue. Please provide: {', '.join(missing_fields)}"
        
        # Send via HITL Message Client (Natural Language)
        if self.config.use_natural_language_hitl:
            await self.hitl_message_client.send_hitl_checkpoint(
                session_id=self.session_id,
                user_id=self.user_id or "unknown",
                content=message,
                checkpoint_type="information_request",
                checkpoint_data={"missing_fields": missing_fields, "run_id": self.run_id},
                tenant=self.tenant
            )
            
        # Return the pause response (standard for API)
        return {
            "run_id": self.run_id,
            "status": "awaiting_human",
            "message": message,
            "missing_fields": missing_fields
        }

    def _get_replicate_config(self) -> Dict[str, Any]:
        """Extracts replicate config from run_input"""
        agent_tool_config = getattr(self.run_input, "agent_tool_config", {})
        # Look for replicate-agent-tool
        config = agent_tool_config.get("replicate-agent-tool") or agent_tool_config.get("replicate_agent_tool")
        if not config:
            # Fallback to provider's internal config if available
            return getattr(self.provider, "config", {})
        return config.get("data") if "data" in config else config

    def _infer_operation_type(self) -> Any:
        """Infers the operation type from description"""
        from llm_backend.core.providers.base import OperationType
        description = (getattr(self.provider, "description", "") or "").lower()
        
        if any(word in description for word in ["image", "picture", "photo"]):
            return OperationType.IMAGE_GENERATION
        elif any(word in description for word in ["video", "movie", "animation"]):
            return OperationType.VIDEO_GENERATION
        elif any(word in description for word in ["audio", "sound", "music"]):
            return OperationType.AUDIO_GENERATION
        
        return OperationType.TEXT_GENERATION
