"""
HITL Orchestrator - Main workflow coordinator
"""

import time
import asyncio
import uuid
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List

from .types import HITLState, StepEvent, HITLConfig, HITLStep, HITLStatus, HITLPolicy
from .websocket_bridge import WebSocketHITLBridge
from .validation import HITLValidator, create_hitl_validation_summary
from .chat_history_client import ChatHistoryClient
from .hitl_message_client import HITLMessageClient
from llm_backend.core.providers.base import AIProvider, ProviderResponse, AttributeDict
from llm_backend.core.types.common import RunInput
from llm_backend.agents.error_recovery_nl_agent import generate_error_recovery_message


class HITLOrchestrator:
    """Main orchestrator for HITL workflow management"""
    
    def __init__(self, provider: AIProvider, config: HITLConfig, run_input: RunInput, state_manager=None, websocket_bridge=None):
        self.provider = provider
        self.config = config
        self.run_input = self._normalize_run_input(run_input)
        self.run_id = str(uuid.uuid4())
        self.state_manager = state_manager
        self.websocket_bridge = websocket_bridge

        # Initialize chat history client for error recovery
        self.chat_history_client = ChatHistoryClient()

        # Initialize HITL message client for sending checkpoints via /from-llm
        self.hitl_message_client = HITLMessageClient()

        # Extract session_id and user_id for messaging
        self.session_id = None
        self.user_id = None
        if hasattr(self.run_input, 'session_id'):
            self.session_id = self.run_input.session_id
        elif isinstance(self.run_input, dict):
            self.session_id = self.run_input.get('session_id')

        if hasattr(self.run_input, 'user_id'):
            self.user_id = self.run_input.user_id
        elif isinstance(self.run_input, dict):
            self.user_id = self.run_input.get('user_id')

        # Debug logging for persistence components
        print("🔧 HITLOrchestrator initialized:")
        print(f"   - run_id: {self.run_id}")
        print(f"   - session_id: {self.session_id}")
        print(f"   - state_manager: {'✅ Available' if state_manager else '❌ None'}")
        print(f"   - websocket_bridge: {'✅ Available' if websocket_bridge else '❌ None'}")
        print(f"   - provider: {provider.__class__.__name__ if provider else 'None'}")

        # Set provider run input
        self.provider.set_run_input(self.run_input)

        # Link orchestrator to provider for form-based workflow
        if hasattr(self.provider, 'set_orchestrator'):
            self.provider.set_orchestrator(self)

        # Initialize state
        self.state = HITLState(
            run_id=self.run_id,
            config=config,
            original_input=self.run_input.model_dump() if hasattr(self.run_input, 'model_dump') else dict(self.run_input)
        )

        # Track human edits across checkpoints
        if not hasattr(self.state, "human_edits") or self.state.human_edits is None:
            self.state.human_edits = {}

        # Set expiration if configured
        if config.timeout_seconds > 0:
            self.state.expires_at = datetime.utcnow() + timedelta(seconds=config.timeout_seconds)

        self._add_step_event(HITLStep.CREATED, HITLStatus.QUEUED, "system", "Run created")
    
    def _normalize_run_input(self, run_input):
        """Ensure run_input is a structured object with attribute access."""
        if run_input is None:
            return AttributeDict()

        if hasattr(run_input, "model_dump"):
            return run_input

        if isinstance(run_input, dict):
            try:
                return RunInput(**run_input)
            except Exception:
                return AttributeDict(run_input)

        if isinstance(run_input, AttributeDict):
            return run_input

        # Fallback: wrap objects lacking attribute access but convertible to dict
        try:
            return AttributeDict(dict(run_input))
        except Exception:
            return AttributeDict()

    def _collect_hitl_edits(self) -> Optional[Dict[str, Any]]:
        """Collect all human edits from state"""
        # Check multiple sources for human edits
        human_edits = {}
        
        # Source 1: Direct state.human_edits
        if hasattr(self.state, 'human_edits') and self.state.human_edits:
            human_edits.update(self.state.human_edits)
            print(f"🔍 Found human edits in state.human_edits: {self.state.human_edits}")
        
        # Source 2: Last approval response
        if hasattr(self.state, 'last_approval') and self.state.last_approval and self.state.last_approval.get('edits'):
            approval_edits = self.state.last_approval['edits']
            human_edits.update(approval_edits)
            print(f"🔍 Found human edits in last_approval: {approval_edits}")
        
        # Source 3: Suggested payload for backward compatibility
        if not human_edits and hasattr(self.state, 'suggested_payload') and isinstance(self.state.suggested_payload, dict):
            suggested = self.state.suggested_payload
            # Check top-level fields first
            for field in ["input_image", "source_image", "driven_audio", "audio_file"]:
                if field in suggested:
                    human_edits[field] = suggested[field]
                    print(f"🔍 Found human edit in suggested_payload top-level: {field} = {suggested[field]}")
            
            # Then check nested input
            if 'input' in suggested and isinstance(suggested['input'], dict):
                for field in ["input_image", "source_image", "driven_audio", "audio_file", "image"]:
                    if field in suggested['input']:
                        # Map back to original field name for consistency
                        if field == "image":
                            human_edits["input_image"] = suggested['input'][field]
                            print(f"🔍 Found human edit in suggested_payload.input: image -> input_image = {suggested['input'][field]}")
                        else:
                            human_edits[field] = suggested['input'][field]
                            print(f"🔍 Found human edit in suggested_payload.input: {field} = {suggested['input'][field]}")
        
        if not human_edits:
            # print(f"🔍 No human edits found anywhere. state.human_edits: {getattr(self.state, 'human_edits', 'MISSING')}, last_approval: {getattr(self.state, 'last_approval', 'MISSING')}")
            return None
        
        print(f"🔍 Collected human edits: {human_edits}")
        return human_edits

    def _add_to_conversation(self, role: str, message: str, metadata: Optional[Dict] = None):
        """
        Add a message to the conversation history.

        Args:
            role: "user", "assistant", or "system"
            message: The message content
            metadata: Optional metadata (e.g., step, checkpoint_type, field)
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

    async def _get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get full conversation history including:
        1. Database chat history (from D1 via CF Workers)
        2. Current HITL conversation (from state)

        Returns combined and sorted conversation.
        """
        # Fetch from database
        db_history = []
        if self.session_id:
            try:
                db_history = await self.chat_history_client.get_session_history(
                    self.session_id
                )
                print(f"📚 Fetched {len(db_history)} messages from database for session {self.session_id}")
            except Exception as e:
                print(f"⚠️ Failed to fetch chat history: {e}")
                db_history = []
        else:
            print("ℹ️ No session_id available, skipping database chat history fetch")

        # Get HITL conversation from state
        hitl_history = getattr(self.state, 'conversation_history', [])
        print(f"💬 HITL conversation has {len(hitl_history)} messages")

        # Combine (database history is older, HITL history is recent)
        combined = db_history + hitl_history

        # Sort by timestamp
        combined.sort(key=lambda x: x.get('timestamp', ''))

        return combined

    def _apply_form_submission(self, approval_response: Optional[Dict[str, Any]]) -> None:
        """Handle form field submissions from user"""
        if not approval_response:
            return

        print(f"📝 Applying form submission")

        edits = approval_response.get("edits", {})
        if not edits:
            print("⚠️ No edits in form submission")
            return

        # Ensure form_data exists
        if not self.state.form_data:
            print("⚠️ No form_data in state, initializing")
            self.state.form_data = {
                "schema": {},
                "classification": {},
                "defaults": {},
                "current_values": {},
                "user_edits": {}
            }

        classification = self.state.form_data.get("classification", {})
        field_classifications = classification.get("field_classifications", {})

        for field, value in edits.items():
            # Store user edit
            self.state.form_data["user_edits"][field] = value
            print(f"📝 User provided: {field} = {value}")

            # Get field classification
            field_class = field_classifications.get(field, {})

            # Handle both object and dict forms
            is_collection = getattr(field_class, 'collection', None) if hasattr(field_class, 'collection') else field_class.get("collection", False)

            if is_collection:
                # For arrays, handle appending/replacing
                current = self.state.form_data["current_values"].get(field, [])
                if isinstance(value, list):
                    # Replace entire array
                    self.state.form_data["current_values"][field] = value
                    print(f"   ✅ Set array field '{field}' with {len(value)} items")
                else:
                    # Append single item
                    if not isinstance(current, list):
                        current = []
                    current.append(value)
                    self.state.form_data["current_values"][field] = current
                    print(f"   ✅ Added to array field '{field}' (now {len(current)} items)")
            else:
                # For non-arrays, direct assignment
                self.state.form_data["current_values"][field] = value
                print(f"   ✅ Set field '{field}' = {value}")

        # Preserve approval metadata
        self.state.last_approval = approval_response

        # Update status
        self.state.pending_actions = []
        self.state.status = HITLStatus.RUNNING
        self.state.updated_at = datetime.utcnow()

        # Persist updated state
        if self.state_manager:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.state_manager.save_state(self.state))
                else:
                    loop.run_until_complete(self.state_manager.save_state(self.state))
            except RuntimeError:
                asyncio.run(self.state_manager.save_state(self.state))

    def _apply_approval_response(self, approval_response: Optional[Dict[str, Any]], step: Optional[HITLStep] = None) -> None:
        """Persist approval details and merge edits into orchestrator state."""
        if not approval_response:
            return

        print(f"🔄 Applying approval response: {approval_response.get('action')} for step {step}")

        # Preserve raw approval for downstream consumers
        self.state.last_approval = approval_response

        edits = approval_response.get("edits") or {}
        if not hasattr(self.state, "human_edits") or self.state.human_edits is None:
            self.state.human_edits = {}

        for field, value in edits.items():
            self.state.human_edits[field] = value
            print(f"🧩 Stored human edit from approval: {field} = {value}")

            if field == "prompt":
                self.run_input.prompt = value
            elif field == "model_config" and isinstance(value, dict):
                if self.run_input.agent_tool_config is None:
                    self.run_input.agent_tool_config = {}
                self.run_input.agent_tool_config.update(value)

        if step:
            self.state.current_step = step

        # Clear pending actions and resume execution state
        self.state.pending_actions = []
        self.state.status = HITLStatus.RUNNING
        self.state.updated_at = datetime.utcnow()

        # Persist updated state asynchronously if a manager is available
        if self.state_manager:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.state_manager.save_state(self.state))
                else:
                    loop.run_until_complete(self.state_manager.save_state(self.state))
            except RuntimeError:
                # get_event_loop may fail outside async context; fall back to direct call
                asyncio.run(self.state_manager.save_state(self.state))

    def _merge_payload_edits(self, payload_dict: Dict[str, Any], hitl_edits: Dict[str, Any]) -> Dict[str, Any]:
        """Merge HITL edits into payload dictionary"""
        merged = payload_dict.copy()

        # Ensure input section exists
        if 'input' not in merged:
            merged['input'] = {}
        
        # Apply edits with smart field mapping
        for key, value in hitl_edits.items():
            # Map common field aliases
            if key in ["input_image", "source_image"] and "image" in merged['input']:
                merged['input']['image'] = value
            elif key in merged['input']:
                merged['input'][key] = value
            else:
                # Store as-is if no mapping found
                merged['input'][key] = value
        
        return merged
    
    async def start_run(self, original_input: Dict[str, Any], user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Start the HITL run and persist initial state to database"""
        print(f"🚀 Starting HITL run with user_id={user_id}, session_id={session_id}")
        
        # Update run input with user/session info
        if user_id:
            self.run_input.user_id = user_id
        if session_id:
            self.run_input.session_id = session_id

        # Sync state with latest run input values before persistence
        updated_input = self.run_input.model_dump() if hasattr(self.run_input, 'model_dump') else self.run_input
        if original_input:
            updated_input.update(original_input)
        self.state.original_input = updated_input

        # Save initial state to database
        if self.state_manager:
            try:
                print("💾 Attempting to save initial state to database...")
                await self.state_manager.save_state(self.state)
                print(f"✅ Initial HITL state saved to database for run_id: {self.run_id}")
            except Exception as e:
                print(f"❌ Failed to save initial HITL state: {e}")
                import traceback
                print(f"📋 Traceback: {traceback.format_exc()}")
        else:
            print("⚠️ No state_manager available - initial state NOT saved to database")
        
        return self.run_id
    
    async def execute(self) -> Dict[str, Any]:
        """Main execution flow with HITL checkpoints"""
        print(f"🚀 HITL Orchestrator starting execution for run_id: {self.run_id}")

        # Fetch and populate conversation history early so it's available for payload creation
        if not self.run_input.conversation and self.session_id:
            try:
                print(f"📚 Fetching conversation history for session {self.session_id}")
                conversation_history = await self._get_conversation_history()
                if conversation_history:
                    self.run_input.conversation = conversation_history
                    print(f"✅ Populated run_input.conversation with {len(conversation_history)} messages")
            except Exception as e:
                print(f"⚠️ Failed to fetch conversation history during initialization: {e}")

        start_time = time.time()

        try:
            return await self._run_pipeline(start_index=0)
        except Exception as e:
            return await self._handle_error(e)
        finally:
            self.state.total_execution_time_ms = int((time.time() - start_time) * 1000)

    async def resume_from_state(self) -> Dict[str, Any]:
        """Resume execution from the current step recorded in state."""
        print(f"🔁 Resuming HITL run {self.run_id} from step {self.state.current_step}")
        start_time = time.time()

        try:
            # Check if resuming from error recovery or information request
            checkpoint_context = getattr(self.state, 'checkpoint_context', {})
            checkpoint_type = checkpoint_context.get('checkpoint_type')

            if checkpoint_type == "error_recovery":
                print("🔧 Resuming from error recovery - user provided fix")

                # Refresh conversation history from database
                self.state.conversation_history = await self._get_conversation_history()

                # The user's response has been parsed and current_values updated
                # Now retry the API call with corrected values
                print(f"🔄 Retrying API call with corrected values...")

                # Re-execute API call step
                return await self._step_api_execution()

            elif checkpoint_type == "information_request":
                print("🔧 Resuming from information request - parsing user's NL response")

                # Refresh conversation history from database
                conversation_history = await self._get_conversation_history()

                # Get the user's latest message (their response to our information request)
                if conversation_history and len(conversation_history) > 0:
                    # Find the last user message
                    user_message = None
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "user":
                            user_message = msg.get("content")
                            break

                    if user_message:
                        from llm_backend.agents.nl_response_parser import parse_natural_language_response

                        # Get current form state
                        classification = self.state.form_data.get("classification", {})
                        current_values = self.state.form_data.get("current_values", {})
                        model_description = getattr(self.provider, 'description', '')

                        # Parse the user's natural language response
                        print(f"🧠 Parsing user message: '{user_message}'")
                        parsed_values = await parse_natural_language_response(
                            user_message=user_message,
                            expected_schema=classification,
                            current_values=current_values,
                            model_description=model_description,
                            conversation_history=conversation_history
                        )

                        print(f"📊 Parsing results:")
                        print(f"   Confidence: {parsed_values.confidence}")
                        print(f"   Extracted fields: {list(parsed_values.extracted_fields.keys())}")

                        # Merge parsed values into form data
                        self.state.form_data["current_values"].update(parsed_values.extracted_fields)
                        print(f"✅ Updated current_values with parsed fields")

                        # Log conversation for debugging
                        self._add_step_event(
                            HITLStep.INFORMATION_REVIEW,
                            HITLStatus.COMPLETED,
                            "human",
                            f"Natural language response: '{user_message}' (confidence: {parsed_values.confidence})",
                            metadata={
                                "parsed_fields": parsed_values.extracted_fields,
                                "confidence": parsed_values.confidence,
                                "ambiguities": parsed_values.ambiguities
                            }
                        )

                # Continue to next step
                resume_index = self._determine_resume_index()
                return await self._run_pipeline(start_index=resume_index)

            # Normal resume logic
            resume_index = self._determine_resume_index()
            return await self._run_pipeline(start_index=resume_index)
        except Exception as e:
            return await self._handle_error(e)
        finally:
            self.state.total_execution_time_ms = int((time.time() - start_time) * 1000)

    def _get_step_pipeline(self) -> List[Callable[[], Dict[str, Any]]]:
        return [
            (HITLStep.FORM_INITIALIZATION, self._step_form_initialization),
            (HITLStep.INFORMATION_REVIEW, self._step_information_review),
            (HITLStep.PAYLOAD_REVIEW, self._step_payload_review),
            (HITLStep.API_CALL, self._step_api_execution),
            (HITLStep.RESPONSE_REVIEW, self._step_response_review),
        ]

    def _determine_resume_index(self) -> int:
        pipeline = self._get_step_pipeline()
        current_step = self.state.current_step or HITLStep.INFORMATION_REVIEW

        for idx, (step, _) in enumerate(pipeline):
            if step == current_step:
                # Move to next step after the one we paused at
                return min(idx + 1, len(pipeline))

        return 0

    async def _run_pipeline(self, start_index: int = 0) -> Dict[str, Any]:
        pipeline = self._get_step_pipeline()

        result: Dict[str, Any] = {"continue": True}
        user_id = getattr(self.run_input, 'user_id', None)
        session_id = getattr(self.run_input, 'session_id', None)

        for idx in range(start_index, len(pipeline)):
            step, handler = pipeline[idx]

            # Signal processing start
            if self.websocket_bridge:
                await self.websocket_bridge.send_thinking_status(
                    message=f"Processing {step.value}",
                    user_id=user_id,
                    session_id=session_id
                )

            try:
                result = await handler()
            except Exception:
                if self.websocket_bridge:
                    await self.websocket_bridge.send_done_thinking(
                        user_id=user_id,
                        session_id=session_id
                    )
                raise
            else:
                if self.websocket_bridge:
                    should_clear = self._is_paused(result) or step != HITLStep.API_CALL
                    if should_clear:
                        await self.websocket_bridge.send_done_thinking(
                            user_id=user_id,
                            session_id=session_id
                        )

            if self._is_paused(result):
                print(f"⏸️ Execution paused at {step.value}")
                return result

        return await self._step_completion()
    
    async def _send_websocket_message(self, message: Dict[str, Any]) -> None:
        """Send HITL message via WebSocket bridge (non-blocking notification)."""
        try:
            print("📤 Preparing WebSocket HITL notification: type=hitl_approval_request")
            
            # Extract session info from run_input
            session_id = getattr(self.run_input, 'session_id', None)
            user_id = getattr(self.run_input, 'user_id', None)
            
            if not session_id:
                print("⚠️ No session_id found for WebSocket notification")
                return
            
            # Use shared bridge if available, otherwise fallback to creating one
            if self.websocket_bridge:
                bridge = self.websocket_bridge
                print("🔧 Using shared WebSocket bridge")
            else:
                # Fallback: Configure bridge from environment
                ws_url = os.getenv("WEBSOCKET_URL", "wss://ws.tohju.com")
                ws_key = os.getenv("WEBSOCKET_API_KEY")
                print(f"🔧 WebSocket config: url={ws_url}, api_key_present={bool(ws_key)}")
                bridge = WebSocketHITLBridge(websocket_url=ws_url, websocket_api_key=ws_key)
            
            # Non-blocking: directly send the approval request envelope
            envelope = {
                "type": "hitl_approval_request",
                "data": {
                    "run_id": self.run_id,
                    "checkpoint_type": str(self.state.current_step.value if hasattr(self.state.current_step, 'value') else self.state.current_step),
                    "context": message,
                    "user_id": user_id,
                    "session_id": session_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            print(f"📤 Sending WebSocket envelope to session={session_id}, user={user_id}")
            await bridge._send_websocket_message(envelope, user_id=user_id, session_id=session_id)
            print(f"✅ WebSocket message sent successfully to session {session_id}")
            
        except Exception as e:
            print(f"❌ WebSocket notification failed: {e}")
            # Don't fail the entire workflow if WebSocket fails

    async def _step_form_initialization(self) -> Dict[str, Any]:
        """Form initialization step - classify and reset fields from example_input"""
        self._transition_to_step(HITLStep.FORM_INITIALIZATION)

        print("📋 Starting form initialization from example_input")

        # Get example_input from provider
        example_input = self.provider.example_input if hasattr(self.provider, 'example_input') else {}
        model_name = self.provider.model_name if hasattr(self.provider, 'model_name') else ""
        description = self.provider.description if hasattr(self.provider, 'description') else ""
        field_metadata = self.provider.field_metadata if hasattr(self.provider, 'field_metadata') else None

        # DEBUG LOGGING
        print(f"🔍 DEBUG: Provider type: {type(self.provider).__name__}")
        print(f"🔍 DEBUG: Provider config keys: {list(self.provider.config.keys()) if hasattr(self.provider, 'config') else 'N/A'}")
        print(f"🔍 DEBUG: example_input from provider: {example_input}")
        print(f"🔍 DEBUG: model_name: {model_name}")
        print(f"🔍 DEBUG: Has example_input: {bool(example_input)}")

        if not example_input:
            print("⚠️ No example_input found, skipping form initialization")
            print(f"⚠️ Provider attributes available: {[attr for attr in dir(self.provider) if not attr.startswith('_') and not callable(getattr(self.provider, attr))]}")
            self._add_step_event(HITLStep.FORM_INITIALIZATION, HITLStatus.COMPLETED, "system", "No example_input - skipped")
            return {"continue": True}

        # NEW: Extract user-provided values from run_input
        user_prompt_raw = getattr(self.run_input, 'prompt', '')

        # Extract URLs from prompt and clean the prompt text
        user_prompt_cleaned, prompt_urls = self._extract_and_clean_urls_from_prompt(user_prompt_raw)

        # Gather explicit attachments (not from prompt)
        explicit_attachments = self._gather_user_supplied_attachments()

        # Combine explicit attachments with URLs extracted from prompt
        user_attachments = list(explicit_attachments) + prompt_urls

        print(f"🔍 User provided: prompt='{user_prompt_cleaned[:50]}...' ({len(user_prompt_cleaned)} chars), attachments={len(user_attachments)} items")

        # Call AI agent to classify fields
        from llm_backend.agents.form_field_classifier import classify_form_fields

        try:
            classification = await classify_form_fields(
                example_input=example_input,
                model_name=model_name,
                model_description=description,
                field_metadata=field_metadata
            )
            print(f"✅ AI classification complete: {len(classification.field_classifications)} fields classified")
            print(f"   Required fields: {classification.required_fields}")
            print(f"   Optional fields: {classification.optional_fields}")
        except Exception as e:
            print(f"❌ Form classification failed: {e}")
            # This will fail the run as we can't proceed without classification
            raise Exception(f"Failed to classify form fields: {e}")

        # NEW: Map user values to field names
        user_provided_values = {}

        # 1. Map prompt field (use cleaned prompt without URLs)
        if user_prompt_cleaned:
            user_provided_values['prompt'] = user_prompt_cleaned
            print(f"🔍 DEBUG: Mapped user prompt to 'prompt' field: {user_prompt_cleaned[:100]}")

        # 2. Extract field values from conversation history
        if hasattr(self.run_input, 'conversation') and self.run_input.conversation:
            field_names = list(example_input.keys())  # Get all possible field names
            historical_values = self._extract_field_values_from_conversation(field_names)
            if historical_values:
                print(f"📚 Found {len(historical_values)} field values in conversation history")
                # Add to user_provided_values (current prompt takes precedence)
                for field, value in historical_values.items():
                    if field not in user_provided_values:
                        user_provided_values[field] = value
                        print(f"   📝 From history: {field}={value}")

        # 3. Map attachments to appropriate fields
        if user_attachments:
            attachment_mapping = await self._map_attachments_to_fields(
                user_attachments,
                classification.field_classifications
            )
            user_provided_values.update(attachment_mapping)
            print(f"📎 Mapped {len(user_attachments)} attachments to fields: {list(attachment_mapping.keys())}")

        print(f"🔍 DEBUG: Total user_provided_values: {list(user_provided_values.keys())}")
        print(f"🔍 DEBUG: Values: {user_provided_values}")

        # Build form with reset logic applied AND user values
        form_data = self._build_form_from_classification(
            example_input,
            classification,
            user_provided_values
        )

        # Extract defaults (values that were NOT reset)
        defaults = self._extract_defaults_from_classification(example_input, classification)

        # Store in state
        self.state.form_data = {
            "schema": example_input,
            "classification": classification.model_dump(),
            "defaults": defaults,
            "current_values": form_data,
            "user_edits": {}
        }

        print(f"📋 Form initialized with {len(form_data)} fields")
        print(f"   Reset fields: {[k for k, v in form_data.items() if v in (None, '', [])]}")
        print(f"   Fields with defaults: {list(defaults.keys())}")

        self._add_step_event(HITLStep.FORM_INITIALIZATION, HITLStatus.COMPLETED, "system", "Form initialized from example_input")
        return {"continue": True}

    def _build_form_from_classification(
        self,
        example_input: Dict[str, Any],
        classification: Any,
        user_provided_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build form by applying reset logic based on AI classification
        Pre-populates with user-provided values if available
        Handles nested objects recursively
        """
        form = {}
        field_classifications = classification.field_classifications
        user_values = user_provided_values or {}

        for field, value in example_input.items():
            if field not in field_classifications:
                # Field not classified, keep as-is
                form[field] = value
                continue

            field_class = field_classifications[field]

            # NEW: Check if user already provided a value for this field
            if field in user_values:
                user_value = user_values[field]
                form[field] = user_value
                print(f"   ✅ Pre-populated '{field}' from user input: {user_value if not isinstance(user_value, list) else f'[{len(user_value)} items]'}")
                continue

            # Get nested_classification (handle both object and dict forms)
            nested_class = getattr(field_class, 'nested_classification', None) if hasattr(field_class, 'nested_classification') else field_class.get('nested_classification')

            # Handle nested objects recursively
            if isinstance(value, dict) and nested_class:
                nested_classification_data = field_class.nested_classification
                # Create a mock classification object for recursion
                nested_classification = type('obj', (object,), {
                    'field_classifications': nested_classification_data
                })()
                form[field] = self._build_form_from_classification(
                    value,
                    nested_classification,
                    user_values.get(field, {}) if isinstance(user_values.get(field), dict) else {}
                )
                continue

            # Handle arrays - ALWAYS reset to empty (if not pre-populated above)
            if isinstance(value, (list, tuple)):
                form[field] = []
                print(f"   🔄 Reset array field '{field}': {value} → []")
                continue

            # Handle based on AI classification reset flag (handle both object and dict forms)
            reset_flag = getattr(field_class, 'reset', None) if hasattr(field_class, 'reset') else field_class.get('reset', False)
            default_val = getattr(field_class, 'default_value', None) if hasattr(field_class, 'default_value') else field_class.get('default_value')
            category = getattr(field_class, 'category', '') if hasattr(field_class, 'category') else field_class.get('category', '')

            if reset_flag:
                # Reset to default_value from classification
                form[field] = default_val
                print(f"   🔄 Reset field '{field}' ({category}): {value} → {default_val}")
            else:
                # Keep the example default value
                form[field] = value
                print(f"   ✅ Keep default for '{field}' ({category}): {value}")

        return form

    def _extract_defaults_from_classification(
        self,
        example_input: Dict[str, Any],
        classification: Any
    ) -> Dict[str, Any]:
        """Extract fields that kept their default values (were not reset)"""
        defaults = {}
        field_classifications = classification.field_classifications

        for field, value in example_input.items():
            if field not in field_classifications:
                continue

            field_class = field_classifications[field]

            # Handle both object and dict forms
            reset_flag = getattr(field_class, 'reset', None) if hasattr(field_class, 'reset') else field_class.get('reset', False)

            # If field was NOT reset, it's a default
            if not reset_flag and not isinstance(value, (list, tuple)):
                defaults[field] = value

        return defaults

    async def _map_attachments_to_fields(
        self,
        attachments: List[str],
        field_classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map user attachments to appropriate form fields using AI agent
        Returns dict of field_name -> attachment (single value or list for arrays)
        """
        if not attachments:
            return {}

        if not field_classifications:
            print(f"   ⚠️ No field classifications available for attachment mapping")
            return {}

        # Get model context for better AI mapping
        model_name = self.provider.model_name if hasattr(self.provider, 'model_name') else ""
        description = self.provider.description if hasattr(self.provider, 'description') else ""
        example_input = self.provider.example_input if hasattr(self.provider, 'example_input') else {}

        try:
            # Use AI agent for intelligent mapping
            from llm_backend.agents.attachment_mapper import map_attachments_to_fields

            print(f"   🤖 Using AI agent to map {len(attachments)} attachment(s) to fields...")
            mapping_result = await map_attachments_to_fields(
                user_attachments=attachments,
                field_classifications=field_classifications,
                example_input=example_input,
                model_name=model_name,
                model_description=description
            )

            # Convert AI mappings to expected format
            mapping = {}
            for field_mapping in mapping_result.mappings:
                field_name = field_mapping.field_name
                attachment = field_mapping.attachment

                # Get field metadata to determine if it's an array
                field_class = field_classifications.get(field_name)
                if field_class:
                    if hasattr(field_class, 'collection'):
                        is_collection = field_class.collection
                    else:
                        is_collection = field_class.get('collection', False)

                    # For arrays, store as list; for single fields, store as string
                    if is_collection:
                        # Append to existing list or create new one
                        if field_name in mapping:
                            mapping[field_name].append(attachment)
                        else:
                            mapping[field_name] = [attachment]
                    else:
                        # Single field gets single value
                        mapping[field_name] = attachment

                    confidence = field_mapping.confidence
                    print(f"   ✅ AI mapped: {field_name} → {attachment} (confidence: {confidence:.2f})")

            if mapping_result.unmapped_attachments:
                print(f"   ⚠️ Unmapped attachments: {mapping_result.unmapped_attachments}")

            if mapping_result.unmapped_fields:
                print(f"   ⚠️ Unmapped CONTENT fields: {mapping_result.unmapped_fields}")

            return mapping

        except Exception as e:
            print(f"   ⚠️ AI mapping failed ({e}), using improved heuristic fallback")
            return self._heuristic_map_attachments_to_fields(attachments, field_classifications)

    def _heuristic_map_attachments_to_fields(
        self,
        attachments: List[str],
        field_classifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Improved heuristic fallback for attachment mapping
        Handles both single fields AND array fields (not just arrays)
        """
        mapping = {}

        # Analyze attachment file types
        from llm_backend.agents.field_analyzer import analyze_url_pattern, analyze_field_name

        attachment_types = {}
        for url in attachments:
            analysis = analyze_url_pattern(url)
            attachment_types[url] = analysis.get('file_type', 'unknown')

        # Priority patterns for different file types (now includes single field names!)
        field_patterns = {
            'image': ['image', 'img', 'photo', 'picture', 'input_image', 'source_image', 'image_input', 'image_url'],
            'audio': ['audio', 'sound', 'music', 'input_audio', 'audio_file', 'audio_input'],
            'video': ['video', 'movie', 'input_video', 'video_file', 'video_input'],
            'document': ['document', 'doc', 'file', 'input_file', 'document_input'],
        }

        # Score each field for each attachment
        for attachment in attachments:
            file_type = attachment_types.get(attachment, 'unknown')
            relevant_patterns = field_patterns.get(file_type, ['input', 'file', 'attachment'])

            best_field = None
            best_score = 0.0

            for field_name, field_class in field_classifications.items():
                # Get field metadata
                if hasattr(field_class, 'category'):
                    category = str(field_class.category)
                    is_collection = field_class.collection
                    required = field_class.required
                else:
                    category = field_class.get('category', 'HYBRID')
                    is_collection = field_class.get('collection', False)
                    required = field_class.get('required', False)

                # Skip CONFIG fields
                if 'CONFIG' in category:
                    continue

                # Calculate match score
                score = 0.0
                field_lower = field_name.lower()

                # Exact or semantic match
                for pattern in relevant_patterns:
                    if pattern == field_lower:
                        score += 0.9  # Exact match (including "image" = "image")
                        break
                    elif pattern in field_lower or field_lower in pattern:
                        score += 0.7  # Partial match
                        break

                # Boost for CONTENT fields
                if 'CONTENT' in category:
                    score += 0.3

                # Boost for required fields
                if required:
                    score += 0.1

                if score > best_score:
                    best_score = score
                    best_field = field_name

            # Map to best field if score is good enough
            if best_field and best_score > 0.5:
                # Check if field is array or single value
                field_class = field_classifications.get(best_field)
                if hasattr(field_class, 'collection'):
                    is_collection = field_class.collection
                else:
                    is_collection = field_class.get('collection', False)

                if is_collection:
                    # Array field: append or create list
                    if best_field in mapping:
                        mapping[best_field].append(attachment)
                    else:
                        mapping[best_field] = [attachment]
                    print(f"   📎 Heuristic: Mapped {attachment} → {best_field}[] (score: {best_score:.2f})")
                else:
                    # Single field: direct assignment
                    mapping[best_field] = attachment
                    print(f"   📎 Heuristic: Mapped {attachment} → {best_field} (score: {best_score:.2f})")

        if not mapping:
            print(f"   ⚠️ No suitable attachment field found in schema")

        return mapping

    async def _step_information_review(self) -> Dict[str, Any]:
        """Information gathering checkpoint - supports both NL conversation and form modes"""
        import traceback
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: _step_information_review() CALLED")
        print(f"   Run ID: {self.run_id}")
        print(f"   Current state status: {self.state.status}")
        print(f"   Call stack:")
        for line in traceback.format_stack()[-5:-1]:
            print(f"      {line.strip()}")
        print(f"{'='*80}\n")

        self._transition_to_step(HITLStep.INFORMATION_REVIEW)

        # Check if we have form data from initialization
        if not self.state.form_data:
            print("⚠️ No form_data found - falling back to legacy validation")
            return await self._legacy_information_review()

        # Get form classification
        classification = self.state.form_data.get("classification", {})
        current_values = self.state.form_data.get("current_values", {})
        field_classifications = classification.get("field_classifications", {})

        print(f"🔍 DEBUG: Current values from form_data: {current_values}")
        print(f"🔍 DEBUG: Field classifications: {list(field_classifications.keys())}")

        # NEW: Check if natural language mode is enabled
        if self.config.use_natural_language_hitl:
            print("💬 Information Review: Natural language conversation mode")
            return await self._natural_language_information_review(
                classification, current_values
            )

        print("📋 Information Review: Form-based prompting")

        # Build form field definitions for UI
        form_fields = []
        required_fields = []
        optional_fields = []

        for field_name, field_class in field_classifications.items():
            current_value = current_values.get(field_name)

            # Handle both object and dict forms
            value_type = getattr(field_class, 'value_type', None) if hasattr(field_class, 'value_type') else field_class.get("value_type", "string")
            category = getattr(field_class, 'category', None) if hasattr(field_class, 'category') else field_class.get("category")
            required = getattr(field_class, 'required', None) if hasattr(field_class, 'required') else field_class.get("required", False)
            default_value = getattr(field_class, 'default_value', None) if hasattr(field_class, 'default_value') else field_class.get("default_value")
            user_prompt = getattr(field_class, 'user_prompt', None) if hasattr(field_class, 'user_prompt') else field_class.get("user_prompt", f"Provide {field_name}")
            collection = getattr(field_class, 'collection', None) if hasattr(field_class, 'collection') else field_class.get("collection", False)

            field_def = {
                "name": field_name,
                "label": field_name.replace("_", " ").title(),
                "type": self._infer_ui_field_type(value_type),
                "category": category,
                "required": required,
                "current_value": current_value,
                "default": default_value,
                "prompt": user_prompt,
                "collection": collection,
            }

            # Add field-specific attributes
            if collection:
                field_def["hint"] = "You can add multiple items"

            form_fields.append(field_def)

            if required:
                required_fields.append(field_name)
            else:
                optional_fields.append(field_name)

        # Check if all required fields are filled
        missing_required = []
        for field_name in required_fields:
            value = current_values.get(field_name)
            if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                missing_required.append(field_name)

        print(f"📊 Form status:")
        print(f"   Total fields: {len(form_fields)}")
        print(f"   Required fields: {len(required_fields)}")
        print(f"   Missing required: {missing_required}")

        # NEW: Skip pause if form is already complete
        if len(missing_required) == 0:
            print("✅ All required fields already filled from user input - skipping HITL pause")
            self._add_step_event(
                HITLStep.INFORMATION_REVIEW,
                HITLStatus.COMPLETED,
                "system",
                "Form complete - no human input needed"
            )
            return {"continue": True}

        # Determine if we need to pause for user input
        needs_user_input = len(missing_required) > 0

        # Also check policy (only if form is incomplete)
        if not needs_user_input:
            # Get provider capabilities for policy check
            try:
                capabilities = self.provider.get_capabilities()
                needs_user_input = self._should_pause_at_information_review(capabilities)
            except Exception:
                pass

        if needs_user_input:
            print("⏸️ PAUSING for form submission - prompting user for required fields")

            pause_response = self._create_pause_response(
                step=HITLStep.INFORMATION_REVIEW,
                message="Please provide required information to continue",
                actions_required=["submit_form", "provide_required_fields"],
                data={
                    "checkpoint_type": "form_requirements",
                    "form": {
                        "title": "Configure Model Parameters",
                        "fields": form_fields,
                    },
                    "required_fields": required_fields,
                    "optional_fields": optional_fields,
                    "missing_required_fields": missing_required,
                }
            )
            
            # Send WebSocket message for HITL approval request and persist state
            print("🔄 About to call request_human_approval...")
            try:
                if self.websocket_bridge:
                    approval_response = await self.websocket_bridge.request_human_approval(
                        run_id=self.run_id,
                        checkpoint_type="form_requirements",
                        context=pause_response,
                        user_id=getattr(self.run_input, 'user_id', None),
                        session_id=getattr(self.run_input, 'session_id', None)
                    )
                    print("🔄 request_human_approval completed")
                    self._apply_form_submission(approval_response)

                    # NEW: Validate form completeness after submission
                    from llm_backend.core.hitl.validation import validate_form_completeness

                    updated_values = self.state.form_data.get("current_values", {})
                    classification = self.state.form_data.get("classification", {})

                    validation_issues = validate_form_completeness(updated_values, classification)

                    if validation_issues:
                        # Form still incomplete after submission - reject and re-prompt
                        print(f"⚠️ Form validation failed after submission: {len(validation_issues)} issues found")
                        for issue in validation_issues:
                            print(f"   ❌ {issue.field}: {issue.issue}")

                        # Keep status as AWAITING_HUMAN and re-pause
                        self.state.status = HITLStatus.AWAITING_HUMAN
                        self._add_step_event(
                            HITLStep.INFORMATION_REVIEW,
                            HITLStatus.AWAITING_HUMAN,
                            "system",
                            f"Form incomplete - {len(validation_issues)} required fields still missing"
                        )

                        # Re-create pause response with validation errors
                        new_pause_response = self._create_pause_response(
                            step=HITLStep.INFORMATION_REVIEW,
                            message=f"Please fill all required fields. Missing: {', '.join([issue.field for issue in validation_issues])}",
                            actions_required=["submit_form", "provide_required_fields"],
                            data={
                                "checkpoint_type": "form_requirements",
                                "form": pause_response["data"]["form"],
                                "required_fields": required_fields,
                                "optional_fields": optional_fields,
                                "missing_required_fields": [issue.field for issue in validation_issues],
                                "validation_errors": [issue.model_dump() for issue in validation_issues]
                            }
                        )

                        # Re-request approval
                        print("🔄 Re-prompting user for missing fields...")
                        await self.websocket_bridge.request_human_approval(
                            run_id=self.run_id,
                            checkpoint_type="form_requirements",
                            context=new_pause_response,
                            user_id=getattr(self.run_input, 'user_id', None),
                            session_id=getattr(self.run_input, 'session_id', None)
                        )
                        return new_pause_response

                    # Form is complete - continue
                    actor = approval_response.get("approved_by")
                    if actor is None or actor == "":
                        actor_label = "human"
                    else:
                        actor_label = f"human:{actor}"

                    self._add_step_event(
                        HITLStep.INFORMATION_REVIEW,
                        HITLStatus.COMPLETED,
                        actor_label,
                        f"Form submitted with {len(approval_response.get('edits', {}))} fields"
                    )
                    return {"continue": True, "approval": approval_response}
                else:
                    print("⚠️ No websocket_bridge available, cannot request approval")
                    raise Exception("WebSocket bridge is required for HITL approval requests")
            except Exception as ws_error:
                print(f"❌ WebSocket notification failed: {ws_error}")
            return pause_response

        print("✅ All required form fields filled - continuing")
        self._add_step_event(HITLStep.INFORMATION_REVIEW, HITLStatus.COMPLETED, "system", "Form complete - all required fields provided")
        return {"continue": True}

    async def _natural_language_information_review(
        self,
        classification: Dict[str, Any],
        current_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Natural language conversation-based information gathering"""
        from llm_backend.agents.nl_prompt_generator import generate_natural_language_prompt
        from llm_backend.agents.nl_response_parser import parse_natural_language_response

        # Get model info
        model_name = getattr(self.provider, 'model_name', 'the model')
        model_description = getattr(self.provider, 'description', '')

        # Generate natural language prompt
        try:
            nl_prompt = await generate_natural_language_prompt(
                classification=classification,
                current_values=current_values,
                model_name=model_name,
                model_description=model_description
            )
        except Exception as e:
            print(f"⚠️ Failed to generate NL prompt: {e}")
            # Fall back to form-based if NL fails
            self.config.use_natural_language_hitl = False
            return await self._step_information_review()

        # If all fields satisfied, auto-skip
        if nl_prompt.all_fields_satisfied:
            print(f"✅ All required fields satisfied - auto-skipping: {nl_prompt.message}")
            self._add_step_event(
                HITLStep.INFORMATION_REVIEW,
                HITLStatus.COMPLETED,
                "system",
                "All required fields present - auto-approved"
            )
            return {"continue": True}

        # Pause and request user input via natural language
        print(f"⏸️ PAUSING for natural language input")
        print(f"   Message: {nl_prompt.message}")
        print(f"   Missing fields: {nl_prompt.missing_field_names}")

        pause_response = self._create_pause_response(
            step=HITLStep.INFORMATION_REVIEW,
            message=nl_prompt.message,
            actions_required=["respond_naturally"],
            data={
                "checkpoint_type": "information_request",
                "conversation_mode": True,
                "nl_prompt": nl_prompt.message,
                "context": nl_prompt.context,
                "missing_fields": nl_prompt.missing_field_names,
            }
        )

        # Send HITL checkpoint as a message via /from-llm (MESSAGE-BASED HITL)
        checkpoint_data = {
            "run_id": self.run_id,
            "nl_prompt": nl_prompt.message,
            "context": nl_prompt.context,
            "missing_fields": nl_prompt.missing_field_names,
        }

        try:
            await self.hitl_message_client.send_hitl_checkpoint(
                session_id=self.session_id,
                user_id=self.user_id or "unknown",
                content=nl_prompt.message,
                checkpoint_type="information_request",
                checkpoint_data=checkpoint_data
            )
            print(f"✅ Sent information request checkpoint as message")
        except Exception as e:
            print(f"❌ Failed to send HITL checkpoint message: {e}")
            pass

        # Return pause response - user will respond when ready
        # When user responds, their message will be detected by /to-llm handler (HITL context detection)
        # and resume_from_state() will be called to continue the workflow
        return pause_response

    async def _legacy_information_review(self) -> Dict[str, Any]:
        """Legacy information review using validation checkpoints (fallback)"""
        # Extract tool config from the correct key structure
        agent_tool_config = self.run_input.agent_tool_config or {}
        replicate_tool_config = agent_tool_config.get("replicate-agent-tool", {})

        # Handle both flat and nested data formats
        if 'data' in replicate_tool_config and isinstance(replicate_tool_config.get('data'), dict) and replicate_tool_config['data']:
            replicate_config = replicate_tool_config['data']
        else:
            replicate_config = replicate_tool_config

        print(f"🔍 Orchestrator: Using tool_config: {replicate_config}")

        validator = HITLValidator(
            run_input=self.run_input,
            tool_config=replicate_config
        )

        validation_checkpoints = validator.validate_pre_execution()
        validation_summary = create_hitl_validation_summary(validation_checkpoints)
        friendly_message = validation_summary.get("user_friendly_message", "I need your help to continue")

        print(f"🚨 Blocking Issues: {validation_summary['blocking_issues']}")

        # Store validation results in state
        self.state.validation_checkpoints = validation_summary
        self.state.validation_summary = validation_summary
        self.state.user_friendly_message = friendly_message

        # Get provider capabilities
        try:
            capabilities = self.provider.get_capabilities()
            self.state.capabilities = capabilities.dict()
        except Exception:
            # Create minimal capabilities to continue
            from llm_backend.core.providers.base import ProviderCapabilities, OperationType
            capabilities = ProviderCapabilities(
                name="remove-bg",
                operation_types=[OperationType.IMAGE_PROCESSING],
                input_types=["image"],
                output_types=["image"]
            )

        # Check if human review is required based on validation or capabilities
        blocking_issues = validation_summary["blocking_issues"] > 0

        # Only pause if there are blocking issues OR policy requires human review
        if blocking_issues:
            needs_review = True
        else:
            needs_review = self._should_pause_at_information_review(capabilities)

        print(f"🤔 Needs review? {needs_review} (blocking_issues: {blocking_issues})")

        if needs_review:
            print("⏸️ PAUSING for human review - sending WebSocket message")
            pause_response = self._create_pause_response(
                step=HITLStep.INFORMATION_REVIEW,
                message="Model capabilities and parameters require human review",
                actions_required=["approve", "edit_prompt", "change_model", "fix_validation_issues"],
                data={
                    "capabilities": capabilities.dict(),
                    "confidence_score": self._calculate_information_confidence(capabilities),
                    "validation_summary": validation_summary,
                    "blocking_issues": validation_summary["blocking_issues"],
                    "checkpoints": validation_summary["checkpoints"]
                }
            )

            # Send WebSocket message for HITL approval request and persist state
            print("🔄 About to call request_human_approval...")
            try:
                if self.websocket_bridge:
                    approval_response = await self.websocket_bridge.request_human_approval(
                        run_id=self.run_id,
                        checkpoint_type="information_review",
                        context=pause_response,
                        user_id=getattr(self.run_input, 'user_id', None),
                        session_id=getattr(self.run_input, 'session_id', None)
                    )
                    print("🔄 request_human_approval completed")
                    self._apply_approval_response(approval_response, HITLStep.INFORMATION_REVIEW)

                    actor = approval_response.get("approved_by")
                    if actor is None or actor == "":
                        actor_label = "human"
                    else:
                        actor_label = f"human:{actor}"

                    self._add_step_event(
                        HITLStep.INFORMATION_REVIEW,
                        HITLStatus.COMPLETED,
                        actor_label,
                        f"Human action: {approval_response.get('action', 'unknown')}"
                    )
                    return {"continue": True, "approval": approval_response}
                else:
                    print("⚠️ No websocket_bridge available, cannot request approval")
                    raise Exception("WebSocket bridge is required for HITL approval requests")
            except Exception as ws_error:
                print(f"❌ WebSocket notification failed: {ws_error}")
            return pause_response

        self._add_step_event(HITLStep.INFORMATION_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved based on validation and thresholds")
        return {"continue": True}

    def _infer_ui_field_type(self, value_type: str) -> str:
        """Map value type to UI field type"""
        type_mapping = {
            "string": "text",
            "integer": "number",
            "number": "number",
            "boolean": "checkbox",
            "array": "array",
            "object": "object",
        }
        return type_mapping.get(value_type, "text")
    
    async def _step_payload_review(self) -> Dict[str, Any]:
        """Enhanced payload review checkpoint with improved error handling"""
        self._transition_to_step(HITLStep.PAYLOAD_REVIEW)
        
        try:
            # Create payload
            operation_type = self._infer_operation_type()
            hitl_edits = self._collect_hitl_edits()

            # Determine attachments source: form data (if available) or full discovery
            if self.state.form_data and self.state.form_data.get("current_values"):
                # Use attachments from form data (already filtered and user-supplied)
                current_values = self.state.form_data.get("current_values", {})
                form_attachments = []

                # Extract attachment arrays from form fields
                for field_name, value in current_values.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        # This looks like an attachment array
                        form_attachments.extend(value)

                attachments_to_use = form_attachments
                print(f"📎 Using {len(attachments_to_use)} attachments from form data")
            else:
                # Fallback to full attachment discovery
                attachments_to_use = self._gather_attachments()

            payload = self.provider.create_payload(
                prompt=self.run_input.prompt,
                attachments=attachments_to_use,
                operation_type=operation_type,
                config=self.run_input.agent_tool_config or {},
                conversation=getattr(self.run_input, 'conversation', None),
                hitl_edits=hitl_edits or None
            )

            if asyncio.iscoroutine(payload):
                payload = await payload

            # Validate payload with same attachments
            validation_issues = self.provider.validate_payload(
                payload,
                self.run_input.prompt,
                attachments_to_use
            )
            
            payload_dict = payload.dict()
            if hitl_edits:
                payload_dict = self._merge_payload_edits(payload_dict, hitl_edits)
            self.state.suggested_payload = payload_dict
            self.state.validation_issues = [issue.dict() for issue in validation_issues]

            # Log validation results for debugging
            if validation_issues:
                print(f"⚠️ Validation found {len(validation_issues)} issue(s):")
                for issue in validation_issues:
                    print(f"   {issue.severity.upper()}: {issue.field} - {issue.issue}")
            else:
                print("✅ Payload validation passed - no issues found")

            # Check for critical validation failures that require human intervention
            critical_issues = [issue for issue in validation_issues if issue.severity == "error" and not issue.auto_fixable]

            if critical_issues:
                # Send natural language message to user about validation failures
                nl_message = self._format_validation_errors_naturally(critical_issues)

                try:
                    await self.hitl_message_client.send_hitl_checkpoint(
                        session_id=self.session_id,
                        user_id=self.user_id or "unknown",
                        content=nl_message,
                        checkpoint_type="payload_validation",
                        checkpoint_data={
                            "run_id": self.run_id,
                            "critical_issues": [issue.dict() for issue in critical_issues],
                            "missing_inputs": self._identify_missing_inputs(critical_issues),
                        }
                    )
                    print(f"✅ Sent payload validation error message to user")
                except Exception as e:
                    print(f"❌ Failed to send payload validation message: {e}")

                return self._create_pause_response(
                    step=HITLStep.PAYLOAD_REVIEW,
                    message="Critical validation failures require human intervention",
                    actions_required=["fix_critical_issues", "provide_missing_inputs", "change_model"],
                    data={
                        "suggested_payload": payload.dict(),
                        "validation_issues": [issue.dict() for issue in validation_issues],
                        "critical_issues": [issue.dict() for issue in critical_issues],
                        "missing_inputs": self._identify_missing_inputs(critical_issues),
                        "suggested_actions": self._suggest_remediation_actions(critical_issues)
                    }
                )
            
            # Check if human review is required for non-critical issues
            if self._should_pause_at_payload_review(payload, validation_issues):
                return self._create_pause_response(
                    step=HITLStep.PAYLOAD_REVIEW,
                    message="Payload requires human review",
                    actions_required=["approve", "edit_payload", "fix_validation_issues"],
                    data={
                        "suggested_payload": payload.dict(),
                        "validation_issues": [issue.dict() for issue in validation_issues],
                        "diff_from_example": self._calculate_payload_diff(payload),
                        "estimated_cost": self.provider.estimate_cost(payload),
                        "auto_fixable_issues": len([issue for issue in validation_issues if issue.auto_fixable])
                    }
                )
            
            # Auto-fix validation issues if possible
            if validation_issues:
                fixed_payload, fix_results = self._auto_fix_payload_with_results(payload, validation_issues)
                if fixed_payload:
                    payload = fixed_payload
                    self.state.suggested_payload = payload.dict()
                    self._add_step_event(HITLStep.PAYLOAD_REVIEW, HITLStatus.RUNNING, "system", f"Auto-fixed {len(fix_results)} validation issues")
            
            self._add_step_event(HITLStep.PAYLOAD_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved payload")
            return {"continue": True, "payload": payload}
            
        except Exception as e:
            # Handle payload creation/validation errors
            self._add_step_event(HITLStep.PAYLOAD_REVIEW, HITLStatus.FAILED, "system", f"Payload creation failed: {str(e)}")
            return self._create_pause_response(
                step=HITLStep.PAYLOAD_REVIEW,
                message="Failed to create or validate payload",
                actions_required=["retry", "change_model", "fix_configuration"],
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "suggested_fixes": self._suggest_payload_error_fixes(e)
                }
            )

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
            # Try to find the field in payload structure
            if hasattr(payload, 'input') and isinstance(payload.input, dict):
                current_value = payload.input.get(field)
            elif 'input' in payload:
                current_value = payload['input'].get(field)

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
        if self.state_manager:
            await self.state_manager.save_state(self.state)

        # Send HITL checkpoint as a message via /from-llm
        checkpoint_data = {
            "run_id": self.run_id,
            "error_type": "validation",
            "error_field": field,
            "current_value": current_value,
            "valid_values": valid_values,
        }

        try:
            await self.hitl_message_client.send_hitl_checkpoint(
                session_id=self.session_id,
                user_id=self.user_id or "unknown",
                content=nl_error,
                checkpoint_type="error_recovery",
                checkpoint_data=checkpoint_data
            )
            print(f"✅ Sent error recovery checkpoint as message")

        except Exception as e:
            print(f"❌ Failed to send HITL checkpoint message: {e}")
            # Fall back to returning pause response without sending message
            pass

        # Return pause response - user will respond when ready
        return self._create_pause_response(
            step=HITLStep.API_CALL,
            message=nl_error,
            actions_required=["respond_naturally"],
            checkpoint_type="error_recovery",
            conversation_mode=True,
            data={
                "error_type": "validation",
                "error_field": field,
                "valid_values": valid_values,
            }
        )

    async def _step_api_execution(self) -> Dict[str, Any]:
        """API execution step"""
        self._transition_to_step(HITLStep.API_CALL, HITLStatus.RUNNING)
        user_id = getattr(self.run_input, 'user_id', None)
        session_id = getattr(self.run_input, 'session_id', None)

        # Signal model execution start
        if self.websocket_bridge:
            await self.websocket_bridge.send_thinking_status(
                message="Running model inference",
                user_id=user_id,
                session_id=session_id
            )
        
        # Get payload from previous step or state
        # Create base payload with HITL edits integration
        hitl_edits = self._collect_hitl_edits()
        if hitl_edits:
            print(f"🎯 Passing HITL edits to provider: {hitl_edits}")

        # Determine attachments source: form data (if available) or full discovery
        if self.state.form_data and self.state.form_data.get("current_values"):
            # Use attachments from form data (already filtered and user-supplied)
            current_values = self.state.form_data.get("current_values", {})
            form_attachments = []

            # Extract attachment arrays from form fields
            for field_name, value in current_values.items():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    # This looks like an attachment array
                    form_attachments.extend(value)

            attachments_to_use = form_attachments
            print(f"📎 Using {len(attachments_to_use)} attachments from form data for execution")
        else:
            # Fallback to full attachment discovery
            attachments_to_use = self._gather_attachments()

        payload = self.provider.create_payload(
            prompt=self.run_input.prompt,
            attachments=attachments_to_use,
            operation_type=self._infer_operation_type(),
            config=self.run_input.agent_tool_config or {},
            conversation=getattr(self.run_input, 'conversation', None),
            hitl_edits=hitl_edits or None
        )

        if asyncio.iscoroutine(payload):
            payload = await payload
        
        # Note: HITL edits are now handled by the intelligent agent in create_payload()
        # This eliminates the need for manual field mapping and override logic
        
        # Execute with provider
        start_time = time.time()
        response = self.provider.execute(payload)
        execution_time = int((time.time() - start_time) * 1000)

        self.state.provider_execution_time_ms = execution_time
        self.state.raw_response = response.raw_response
        self.state.processed_response = response.processed_response

        # Check for errors and handle recoverable ones
        if response.error:
            error_details = response.metadata.get("error_details", {})
            recoverable = response.metadata.get("recoverable", False)

            print(f"❌ API error: {response.error}")
            print(f"   Recoverable: {recoverable}")
            print(f"   Error type: {error_details.get('error_type')}")

            # Handle validation errors with NL recovery
            if recoverable and error_details.get("error_type") == "validation":
                return await self._handle_validation_error_nl(response, payload)

            # Non-recoverable errors fail immediately
            raise Exception(f"Provider execution failed: {response.error}")

        # Don't send done_thinking here - Replicate webhook will handle completion status
        # The webhook handler elsewhere will send done_thinking when the actual result arrives

        audited_response = self.provider.audit_response(response)
        
        # Check if human review is required
        if self._should_pause_at_response_review(response, audited_response):
            return self._create_pause_response(
                step=HITLStep.RESPONSE_REVIEW,
                message="Response requires human review",
                actions_required=["approve", "edit_response", "retry"],
                data={
                    "raw_response": response.raw_response,
                    "processed_response": response.processed_response,
                    "audited_response": audited_response,
                    "quality_score": self._calculate_response_quality(response)
                }
            )
        
        self.state.final_result = audited_response
        self._add_step_event(HITLStep.API_CALL, HITLStatus.COMPLETED, "system", f"Executed in {execution_time}ms")
        return {"continue": True, "response": response}
    
    async def _step_response_review(self) -> Dict[str, Any]:
        """Response review checkpoint"""
        self._transition_to_step(HITLStep.RESPONSE_REVIEW)
        
        # Audit response
        response_dict = {
            "raw_response": self.state.raw_response,
            "processed_response": self.state.processed_response,
            "metadata": {},
            "execution_time_ms": self.state.provider_execution_time_ms or 0
        }
        response = ProviderResponse(**response_dict)
        audited_response = self.provider.audit_response(response)
        
        # Check if human review is required
        if self._should_pause_at_response_review(response, audited_response):
            return self._create_pause_response(
                step=HITLStep.RESPONSE_REVIEW,
                message="Response requires human review",
                actions_required=["approve", "edit_response", "retry"],
                data={
                    "raw_response": response.raw_response,
                    "processed_response": response.processed_response,
                    "audited_response": audited_response,
                    "quality_score": self._calculate_response_quality(response)
                }
            )
        
        self.state.final_result = audited_response
        self._add_step_event(HITLStep.RESPONSE_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved response")
        return {"continue": True, "final_result": audited_response}
    
    async def _step_completion(self) -> Dict[str, Any]:
        """Completion step"""
        self._transition_to_step(HITLStep.COMPLETED, HITLStatus.COMPLETED)
        
        return {
            "run_id": self.run_id,
            "status": "completed",
            "result": self.state.final_result,
            "metadata": {
                "total_execution_time_ms": self.state.total_execution_time_ms,
                "provider_execution_time_ms": self.state.provider_execution_time_ms,
                "human_review_time_ms": self.state.human_review_time_ms,
                "step_count": len(self.state.step_history)
            }
        }
    
    async def approve_current_step(self, actor: str, message: Optional[str] = None) -> Dict[str, Any]:
        """Approve the current step and continue execution"""
        if self.state.status != HITLStatus.AWAITING_HUMAN:
            raise ValueError("No pending human action")
        
        self._add_step_event(self.state.current_step, HITLStatus.COMPLETED, actor, message or "Approved")
        
        # Continue execution from current step
        return await self.execute()
    
    async def edit_current_step(self, actor: str, edits: Dict[str, Any], message: Optional[str] = None) -> Dict[str, Any]:
        """Apply edits to current step and continue"""
        if self.state.status != HITLStatus.AWAITING_HUMAN:
            raise ValueError("No pending human action")
        
        # Apply edits based on current step
        if self.state.current_step == HITLStep.INFORMATION_REVIEW:
            if "prompt" in edits:
                self.run_input.prompt = edits["prompt"]
            if "model_config" in edits:
                self.run_input.agent_tool_config.update(edits["model_config"])
            
            # Store human edits in state for persistence across steps
            for field, value in edits.items():
                if field in ["input_image", "source_image", "driven_audio", "audio_file"]:
                    self.state.human_edits[field] = value
                    print(f"🧩 Stored human edit: {field} = {value}")
            
            # Also update suggested_payload for backward compatibility
            file_fields = ["input_image", "source_image", "driven_audio", "audio_file"]
            field_aliases = {
                "input_image": ["image", "input_image", "source_image"],
                "source_image": ["image", "source_image", "input_image"],
                "driven_audio": ["audio", "driven_audio", "audio_file"],
                "audio_file": ["audio", "audio_file", "driven_audio"],
            }

            for field in file_fields:
                if field in edits:
                    if not hasattr(self.state, 'suggested_payload') or self.state.suggested_payload is None:
                        self.state.suggested_payload = {}

                    # Store raw edit for reference
                    self.state.suggested_payload[field] = edits[field]

                    # Ensure nested input payload exists
                    if "input" not in self.state.suggested_payload or not isinstance(self.state.suggested_payload["input"], dict):
                        self.state.suggested_payload["input"] = {}

                    input_payload = self.state.suggested_payload["input"]
                    aliases = field_aliases.get(field, [field])
                    updated = False

                    # Prefer updating existing alias keys
                    for alias in aliases:
                        if alias in input_payload:
                            input_payload[alias] = edits[field]
                            updated = True

                    # If no existing alias present, set the first alias as default
                    if not updated and aliases:
                        primary_alias = aliases[0]
                        input_payload[primary_alias] = edits[field]

                    print(
                        f"🧩 Applied file edit '{field}' -> aliases {aliases}"
                        f" (set value: {edits[field]})"
                    )
        
        elif self.state.current_step == HITLStep.PAYLOAD_REVIEW:
            if "payload" in edits:
                self.state.suggested_payload.update(edits["payload"])
            else:
                # Apply individual field edits directly to suggested_payload
                if not hasattr(self.state, 'suggested_payload') or self.state.suggested_payload is None:
                    self.state.suggested_payload = {}
                self.state.suggested_payload.update(edits)
        
        elif self.state.current_step == HITLStep.RESPONSE_REVIEW:
            if "response" in edits:
                self.state.final_result = edits["response"]
        
        self._add_step_event(self.state.current_step, HITLStatus.COMPLETED, actor, f"Edited: {message}")
        
        # Save state after applying edits
        if self.state_manager:
            try:
                await self.state_manager.save_state(self.state)
                print("💾 Saved state after applying edits")
            except Exception as e:
                print(f"❌ Failed to save state after edits: {e}")
        
        # Continue execution
        return await self.execute()
    
    async def reject_current_step(self, actor: str, reason: str) -> Dict[str, Any]:
        """Reject current step and cancel run"""
        self._transition_to_step(self.state.current_step, HITLStatus.CANCELLED)
        self._add_step_event(self.state.current_step, HITLStatus.CANCELLED, actor, f"Rejected: {reason}")
        
        return {
            "run_id": self.run_id,
            "status": "cancelled",
            "reason": reason,
            "cancelled_at_step": self.state.current_step
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current run state"""
        return {
            "run_id": self.run_id,
            "status": self.state.status,
            "current_step": self.state.current_step,
            "pending_actions": self.state.pending_actions,
            "created_at": self.state.created_at.isoformat(),
            "updated_at": self.state.updated_at.isoformat(),
            "expires_at": self.state.expires_at.isoformat() if self.state.expires_at else None,
            "step_history": [event.dict() for event in self.state.step_history],
            "metadata": {
                "total_execution_time_ms": self.state.total_execution_time_ms,
                "human_review_time_ms": self.state.human_review_time_ms
            }
        }
    
    # Helper methods
    def _should_pause_at_information_review(self, capabilities) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.INFORMATION_REVIEW in self.config.allowed_steps
        
        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            confidence = self._calculate_information_confidence(capabilities)
            return confidence < self.config.auto_approve_confidence_threshold
        
        return False
    
    def _should_pause_at_payload_review(self, payload, issues) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.PAYLOAD_REVIEW in self.config.allowed_steps

        # Only pause for NON-auto-fixable errors
        non_fixable_errors = [
            issue for issue in issues
            if issue.severity == "error" and not issue.auto_fixable
        ]
        if non_fixable_errors:
            return True

        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            # Check payload changes threshold
            changes = self._count_payload_changes(payload)
            if changes > self.config.max_payload_changes:
                return True

        return False
    
    def _should_pause_at_response_review(self, response, audited) -> bool:
        if self.config.policy == HITLPolicy.REQUIRE_HUMAN:
            return HITLStep.RESPONSE_REVIEW in self.config.allowed_steps
        
        if self.config.policy == HITLPolicy.AUTO_WITH_THRESHOLDS:
            quality = self._calculate_response_quality(response)
            threshold = self.config.review_thresholds.get("response_quality_min", 0.7) if self.config.review_thresholds else 0.7
            return quality < threshold
        
        return False
    
    def _create_pause_response(self, step: HITLStep, message: str, actions_required: list, data: Dict[str, Any] = None, checkpoint_type: str = None, conversation_mode: bool = False) -> Dict[str, Any]:
        self.state.status = HITLStatus.AWAITING_HUMAN
        self.state.pending_actions = actions_required
        self.state.approval_token = str(uuid.uuid4())
        self.state.updated_at = datetime.utcnow()

        # Store checkpoint metadata in state for resume logic
        if not hasattr(self.state, 'checkpoint_context'):
            self.state.checkpoint_context = {}

        if checkpoint_type:
            self.state.checkpoint_context['checkpoint_type'] = checkpoint_type
        if conversation_mode:
            self.state.checkpoint_context['conversation_mode'] = conversation_mode

        response = {
            "run_id": self.run_id,
            "status": "awaiting_human",
            "current_step": step,
            "message": message,
            "actions_required": actions_required,
            "approval_token": self.state.approval_token,
            "expires_at": self.state.expires_at.isoformat() if self.state.expires_at else None,
            "events_url": f"/hitl/runs/{self.run_id}/events"
        }

        if checkpoint_type:
            response["checkpoint_type"] = checkpoint_type
        if conversation_mode:
            response["conversation_mode"] = conversation_mode

        if data:
            response.update(data)

        return response
    
    def _transition_to_step(self, step: HITLStep, status: HITLStatus = HITLStatus.RUNNING):
        """Transition to a new step"""
        self.state.current_step = step
        self.state.status = status
        self.state.updated_at = datetime.utcnow()
    
    def _add_step_event(self, step: HITLStep, status: HITLStatus, actor: str, message: str):
        event = StepEvent(
            step=step,
            status=status,
            timestamp=datetime.utcnow(),
            actor=actor,
            message=message
        )
        self.state.step_history.append(event)
        print(f"📝 Added step event: {step.value} -> {status.value} ({actor}: {message})")
        
        if self.state_manager:
            print("💾 Saving step event to database...")
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(self.state_manager.save_state(self.state))
        else:
            print("⚠️ No state_manager - step event NOT saved to database")
            try:
                pass
            except Exception as e:
                print(f"⚠️ Failed to save HITL state: {e}")
    
    def _is_paused(self, result: Dict[str, Any]) -> bool:
        return result.get("status") == "awaiting_human"
    async def _handle_error(self, error: Exception) -> Dict[str, Any]:
        self._transition_to_step(self.state.current_step, HITLStatus.FAILED)
        self._add_step_event(self.state.current_step, HITLStatus.FAILED, "system", str(error))
        
        return {
            "run_id": self.run_id,
            "status": "failed",
            "error": str(error),
            "failed_at_step": self.state.current_step
        }
    
    # Placeholder helper methods (to be implemented based on specific needs)
    def _infer_operation_type(self):
        from llm_backend.core.providers.base import OperationType
        return OperationType.TEXT_GENERATION  # Default, should be smarter
    
    def _calculate_information_confidence(self, capabilities) -> float:
        return 0.8  # Placeholder
    
    def _calculate_payload_diff(self, payload) -> Dict[str, Any]:
        return {}  # Placeholder
    
    def _auto_fix_payload(self, payload, issues):
        return payload  # Placeholder
    
    def _count_payload_changes(self, payload) -> int:
        return 0  # Placeholder
    
    def _calculate_response_quality(self, response) -> float:
        return 0.8  # Placeholder
    
    def _identify_missing_inputs(self, critical_issues) -> Dict[str, str]:
        """Identify what inputs are missing based on critical validation issues"""
        missing_inputs = {}
        
        for issue in critical_issues:
            if "missing" in issue.issue.lower() or "required" in issue.issue.lower():
                if "prompt" in issue.field.lower() or "text" in issue.field.lower():
                    missing_inputs["text_input"] = "Please provide a clear text prompt describing what you want to do"
                elif "image" in issue.field.lower():
                    missing_inputs["image_input"] = "Please upload an image file"
                elif "audio" in issue.field.lower():
                    missing_inputs["audio_input"] = "Please upload an audio file"
                elif "file" in issue.field.lower():
                    missing_inputs["file_input"] = "Please upload the required file"
                else:
                    missing_inputs[issue.field] = issue.suggested_fix
        
        return missing_inputs
    
    def _format_validation_errors_naturally(self, critical_issues) -> str:
        """Format validation errors as a natural language message for the user"""
        if not critical_issues:
            return "Everything looks good!"

        # Group issues by type
        missing_fields = []
        invalid_values = []
        other_issues = []

        for issue in critical_issues:
            if "missing" in issue.issue.lower() or "required" in issue.issue.lower():
                missing_fields.append(issue)
            elif "invalid" in issue.issue.lower() or "not found" in issue.issue.lower():
                invalid_values.append(issue)
            else:
                other_issues.append(issue)

        # Build natural language message
        parts = []

        if missing_fields:
            if len(missing_fields) == 1:
                parts.append(f"Oops! I need {missing_fields[0].suggested_fix or missing_fields[0].issue}")
            else:
                missing_list = ", ".join([issue.field for issue in missing_fields])
                parts.append(f"Oops! I'm missing some required information: {missing_list}")

        if invalid_values:
            for issue in invalid_values:
                parts.append(f"There's an issue with {issue.field}: {issue.issue}")
                if issue.suggested_fix:
                    parts.append(f"Try: {issue.suggested_fix}")

        if other_issues:
            for issue in other_issues:
                parts.append(f"{issue.issue}")

        # Add friendly closing
        if len(critical_issues) > 1:
            parts.append("Can you help me with these?")
        else:
            parts.append("Can you help with that?")

        return " ".join(parts)

    def _suggest_remediation_actions(self, critical_issues) -> list:
        """Suggest specific actions to remediate critical validation issues"""
        actions = []

        for issue in critical_issues:
            action = {
                "field": issue.field,
                "issue": issue.issue,
                "action": issue.suggested_fix,
                "priority": "high" if issue.severity == "error" else "medium"
            }
            actions.append(action)

        return actions
    
    def _auto_fix_payload_with_results(self, payload, validation_issues):
        """Auto-fix payload validation issues and return results"""
        fixed_payload = payload
        fix_results = []
        
        for issue in validation_issues:
            if issue.auto_fixable:
                try:
                    # Apply auto-fixes based on issue type
                    if "prompt" in issue.field.lower() and hasattr(fixed_payload, 'input'):
                        # Map prompt to appropriate field
                        if 'prompt' in fixed_payload.input:
                            fixed_payload.input['prompt'] = self.run_input.prompt
                        elif 'text' in fixed_payload.input:
                            fixed_payload.input['text'] = self.run_input.prompt
                        
                        fix_results.append({
                            "field": issue.field,
                            "fix_applied": f"Mapped prompt to {issue.field}",
                            "success": True
                        })
                    
                    elif "image" in issue.field.lower():
                        # Try to map an image from form data or gathered attachments
                        attachments = []

                        # Try form data first
                        if self.state.form_data and self.state.form_data.get("current_values"):
                            current_values = self.state.form_data.get("current_values", {})
                            for field_name, value in current_values.items():
                                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                    attachments.extend(value)

                        # Fallback to full discovery if form data empty
                        if not attachments:
                            attachments = self._gather_attachments()

                        if attachments and hasattr(fixed_payload, 'input') and 'image' in fixed_payload.input:
                            fixed_payload.input['image'] = attachments[0]

                        fix_results.append({
                            "field": issue.field,
                            "fix_applied": f"Mapped discovered attachment to {issue.field}",
                            "success": True
                        })
                
                except Exception as e:
                    fix_results.append({
                        "field": issue.field,
                        "fix_applied": f"Failed to auto-fix: {str(e)}",
                        "success": False
                    })
        
        return fixed_payload if fix_results else None, fix_results

    def _extract_and_clean_urls_from_prompt(self, prompt: str) -> tuple:
        """
        Extract URLs from prompt text and return cleaned prompt without URLs.

        Args:
            prompt: The original prompt text that may contain URLs

        Returns:
            Tuple of (cleaned_prompt, extracted_urls)
        """
        if not prompt or not isinstance(prompt, str):
            return prompt, []

        extracted_urls = []
        cleaned_prompt = prompt

        # Find all URLs in the prompt
        url_pattern = r'https?://\S+'
        matches = list(re.finditer(url_pattern, prompt))

        # Process matches in reverse to maintain string positions
        for match in reversed(matches):
            url = match.group(0).rstrip(')>,.;\'"')
            extracted_urls.insert(0, url)  # Insert at beginning to maintain order
            # Remove the URL from prompt
            start, end = match.span()
            cleaned_prompt = cleaned_prompt[:start] + cleaned_prompt[end:]

        # Strip breadcrumb markers for attachments (e.g., ":-> attached document:")
        cleaned_prompt = re.sub(r'\s*:->\s*attached\s+document:?', '', cleaned_prompt, flags=re.IGNORECASE)
        cleaned_prompt = re.sub(r'\n\s*:->\s*attached\s+document:?', '', cleaned_prompt, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()

        if extracted_urls:
            print(f"🔗 Extracted {len(extracted_urls)} URL(s) from prompt text")
            print(f"📝 Cleaned prompt: '{cleaned_prompt[:100]}{'...' if len(cleaned_prompt) > 100 else ''}'")

        return cleaned_prompt, extracted_urls

    def _gather_user_supplied_attachments(self) -> list:
        """Gather ONLY explicit attachments supplied by the user in this run.

        This is used for form pre-population and excludes:
        - Example/demo URLs from agent_tool_config
        - Chat history (which may contain old attachments from previous runs)
        - HITL edits (not yet provided during initialization)
        - URLs embedded in prompt (handled separately by _extract_and_clean_urls_from_prompt)

        Only includes:
        - Attachments from run_input.attachments
        """
        attachments: List[str] = []

        def add_attachment(value: Any):
            if isinstance(value, str):
                candidate = value.strip()
                if candidate and candidate not in attachments:
                    attachments.append(candidate)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add_attachment(item)

        # Attachments explicitly supplied on run input
        run_input_attachments = getattr(self.run_input, "attachments", None)
        add_attachment(run_input_attachments)

        if attachments:
            print(f"🔗 User supplied explicit attachments for run {self.run_id}: {attachments}")

        return attachments

    def _gather_attachments(self) -> list:
        """Gather attachments from available sources.

        Combines chat history discovery, run input metadata, HITL edits,
        and URLs embedded in the prompt."""

        attachments: List[str] = []

        def add_attachment(value: Any):
            if isinstance(value, str):
                candidate = value.strip()
                if candidate and candidate not in attachments:
                    attachments.append(candidate)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add_attachment(item)

        # Discover attachments from chat history (placeholder implementation)
        history_assets = self._attachments_from_chat_history()
        add_attachment(history_assets)

        # Attachments explicitly supplied on run input
        run_input_attachments = getattr(self.run_input, "attachments", None)
        add_attachment(run_input_attachments)

        # Attachments from agent tool configuration (including nested payloads)
        agent_tool_config = getattr(self.run_input, "agent_tool_config", {}) or {}

        def collect_from_config(config: Any):
            if isinstance(config, dict):
                for key, value in config.items():
                    if key in {
                        "attachments",
                        "attachment_urls",
                        "images",
                        "image_urls",
                        "image_input",
                        "input_image",
                        "source_image",
                        "image",
                        "media",
                        "media_urls",
                        "file",
                        "file_input",
                    }:
                        add_attachment(value)
                    collect_from_config(value)
            elif isinstance(config, (list, tuple, set)):
                for item in config:
                    collect_from_config(item)

        collect_from_config(agent_tool_config)

        # Attachments derived from human edits (e.g., manual overrides)
        hitl_edits = self._collect_hitl_edits()
        if hitl_edits:
            add_attachment(hitl_edits.values())

        # Parse URLs embedded directly inside the prompt text
        prompt_text = getattr(self.run_input, "prompt", "")
        if isinstance(prompt_text, str) and prompt_text:
            for match in re.findall(r"https?://\S+", prompt_text):
                cleaned = match.rstrip(')>,.;\'"')
                add_attachment(cleaned)

        if attachments:
            print(f"🔗 Gathered attachments for run {self.run_id}: {attachments}")

        return attachments

    def _calculate_information_confidence(self, capabilities) -> float:
        """Calculate confidence score for information review step"""
        # Placeholder implementation - could be enhanced with actual confidence metrics
        return 0.85
    
    def _calculate_response_quality(self, response) -> float:
        """Calculate quality score for response review step"""
        # Placeholder implementation - could analyze response content, length, etc.
        return 0.9
    
    def _count_payload_changes(self, payload) -> int:
        """Count number of changes made to payload from example input"""
        # Placeholder implementation - could compare against example_input
        return 0

    def _attachments_from_chat_history(self) -> list:
        """Discover recent media assets (images, videos, audio) from chat history.

        Extracts URLs from conversation history that appear to be media files.
        Filters out HITL checkpoint messages to avoid processing system messages.
        """
        attachments = []

        try:
            # Get conversation from run_input if available
            conversation = getattr(self.run_input, "conversation", None)

            if not conversation or not isinstance(conversation, list):
                return []

            # Media file extensions to look for
            media_extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg',  # Images
                '.mp4', '.webm', '.mov', '.avi', '.mkv',  # Videos
                '.mp3', '.wav', '.ogg', '.m4a', '.flac',  # Audio
            }

            # Process messages from most recent backwards (limit to last 10 for performance)
            for message in reversed(conversation[-10:]):
                if not isinstance(message, dict):
                    continue

                # Skip HITL checkpoint/system messages
                message_type = message.get("message_type", "")
                if "hitl" in message_type.lower() or "checkpoint" in message_type.lower():
                    continue

                role = message.get("role", "")
                if role == "system":
                    continue

                # Extract content
                content = message.get("content", "")
                if not isinstance(content, str):
                    continue

                # Find all URLs in the message
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                urls = re.findall(url_pattern, content)

                for url in urls:
                    # Clean trailing punctuation
                    url = url.rstrip('.,;:!?)')

                    # Check if it's a media file
                    url_lower = url.lower()
                    if any(url_lower.endswith(ext) for ext in media_extensions):
                        if url not in attachments:
                            attachments.append(url)
                            print(f"📎 Found attachment in chat history: {url}")

            return attachments

        except Exception as e:
            print(f"⚠️ Error extracting attachments from chat history: {e}")
            return []

    def _extract_field_values_from_conversation(self, field_names: List[str]) -> Dict[str, Any]:
        """
        Parse conversation history for field=value patterns.

        Looks for patterns like:
        - field_name=value
        - field_name: value
        - field_name | value
        - field name = value (with spaces)

        Extracts and parses values (booleans, numbers, strings).
        """
        extracted_values = {}

        try:
            # Get conversation from run_input
            conversation = getattr(self.run_input, "conversation", None)

            if not conversation or not isinstance(conversation, list):
                return {}

            # Process last 10 messages (most recent context)
            recent_messages = conversation[-10:]

            for message in recent_messages:
                if not isinstance(message, dict):
                    continue

                # Skip system/HITL messages
                role = message.get("role", "")
                if role == "system":
                    continue

                message_type = message.get("message_type", "")
                if "hitl" in message_type.lower() or "checkpoint" in message_type.lower():
                    continue

                content = message.get("content", "")
                if not isinstance(content, str):
                    continue

                # Try to match each field name
                for field_name in field_names:
                    # Skip if we already found this field
                    if field_name in extracted_values:
                        continue

                    # Normalize field name for matching (handle snake_case variations)
                    field_variations = [
                        field_name,
                        field_name.replace('_', ' '),  # aspect_ratio → aspect ratio
                        field_name.replace('_', '-'),  # aspect_ratio → aspect-ratio
                    ]

                    for field_variant in field_variations:
                        # Patterns to match: field=value, field: value, field | value
                        import re
                        # Case-insensitive matching
                        patterns = [
                            rf'\b{re.escape(field_variant)}\s*=\s*([^\s,;]+)',  # field=value
                            rf'\b{re.escape(field_variant)}\s*:\s*([^\s,;]+)',  # field: value
                            rf'\b{re.escape(field_variant)}\s*\|\s*([^\s,;]+)',  # field | value
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, content, re.IGNORECASE)
                            if match:
                                raw_value = match.group(1).strip()

                                # Parse the value
                                parsed_value = self._parse_field_value(raw_value)

                                if parsed_value is not None:
                                    extracted_values[field_name] = parsed_value
                                    print(f"📚 Extracted from conversation: {field_name}={parsed_value}")
                                    break  # Found match, no need to check other patterns

                        if field_name in extracted_values:
                            break  # Found match, no need to check other variations

            return extracted_values

        except Exception as e:
            print(f"⚠️ Error extracting field values from conversation: {e}")
            return {}

    def _parse_field_value(self, raw_value: str) -> Any:
        """Parse a string value into appropriate Python type."""
        raw_value = raw_value.strip().rstrip(',;.')

        # Try boolean
        if raw_value.lower() in ('true', 'yes', 'on', '1'):
            return True
        if raw_value.lower() in ('false', 'no', 'off', '0'):
            return False

        # Try integer
        try:
            return int(raw_value)
        except ValueError:
            pass

        # Try float
        try:
            return float(raw_value)
        except ValueError:
            pass

        # Return as string (remove quotes if present)
        if raw_value.startswith('"') and raw_value.endswith('"'):
            return raw_value[1:-1]
        if raw_value.startswith("'") and raw_value.endswith("'"):
            return raw_value[1:-1]

        return raw_value

    def _suggest_payload_error_fixes(self, error: Exception) -> list:
        """Suggest fixes for payload creation/validation errors"""
        fixes = []
        error_str = str(error).lower()
        
        if "missing" in error_str:
            fixes.append("Provide all required parameters")
            fixes.append("Check model documentation for required fields")
        
        if "invalid" in error_str or "format" in error_str:
            fixes.append("Verify input format matches model requirements")
            fixes.append("Check file type and size limits")
        
        if "authentication" in error_str or "api" in error_str:
            fixes.append("Verify API credentials and permissions")
            fixes.append("Check model availability and access")
        
        if not fixes:
            fixes.append("Review model configuration and try again")
            fixes.append("Contact support if the issue persists")
        
        return fixes
