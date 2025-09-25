"""
HITL Orchestrator - Main workflow coordinator
"""

import time
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .types import HITLState, StepEvent, HITLConfig, HITLStep, HITLStatus, HITLPolicy
from .websocket_bridge import WebSocketHITLBridge
from .validation import HITLValidator, create_hitl_validation_summary
from llm_backend.core.providers.base import AIProvider, ProviderResponse
from llm_backend.core.types.common import RunInput


class HITLOrchestrator:
    """Main orchestrator for HITL workflow management"""
    
    def __init__(self, provider: AIProvider, config: HITLConfig, run_input: RunInput):
        self.provider = provider
        self.config = config
        self.run_input = run_input
        self.run_id = str(uuid.uuid4())
        
        # Set provider run input
        self.provider.set_run_input(run_input)
        
        # Initialize state
        self.state = HITLState(
            run_id=self.run_id,
            config=config,
            original_input=run_input.dict()
        )
        
        # Set expiration if configured
        if config.timeout_seconds > 0:
            self.state.expires_at = datetime.utcnow() + timedelta(seconds=config.timeout_seconds)
        
        self._add_step_event(HITLStep.CREATED, HITLStatus.QUEUED, "system", "Run created")
    
    async def execute(self) -> Dict[str, Any]:
        """Main execution flow with HITL checkpoints"""
        print(f"🚀 HITL Orchestrator starting execution for run_id: {self.run_id}")
        
        start_time = time.time()
        
        try:
            # Step 1: Information Review
            result = await self._step_information_review()
            # print(f"✅ Information Review result: {result}")
            if self._is_paused(result):
                print("⏸️ Execution paused at Information Review")
                return result
            
            # Step 2: Payload Review
            result = await self._step_payload_review()
            if self._is_paused(result):
                print("⏸️ Execution paused at Payload Review")
                return result
            
            # Step 3: API Execution
            result = await self._step_api_execution()
            if self._is_paused(result):
                print("⏸️ Execution paused at API Execution")
                return result
            
            # Step 4: Response Review
            result = await self._step_response_review()
            if self._is_paused(result):
                return result
            
            # Step 5: Completion
            return await self._step_completion()
            
        except Exception as e:
            return await self._handle_error(e)
        finally:
            self.state.total_execution_time_ms = int((time.time() - start_time) * 1000)
    
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
            
            # Configure bridge from environment
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
    
    async def _step_information_review(self) -> Dict[str, Any]:
        """Enhanced information review checkpoint with comprehensive validation"""
        self._transition_to_step(HITLStep.INFORMATION_REVIEW)
        
        # Run comprehensive pre-execution validation
        # Extract tool config from the correct key structure
        agent_tool_config = self.run_input.agent_tool_config or {}
        replicate_config = agent_tool_config.get("replicate-agent-tool", {}).get("data", {})
        print(f"🔍 Orchestrator: Using tool_config: {replicate_config}")
        
        validator = HITLValidator(
            run_input=self.run_input,
            tool_config=replicate_config
        )
        
        validation_checkpoints = validator.validate_pre_execution()
        validation_summary = create_hitl_validation_summary(validation_checkpoints)
        
        print(f"🚨 Blocking Issues: {validation_summary['blocking_issues']}")
        
        # Store validation results in state
        self.state.validation_checkpoints = validation_summary
        
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
        needs_review = self._should_pause_at_information_review(capabilities) or blocking_issues
        
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
            
            # Send WebSocket message for HITL approval request
            print("🔄 About to call _send_websocket_message...")
            try:
                await self._send_websocket_message(pause_response)
                print("🔄 _send_websocket_message completed")
            except Exception as ws_error:
                print(f"❌ WebSocket notification failed: {ws_error}")
            return pause_response
        
        self._add_step_event(HITLStep.INFORMATION_REVIEW, HITLStatus.COMPLETED, "system", "Auto-approved based on validation and thresholds")
        return {"continue": True}
    
    async def _step_payload_review(self) -> Dict[str, Any]:
        """Enhanced payload review checkpoint with improved error handling"""
        self._transition_to_step(HITLStep.PAYLOAD_REVIEW)
        
        try:
            # Create payload
            operation_type = self._infer_operation_type()
            payload = self.provider.create_payload(
                prompt=self.run_input.prompt,
                attachments=[self.run_input.document_url] if self.run_input.document_url else [],
                operation_type=operation_type,
                config=self.run_input.agent_tool_config or {}
            )
            
            # Validate payload with enhanced validation
            validation_issues = self.provider.validate_payload(
                payload,
                self.run_input.prompt,
                [self.run_input.document_url] if self.run_input.document_url else []
            )
            
            self.state.suggested_payload = payload.dict()
            self.state.validation_issues = [issue.dict() for issue in validation_issues]
            
            # Check for critical validation failures that require human intervention
            critical_issues = [issue for issue in validation_issues if issue.severity == "error" and not issue.auto_fixable]
            
            if critical_issues:
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
    
    async def _step_api_execution(self) -> Dict[str, Any]:
        """API execution step"""
        self._transition_to_step(HITLStep.API_CALL, HITLStatus.RUNNING)
        
        # Get payload from previous step or state
        # Reconstruct payload object (this would need proper payload class detection)
        payload = self.provider.create_payload(
            prompt=self.run_input.prompt,
            attachments=[self.run_input.document_url] if self.run_input.document_url else [],
            operation_type=self._infer_operation_type(),
            config=self.run_input.agent_tool_config or {}
        )
        
        # Execute with provider
        start_time = time.time()
        response = self.provider.execute(payload)
        execution_time = int((time.time() - start_time) * 1000)
        
        self.state.provider_execution_time_ms = execution_time
        self.state.raw_response = response.raw_response
        self.state.processed_response = response.processed_response
        
        if response.error:
            raise Exception(f"Provider execution failed: {response.error}")
        
        self._add_step_event(HITLStep.API_CALL, HITLStatus.COMPLETED, "system", f"Executed in {execution_time}ms")
        return {"continue": True, "response": response}
    
    async def _step_response_review(self) -> Dict[str, Any]:
        """Response review checkpoint"""
        self._transition_to_step(HITLStep.RESPONSE_REVIEW)
        
        # Audit response
        response = ProviderResponse(
            raw_response=self.state.raw_response,
            processed_response=self.state.processed_response,
            metadata={},
            execution_time_ms=self.state.provider_execution_time_ms
        )
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
        
        elif self.state.current_step == HITLStep.PAYLOAD_REVIEW:
            if "payload" in edits:
                self.state.suggested_payload.update(edits["payload"])
        
        elif self.state.current_step == HITLStep.RESPONSE_REVIEW:
            if "response" in edits:
                self.state.final_result = edits["response"]
        
        self._add_step_event(self.state.current_step, HITLStatus.COMPLETED, actor, f"Edited: {message}")
        
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
        
        # Always pause for errors
        if any(issue.severity == "error" for issue in issues):
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
    
    def _create_pause_response(self, step: HITLStep, message: str, actions_required: list, data: Dict[str, Any] = None) -> Dict[str, Any]:
        self.state.status = HITLStatus.AWAITING_HUMAN
        self.state.pending_actions = actions_required
        self.state.approval_token = str(uuid.uuid4())
        self.state.updated_at = datetime.utcnow()
        
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
        
        if data:
            response.update(data)
        
        return response
    
    def _transition_to_step(self, step: HITLStep, status: HITLStatus = HITLStatus.RUNNING):
        self.state.current_step = step
        self.state.status = status
        self.state.updated_at = datetime.utcnow()
    
    def _add_step_event(self, step: HITLStep, status: HITLStatus, actor: str, message: str):
        event = StepEvent(
            step=step,
            status=status,
            actor=actor,
            message=message
        )
        self.state.step_history.append(event)
    
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
                    
                    elif "image" in issue.field.lower() and self.run_input.document_url:
                        # Map image to appropriate field
                        if hasattr(fixed_payload, 'input') and 'image' in fixed_payload.input:
                            fixed_payload.input['image'] = self.run_input.document_url
                        
                        fix_results.append({
                            "field": issue.field,
                            "fix_applied": f"Mapped uploaded file to {issue.field}",
                            "success": True
                        })
                
                except Exception as e:
                    fix_results.append({
                        "field": issue.field,
                        "fix_applied": f"Failed to auto-fix: {str(e)}",
                        "success": False
                    })
        
        return fixed_payload if fix_results else None, fix_results
    
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
