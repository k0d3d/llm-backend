"""
Core HITL functionality tests focusing on session resumability and persistence

This simplified test suite verifies the essential HITL features:
1. Database state persistence
2. Session resumability 
3. Checkpoint context preservation
4. User/session isolation
"""

import pytest
import uuid
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend.core.hitl.persistence import DatabaseStateStore
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep, HITLState, StepEvent


@pytest.fixture
def db_store():
    """Database state store for testing"""
    return DatabaseStateStore("sqlite:///:memory:")


class TestHITLCoreFunctionality:
    """Test core HITL functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_state_persistence(self, db_store):
        """Test basic state save and load functionality"""
        run_id = str(uuid.uuid4())
        session_id = "test_session_123"
        user_id = "test_user_456"
        
        # Create test state
        original_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "Remove background from image",
                "session_id": session_id,
                "user_id": user_id
            },
            validation_issues=[
                {
                    "field": "input_image",
                    "severity": "error", 
                    "issue": "Required parameter missing"
                }
            ],
            pending_actions=["upload_image", "approve"]
        )
        
        # Save state
        await db_store.save_state(original_state)
        
        # Load state
        loaded_state = await db_store.load_state(run_id)
        
        # Verify state persistence
        assert loaded_state is not None
        assert loaded_state.run_id == run_id
        assert loaded_state.current_step == HITLStep.INFORMATION_REVIEW
        assert loaded_state.status == HITLStatus.AWAITING_HUMAN
        assert loaded_state.original_input["session_id"] == session_id
        assert loaded_state.original_input["user_id"] == user_id
        assert len(loaded_state.validation_issues) == 1
        assert "upload_image" in loaded_state.pending_actions
        
        print("✅ Basic state persistence test passed")
    
    @pytest.mark.asyncio
    async def test_session_pause_resume_cycle(self, db_store):
        """Test complete pause and resume cycle"""
        run_id = str(uuid.uuid4())
        session_id = "pause_resume_session"
        user_id = "pause_resume_user"
        
        # Create initial running state
        initial_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.PAYLOAD_REVIEW,
            status=HITLStatus.RUNNING,
            config=HITLConfig(),
            original_input={
                "prompt": "Process this request",
                "session_id": session_id,
                "user_id": user_id
            }
        )
        
        # Save initial state
        await db_store.save_state(initial_state)
        
        # Pause the run with checkpoint context
        checkpoint_context = {
            "requires_confirmation": True,
            "confidence_score": 0.75,
            "user_message": "Please review and approve this action"
        }
        
        await db_store.pause_run(run_id, "human_approval_required", checkpoint_context)
        
        # Verify paused state
        paused_state = await db_store.load_state(run_id)
        assert paused_state.status == HITLStatus.AWAITING_HUMAN
        assert paused_state.checkpoint_context is not None
        assert paused_state.checkpoint_context["type"] == "human_approval_required"
        assert "requires_confirmation" in paused_state.checkpoint_context["context"]
        
        # Resume with approval response
        approval_response = {
            "action": "approve",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "modifications": {}
        }
        
        resumed_state = await db_store.resume_run(run_id, approval_response)
        
        # Verify resumed state
        assert resumed_state.status == HITLStatus.RUNNING
        assert resumed_state.last_approval is not None
        assert resumed_state.last_approval["action"] == "approve"
        
        print("✅ Session pause/resume cycle test passed")
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, db_store):
        """Test that different sessions are properly isolated"""
        # Create multiple sessions
        sessions = [
            {"session_id": "session_1", "user_id": "user_1", "run_id": str(uuid.uuid4())},
            {"session_id": "session_2", "user_id": "user_2", "run_id": str(uuid.uuid4())},
            {"session_id": "session_1", "user_id": "user_3", "run_id": str(uuid.uuid4())},  # Same session, different user
        ]
        
        # Create states for each session
        for session in sessions:
            state = HITLState(
                run_id=session["run_id"],
                current_step=HITLStep.INFORMATION_REVIEW,
                status=HITLStatus.AWAITING_HUMAN,
                config=HITLConfig(),
                original_input={
                    "prompt": f"Request for {session['session_id']}",
                    "session_id": session["session_id"],
                    "user_id": session["user_id"]
                }
            )
            await db_store.save_state(state)
        
        # Test session-based filtering
        session_1_runs = await db_store.list_active_runs(session_id="session_1")
        session_2_runs = await db_store.list_active_runs(session_id="session_2")
        
        # Verify isolation
        assert len(session_1_runs) == 2  # Two runs for session_1
        assert len(session_2_runs) == 1  # One run for session_2
        
        # Verify correct runs are returned
        session_1_run_ids = [run["run_id"] for run in session_1_runs]
        assert sessions[0]["run_id"] in session_1_run_ids
        assert sessions[2]["run_id"] in session_1_run_ids
        assert sessions[1]["run_id"] not in session_1_run_ids
        
        print("✅ Session isolation test passed")
    
    @pytest.mark.asyncio
    async def test_checkpoint_context_preservation(self, db_store):
        """Test that checkpoint context is properly preserved for resumability"""
        run_id = str(uuid.uuid4())
        
        # Create state with rich checkpoint context
        complex_context = {
            "type": "file_validation",
            "context": {
                "requires_file_input": True,
                "file_type": "image",
                "validation_rules": {
                    "max_size": "10MB",
                    "allowed_formats": ["jpg", "png", "webp"],
                    "min_dimensions": {"width": 100, "height": 100}
                },
                "user_friendly_message": "Please upload an image file",
                "auto_discovery": {
                    "enabled": True,
                    "search_chat_history": True,
                    "fallback_to_prompt": False
                }
            },
            "paused_at": datetime.utcnow().isoformat()
        }
        
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "Process this image",
                "session_id": "checkpoint_test_session",
                "user_id": "checkpoint_test_user"
            },
            checkpoint_context=complex_context,
            validation_issues=[
                {
                    "field": "input_image",
                    "severity": "error",
                    "issue": "Required file input missing",
                    "suggested_action": "upload_file"
                }
            ]
        )
        
        # Save and reload state
        await db_store.save_state(state)
        loaded_state = await db_store.load_state(run_id)
        
        # Verify complex checkpoint context preservation
        assert loaded_state.checkpoint_context is not None
        assert loaded_state.checkpoint_context["type"] == "file_validation"
        
        context = loaded_state.checkpoint_context["context"]
        assert context["requires_file_input"] is True
        assert context["file_type"] == "image"
        assert "validation_rules" in context
        assert context["validation_rules"]["max_size"] == "10MB"
        assert "jpg" in context["validation_rules"]["allowed_formats"]
        assert context["auto_discovery"]["enabled"] is True
        
        # Verify validation issues are preserved
        assert len(loaded_state.validation_issues) == 1
        assert loaded_state.validation_issues[0]["field"] == "input_image"
        assert loaded_state.validation_issues[0]["suggested_action"] == "upload_file"
        
        print("✅ Checkpoint context preservation test passed")
    
    @pytest.mark.asyncio
    async def test_step_history_tracking(self, db_store):
        """Test that step history is properly tracked for resumability"""
        run_id = str(uuid.uuid4())
        
        # Create state with step history
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.PAYLOAD_REVIEW,
            status=HITLStatus.RUNNING,
            config=HITLConfig(),
            original_input={
                "prompt": "Multi-step process",
                "session_id": "history_test_session",
                "user_id": "history_test_user"
            },
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow() - timedelta(minutes=5),
                    actor="system",
                    message="HITL run created"
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.AWAITING_HUMAN,
                    timestamp=datetime.utcnow() - timedelta(minutes=3),
                    actor="system",
                    message="Awaiting human review",
                    metadata={"confidence": 0.8}
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.RUNNING,
                    timestamp=datetime.utcnow() - timedelta(minutes=1),
                    actor="user_123",
                    message="Human approved with modifications",
                    metadata={"action": "approve", "modifications": {"param1": "new_value"}}
                )
            ]
        )
        
        # Save and reload state
        await db_store.save_state(state)
        loaded_state = await db_store.load_state(run_id)
        
        # Verify step history preservation
        assert len(loaded_state.step_history) == 3
        
        # Check first step
        first_step = loaded_state.step_history[0]
        assert first_step.step == HITLStep.CREATED
        assert first_step.status == HITLStatus.QUEUED
        assert first_step.actor == "system"
        
        # Check last step with metadata
        last_step = loaded_state.step_history[2]
        assert last_step.step == HITLStep.INFORMATION_REVIEW
        assert last_step.actor == "user_123"
        assert last_step.metadata is not None
        assert last_step.metadata["action"] == "approve"
        assert "modifications" in last_step.metadata
        
        print("✅ Step history tracking test passed")
    
    @pytest.mark.asyncio
    async def test_database_state_reflects_resumability(self, db_store):
        """Test that database state properly reflects all resumability requirements"""
        run_id = str(uuid.uuid4())
        session_id = "resumability_test_session"
        user_id = "resumability_test_user"
        
        # Create comprehensive state for resumability testing
        comprehensive_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(
                policy="require_human",
                timeout_seconds=3600,
                auto_approve_confidence_threshold=0.9
            ),
            original_input={
                "prompt": "Complex multi-step task requiring human oversight",
                "session_id": session_id,
                "user_id": user_id,
                "metadata": {"priority": "high", "category": "image_processing"}
            },
            validation_issues=[
                {
                    "field": "input_image",
                    "severity": "error",
                    "issue": "Required parameter 'input_image' is missing",
                    "suggested_action": "upload_file"
                },
                {
                    "field": "output_format",
                    "severity": "warning", 
                    "issue": "Output format not specified, using default",
                    "suggested_action": "specify_format"
                }
            ],
            pending_actions=["upload_image", "specify_format", "approve"],
            checkpoint_context={
                "type": "file_validation",
                "context": {
                    "requires_file_input": True,
                    "file_type": "image",
                    "user_friendly_message": "Please upload an image file to continue"
                },
                "paused_at": datetime.utcnow().isoformat()
            },
            capabilities={
                "models": ["background-removal", "image-enhancement"],
                "input_types": ["image"],
                "output_types": ["image"],
                "features": {"batch_processing": False, "real_time": True}
            },
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow() - timedelta(minutes=2),
                    actor="system",
                    message="HITL run created and queued"
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.AWAITING_HUMAN,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Paused for human input - file upload required"
                )
            ]
        )
        
        # Save comprehensive state
        await db_store.save_state(comprehensive_state)
        
        # Load and verify all resumability data is preserved
        loaded_state = await db_store.load_state(run_id)
        
        # Verify core resumability data
        assert loaded_state.run_id == run_id
        assert loaded_state.original_input["session_id"] == session_id
        assert loaded_state.original_input["user_id"] == user_id
        assert loaded_state.status == HITLStatus.AWAITING_HUMAN
        assert loaded_state.current_step == HITLStep.INFORMATION_REVIEW
        
        # Verify checkpoint context for resumability
        assert loaded_state.checkpoint_context is not None
        assert loaded_state.checkpoint_context["type"] == "file_validation"
        assert loaded_state.checkpoint_context["context"]["requires_file_input"] is True
        
        # Verify validation issues for user guidance
        assert len(loaded_state.validation_issues) == 2
        error_issue = next(issue for issue in loaded_state.validation_issues if issue["severity"] == "error")
        assert error_issue["field"] == "input_image"
        assert error_issue["suggested_action"] == "upload_file"
        
        # Verify pending actions for resumability
        assert "upload_image" in loaded_state.pending_actions
        assert "approve" in loaded_state.pending_actions
        
        # Verify step history for context
        assert len(loaded_state.step_history) == 2
        assert loaded_state.step_history[0].step == HITLStep.CREATED
        assert loaded_state.step_history[1].step == HITLStep.INFORMATION_REVIEW
        
        # Verify capabilities for provider context
        assert "background-removal" in loaded_state.capabilities["models"]
        assert loaded_state.capabilities["features"]["real_time"] is True
        
        # Test session-based query works
        session_runs = await db_store.list_active_runs(session_id=session_id)
        assert len(session_runs) == 1
        assert session_runs[0]["run_id"] == run_id
        assert session_runs[0]["status"] == HITLStatus.AWAITING_HUMAN.value
        
        print("✅ Database state reflects resumability requirements test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
