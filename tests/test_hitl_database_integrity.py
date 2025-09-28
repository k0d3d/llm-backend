"""
Tests for HITL database integrity and data consistency

This test suite verifies:
1. Database schema integrity
2. Data consistency across tables
3. Session/user data isolation
4. Checkpoint state accuracy
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend.core.hitl.persistence import DatabaseStateStore, HITLRun, HITLStepEvent, HITLApproval, Base
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep, HITLState, StepEvent


@pytest.fixture
def test_engine():
    """Create test database engine"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(test_engine):
    """Create test database session"""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest_asyncio.fixture
async def db_store():
    """Database store for testing"""
    store = DatabaseStateStore("sqlite:///:memory:")
    return store


class TestDatabaseSchema:
    """Test database schema integrity"""
    
    def test_hitl_runs_table_structure(self, test_session):
        """Test HITLRun table structure and constraints"""
        # Create test run
        run_id = str(uuid.uuid4())
        run = HITLRun(
            run_id=run_id,
            status="queued",
            current_step="created",
            provider_name="test_provider",
            session_id="test_session",
            user_id="test_user",
            original_input={"prompt": "test"},
            hitl_config={"policy": "auto"},
            capabilities={"test": True},
            suggested_payload={"model": "test"},
            validation_issues=[{"field": "test", "issue": "test"}],
            pending_actions=["test_action"],
            approval_token="test_token"
        )
        
        test_session.add(run)
        test_session.commit()
        
        # Verify data integrity
        retrieved_run = test_session.query(HITLRun).filter(HITLRun.run_id == run_id).first()
        assert retrieved_run is not None
        assert retrieved_run.session_id == "test_session"
        assert retrieved_run.user_id == "test_user"
        assert retrieved_run.original_input["prompt"] == "test"
        assert retrieved_run.hitl_config["policy"] == "auto"
        assert len(retrieved_run.validation_issues) == 1
        assert "test_action" in retrieved_run.pending_actions
        
        print("✅ HITLRun table structure test passed")
    
    def test_step_events_table_structure(self, test_session):
        """Test HITLStepEvent table structure"""
        run_id = str(uuid.uuid4())
        
        # Create step events
        events = [
            HITLStepEvent(
                run_id=run_id,
                step="created",
                status="queued",
                actor="system",
                message="Run created",
                event_metadata={"source": "test"}
            ),
            HITLStepEvent(
                run_id=run_id,
                step="information_review",
                status="awaiting_human",
                actor="system",
                message="Validation required",
                event_metadata={"issues": 1}
            )
        ]
        
        for event in events:
            test_session.add(event)
        test_session.commit()
        
        # Verify events
        retrieved_events = test_session.query(HITLStepEvent).filter(
            HITLStepEvent.run_id == run_id
        ).order_by(HITLStepEvent.timestamp).all()
        
        assert len(retrieved_events) == 2
        assert retrieved_events[0].step == "created"
        assert retrieved_events[1].step == "information_review"
        assert retrieved_events[0].event_metadata["source"] == "test"
        assert retrieved_events[1].event_metadata["issues"] == 1
        
        print("✅ HITLStepEvent table structure test passed")
    
    def test_approvals_table_structure(self, test_session):
        """Test HITLApproval table structure"""
        run_id = str(uuid.uuid4())
        approval_id = str(uuid.uuid4())
        
        approval = HITLApproval(
            run_id=run_id,
            approval_id=approval_id,
            checkpoint_type="information_review",
            context={
                "message": "Approval required",
                "validation_issues": [{"field": "test", "severity": "error"}]
            },
            response={
                "action": "approve",
                "approved_by": "test_user"
            },
            approved_by="test_user",
            approved_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        test_session.add(approval)
        test_session.commit()
        
        # Verify approval
        retrieved_approval = test_session.query(HITLApproval).filter(
            HITLApproval.approval_id == approval_id
        ).first()
        
        assert retrieved_approval is not None
        assert retrieved_approval.run_id == run_id
        assert retrieved_approval.checkpoint_type == "information_review"
        assert retrieved_approval.context["message"] == "Approval required"
        assert retrieved_approval.response["action"] == "approve"
        assert retrieved_approval.approved_by == "test_user"
        
        print("✅ HITLApproval table structure test passed")


class TestDataConsistency:
    """Test data consistency across tables"""
    
    @pytest.mark.asyncio
    async def test_run_and_events_consistency(self, db_store):
        """Test consistency between runs and step events"""
        run_id = str(uuid.uuid4())
        
        # Create state with step history
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "consistency_test_session",
                "user_id": "consistency_test_user"
            },
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Run created"
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.AWAITING_HUMAN,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Validation checkpoint"
                )
            ]
        )
        
        # Save state
        await db_store.save_state(state)
        
        # Load state and verify consistency
        loaded_state = await db_store.load_state(run_id)
        
        # Verify run data matches step events
        assert loaded_state.current_step == HITLStep.INFORMATION_REVIEW
        assert loaded_state.status == HITLStatus.AWAITING_HUMAN
        assert len(loaded_state.step_history) == 2
        
        # Verify last step event matches current state
        last_event = loaded_state.step_history[-1]
        assert last_event.step == loaded_state.current_step
        assert last_event.status == loaded_state.status
        
        print("✅ Run and events consistency test passed")
    
    @pytest.mark.asyncio
    async def test_session_data_isolation(self, db_store):
        """Test that session data is properly isolated"""
        # Create multiple runs for different sessions
        sessions_data = [
            {"session_id": "session_a", "user_id": "user_1", "prompt": "Session A prompt"},
            {"session_id": "session_b", "user_id": "user_2", "prompt": "Session B prompt"},
            {"session_id": "session_c", "user_id": "user_1", "prompt": "Session C prompt"}
        ]
        
        run_ids = []
        for session_data in sessions_data:
            run_id = str(uuid.uuid4())
            run_ids.append(run_id)
            
            state = HITLState(
                run_id=run_id,
                current_step=HITLStep.INFORMATION_REVIEW,
                status=HITLStatus.AWAITING_HUMAN,
                config=HITLConfig(),
                original_input=session_data
            )
            
            await db_store.save_state(state)
        
        # Test session isolation
        session_a_runs = await db_store.list_active_runs(session_id="session_a")
        session_b_runs = await db_store.list_active_runs(session_id="session_b")
        session_c_runs = await db_store.list_active_runs(session_id="session_c")
        
        assert len(session_a_runs) == 1
        assert len(session_b_runs) == 1
        assert len(session_c_runs) == 1
        
        # Verify session data integrity
        assert session_a_runs[0]["session_id"] == "session_a"
        assert session_b_runs[0]["session_id"] == "session_b"
        assert session_c_runs[0]["session_id"] == "session_c"
        
        # Test user-based queries
        user_1_runs = await db_store.list_active_runs(user_id="user_1")
        user_2_runs = await db_store.list_active_runs(user_id="user_2")
        
        assert len(user_1_runs) == 2  # session_a and session_c
        assert len(user_2_runs) == 1   # session_b
        
        # Verify no cross-contamination
        user_1_sessions = {run["session_id"] for run in user_1_runs}
        user_2_sessions = {run["session_id"] for run in user_2_runs}
        
        assert "session_a" in user_1_sessions
        assert "session_c" in user_1_sessions
        assert "session_b" not in user_1_sessions
        
        assert "session_b" in user_2_sessions
        assert "session_a" not in user_2_sessions
        assert "session_c" not in user_2_sessions
        
        print("✅ Session data isolation test passed")


class TestCheckpointStateAccuracy:
    """Test that checkpoint states accurately reflect resumability requirements"""
    
    @pytest.mark.asyncio
    async def test_checkpoint_context_preservation(self, db_store):
        """Test that checkpoint context is preserved for resumability"""
        run_id = str(uuid.uuid4())
        
        # Create state with detailed checkpoint context
        checkpoint_context = {
            "type": "file_validation",
            "context": {
                "message": "Image file required for background removal",
                "validation_issues": [
                    {
                        "field": "input_image",
                        "severity": "error",
                        "issue": "Required parameter 'input_image' is missing or empty",
                        "suggested_fix": "Upload an image file (JPG, PNG, or WebP)"
                    }
                ],
                "requires_file_input": True,
                "file_type": "image",
                "supported_formats": ["jpg", "png", "webp"],
                "max_file_size": 10485760,
                "confidence_score": 0.2,
                "blocking_issues": 1,
                "user_friendly_message": "I need an image file to remove the background. Please upload a JPG, PNG, or WebP image."
            },
            "paused_at": datetime.utcnow().isoformat()
        }
        
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "Remove background from image",
                "session_id": "checkpoint_test_session",
                "user_id": "checkpoint_test_user"
            },
            validation_issues=[
                {
                    "field": "input_image",
                    "severity": "error",
                    "issue": "Required parameter 'input_image' is missing or empty"
                }
            ],
            pending_actions=["upload_image", "approve"],
            checkpoint_context=checkpoint_context
        )
        
        # Save state
        await db_store.save_state(state)
        
        # Load and verify checkpoint context
        loaded_state = await db_store.load_state(run_id)
        
        assert loaded_state.checkpoint_context is not None
        assert loaded_state.checkpoint_context["type"] == "file_validation"
        
        context = loaded_state.checkpoint_context["context"]
        assert context["requires_file_input"] is True
        assert context["file_type"] == "image"
        assert context["blocking_issues"] == 1
        assert "JPG, PNG, or WebP" in context["user_friendly_message"]
        assert len(context["validation_issues"]) == 1
        assert context["validation_issues"][0]["field"] == "input_image"
        
        # Verify pending actions match checkpoint requirements
        assert "upload_image" in loaded_state.pending_actions
        assert "approve" in loaded_state.pending_actions
        
        print("✅ Checkpoint context preservation test passed")
    
    @pytest.mark.asyncio
    async def test_resumability_state_completeness(self, db_store):
        """Test that all data needed for resumability is preserved"""
        run_id = str(uuid.uuid4())
        
        # Create comprehensive state for resumability testing
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.PAYLOAD_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=2),
            config=HITLConfig(
                policy="require_human",
                timeout_seconds=7200,
                auto_approve_confidence_threshold=0.8
            ),
            original_input={
                "prompt": "Generate creative content",
                "session_id": "resumability_test_session",
                "user_id": "resumability_test_user",
                "agent_tool_config": {
                    "REPLICATETOOL": {
                        "data": {"model": "creative-model"}
                    }
                }
            },
            capabilities={
                "supported_models": ["creative-model"],
                "max_tokens": 4096,
                "supports_streaming": True
            },
            suggested_payload={
                "model": "creative-model",
                "prompt": "Generate creative content",
                "max_tokens": 2048,
                "temperature": 0.7
            },
            validation_issues=[
                {
                    "field": "temperature",
                    "severity": "warning",
                    "issue": "Temperature value is high",
                    "suggested_fix": "Consider reducing to 0.5 for more consistent results"
                }
            ],
            pending_actions=["review_temperature", "approve_payload"],
            approval_token="resumability_token_456",
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow() - timedelta(minutes=5),
                    actor="system",
                    message="Run created"
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.RUNNING,
                    timestamp=datetime.utcnow() - timedelta(minutes=4),
                    actor="system",
                    message="Information review completed"
                ),
                StepEvent(
                    step=HITLStep.PAYLOAD_REVIEW,
                    status=HITLStatus.AWAITING_HUMAN,
                    timestamp=datetime.utcnow() - timedelta(minutes=1),
                    actor="system",
                    message="Payload review required"
                )
            ],
            total_execution_time_ms=240000,  # 4 minutes
            human_review_time_ms=60000       # 1 minute
        )
        
        # Save comprehensive state
        await db_store.save_state(state)
        
        # Load and verify all resumability data
        loaded_state = await db_store.load_state(run_id)
        
        # Verify core resumability fields
        assert loaded_state.run_id == run_id
        assert loaded_state.current_step == HITLStep.PAYLOAD_REVIEW
        assert loaded_state.status == HITLStatus.AWAITING_HUMAN
        assert loaded_state.expires_at is not None
        
        # Verify original input preservation
        assert loaded_state.original_input["prompt"] == "Generate creative content"
        assert loaded_state.original_input["session_id"] == "resumability_test_session"
        assert "REPLICATETOOL" in loaded_state.original_input["agent_tool_config"]
        
        # Verify workflow artifacts
        assert loaded_state.capabilities["max_tokens"] == 4096
        assert loaded_state.suggested_payload["temperature"] == 0.7
        assert len(loaded_state.validation_issues) == 1
        assert "temperature" in loaded_state.validation_issues[0]["field"]
        
        # Verify human interaction state
        assert "review_temperature" in loaded_state.pending_actions
        assert loaded_state.approval_token == "resumability_token_456"
        
        # Verify step history completeness
        assert len(loaded_state.step_history) == 3
        assert loaded_state.step_history[0].step == HITLStep.CREATED
        assert loaded_state.step_history[-1].step == HITLStep.PAYLOAD_REVIEW
        
        # Verify timing data
        assert loaded_state.total_execution_time_ms == 240000
        assert loaded_state.human_review_time_ms == 60000
        
        print("✅ Resumability state completeness test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
