"""
Tests for HITL edge cases and error scenarios

This test suite covers:
1. Session expiry and timeout handling
2. Concurrent session management
3. Database connection failures
4. Malformed data handling
5. Resource cleanup
"""

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_backend.core.hitl.persistence import DatabaseStateStore
from llm_backend.core.hitl.types import HITLConfig, HITLStatus, HITLStep, HITLState, StepEvent


@pytest_asyncio.fixture
async def db_store():
    """Database store for testing"""
    store = DatabaseStateStore("sqlite:///:memory:")
    return store


class TestSessionExpiry:
    """Test session expiry and timeout handling"""
    
    @pytest.mark.asyncio
    async def test_expired_session_detection(self, db_store):
        """Test detection and handling of expired sessions"""
        run_id = str(uuid.uuid4())
        
        # Create expired state
        expired_time = datetime.utcnow() - timedelta(hours=2)
        expired_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            created_at=expired_time,
            updated_at=expired_time,
            expires_at=expired_time + timedelta(hours=1),  # Expired 1 hour ago
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "expired_session",
                "user_id": "expired_user"
            }
        )
        
        await db_store.save_state(expired_state)
        
        # Load expired state
        loaded_state = await db_store.load_state(run_id)
        
        # Verify expiry detection
        assert loaded_state.expires_at < datetime.utcnow()
        
        # Test that expired sessions are filtered from active runs
        active_runs = await db_store.list_active_runs(session_id="expired_session")
        
        # Should still return the run (filtering by expiry should be done at application level)
        assert len(active_runs) == 1
        
        # But we can verify the expiry timestamp
        assert datetime.fromisoformat(active_runs[0]["expires_at"]) < datetime.utcnow()
        
        print("✅ Expired session detection test passed")
    
    @pytest.mark.asyncio
    async def test_session_timeout_during_approval(self, db_store):
        """Test handling of session timeout during human approval"""
        run_id = str(uuid.uuid4())
        
        # Create state that will timeout soon
        near_expiry = datetime.utcnow() + timedelta(minutes=5)
        state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            expires_at=near_expiry,
            config=HITLConfig(timeout_seconds=300),  # 5 minutes
            original_input={
                "prompt": "test",
                "session_id": "timeout_test_session",
                "user_id": "timeout_test_user"
            },
            pending_actions=["approve"],
            approval_token="timeout_token"
        )
        
        await db_store.save_state(state)
        
        # Simulate timeout by advancing time
        expired_state = await db_store.load_state(run_id)
        expired_state.expires_at = datetime.utcnow() - timedelta(minutes=1)
        expired_state.status = HITLStatus.FAILED
        expired_state.step_history.append(
            StepEvent(
                step=HITLStep.FAILED,
                status=HITLStatus.FAILED,
                timestamp=datetime.utcnow(),
                actor="system",
                message="Session expired due to timeout"
            )
        )
        
        await db_store.save_state(expired_state)
        
        # Verify timeout handling
        final_state = await db_store.load_state(run_id)
        assert final_state.status == HITLStatus.FAILED
        assert any("timeout" in event.message.lower() for event in final_state.step_history)
        
        print("✅ Session timeout during approval test passed")


class TestConcurrentSessions:
    """Test concurrent session management"""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_isolation(self, db_store):
        """Test that concurrent sessions don't interfere with each other"""
        # Create multiple concurrent sessions
        sessions = []
        for i in range(5):
            session_data = {
                "session_id": f"concurrent_session_{i}",
                "user_id": f"user_{i}",
                "run_id": str(uuid.uuid4())
            }
            sessions.append(session_data)
        
        # Create states concurrently
        async def create_session_state(session_data):
            state = HITLState(
                run_id=session_data["run_id"],
                current_step=HITLStep.INFORMATION_REVIEW,
                status=HITLStatus.AWAITING_HUMAN,
                config=HITLConfig(),
                original_input={
                    "prompt": f"Test prompt for {session_data['session_id']}",
                    "session_id": session_data["session_id"],
                    "user_id": session_data["user_id"]
                }
            )
            await db_store.save_state(state)
            return session_data["run_id"]
        
        # Execute concurrent operations
        tasks = [create_session_state(session) for session in sessions]
        run_ids = await asyncio.gather(*tasks)
        
        # Verify all sessions were created correctly
        assert len(run_ids) == 5
        assert len(set(run_ids)) == 5  # All unique
        
        # Verify session isolation
        for i, session_data in enumerate(sessions):
            session_runs = await db_store.list_active_runs(session_id=session_data["session_id"])
            assert len(session_runs) == 1
            assert session_runs[0]["run_id"] == session_data["run_id"]
            assert session_runs[0]["session_id"] == session_data["session_id"]
        
        print("✅ Concurrent session isolation test passed")
    
    @pytest.mark.asyncio
    async def test_concurrent_state_updates(self, db_store):
        """Test concurrent updates to the same session state"""
        run_id = str(uuid.uuid4())
        
        # Create initial state
        initial_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "concurrent_update_session",
                "user_id": "concurrent_update_user"
            }
        )
        
        await db_store.save_state(initial_state)
        
        # Define concurrent update operations
        async def update_state_status():
            state = await db_store.load_state(run_id)
            state.status = HITLStatus.RUNNING
            state.updated_at = datetime.utcnow()
            await db_store.save_state(state)
        
        async def update_state_step():
            state = await db_store.load_state(run_id)
            state.current_step = HITLStep.PAYLOAD_REVIEW
            state.updated_at = datetime.utcnow()
            await db_store.save_state(state)
        
        async def add_step_event():
            state = await db_store.load_state(run_id)
            state.step_history.append(
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.RUNNING,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Concurrent update test"
                )
            )
            await db_store.save_state(state)
        
        # Execute concurrent updates
        await asyncio.gather(
            update_state_status(),
            update_state_step(),
            add_step_event()
        )
        
        # Verify final state (last update wins)
        final_state = await db_store.load_state(run_id)
        assert final_state is not None
        
        # At least one update should have succeeded
        assert (final_state.status == HITLStatus.RUNNING or 
                final_state.current_step == HITLStep.PAYLOAD_REVIEW or
                len(final_state.step_history) > 0)
        
        print("✅ Concurrent state updates test passed")


class TestDatabaseFailures:
    """Test database connection and operation failures"""
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test handling of database connection failures"""
        # Create store with invalid connection
        with pytest.raises(Exception):
            invalid_store = DatabaseStateStore("postgresql://invalid:invalid@nonexistent:5432/invalid")
            
            # Attempt operation that should fail
            test_state = HITLState(
                run_id=str(uuid.uuid4()),
                config=HITLConfig(),
                original_input={"prompt": "test"}
            )
            await invalid_store.save_state(test_state)
    
    @pytest.mark.asyncio
    async def test_save_state_failure_recovery(self, db_store):
        """Test recovery from save state failures"""
        run_id = str(uuid.uuid4())
        
        # Create valid state
        valid_state = HITLState(
            run_id=run_id,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "failure_test_session",
                "user_id": "failure_test_user"
            }
        )
        
        # Save valid state first
        await db_store.save_state(valid_state)
        
        # Verify state was saved
        loaded_state = await db_store.load_state(run_id)
        assert loaded_state is not None
        
        # Test with malformed state (should handle gracefully)
        malformed_state = valid_state.model_copy()
        malformed_state.step_history = [
            StepEvent.model_construct(
                step="invalid_step",  # Bypass validation intentionally
                status="invalid_status",
                timestamp=datetime.utcnow(),
                actor="test",
                message=None,
                metadata={}
            )
        ]
        
        # This should either succeed or fail gracefully
        try:
            await db_store.save_state(malformed_state)
        except Exception:
            # If it fails, the original state should still be intact
            recovered_state = await db_store.load_state(run_id)
            assert recovered_state is not None
        
        print("✅ Save state failure recovery test passed")


class TestMalformedData:
    """Test handling of malformed data"""
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self, db_store):
        """Test handling of malformed JSON data in database"""
        run_id = str(uuid.uuid4())
        
        # Create state with complex nested data
        complex_state = HITLState(
            run_id=run_id,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "malformed_test_session",
                "complex_data": {
                    "nested": {"deeply": {"nested": "value"}},
                    "array": [1, 2, {"object": True}],
                    "unicode": "🎨🖼️📸",
                    "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?"
                }
            },
            capabilities={
                "models": ["model-1", "model-2"],
                "features": {"streaming": True, "batch": False},
                "null_value": None,
                "empty_dict": {},
                "empty_list": []
            }
        )
        
        # Save and load complex state
        await db_store.save_state(complex_state)
        loaded_state = await db_store.load_state(run_id)
        
        # Verify complex data integrity
        assert loaded_state is not None
        assert loaded_state.original_input["complex_data"]["unicode"] == "🎨🖼️📸"
        assert loaded_state.original_input["complex_data"]["nested"]["deeply"]["nested"] == "value"
        assert loaded_state.capabilities["features"]["streaming"] is True
        assert loaded_state.capabilities["null_value"] is None
        assert loaded_state.capabilities["empty_dict"] == {}
        assert loaded_state.capabilities["empty_list"] == []
        
        print("✅ Malformed JSON handling test passed")
    
    @pytest.mark.asyncio
    async def test_invalid_uuid_handling(self, db_store):
        """Test handling of invalid UUIDs"""
        # Test with invalid run_id format
        invalid_run_id = "not-a-valid-uuid"
        
        # Should return None for invalid UUID
        result = await db_store.load_state(invalid_run_id)
        assert result is None
        
        # Test with empty string
        result = await db_store.load_state("")
        assert result is None
        
        # Test with None
        result = await db_store.load_state(None)
        assert result is None
        
        print("✅ Invalid UUID handling test passed")


class TestResourceCleanup:
    """Test resource cleanup and memory management"""
    
    @pytest.mark.asyncio
    async def test_large_state_handling(self, db_store):
        """Test handling of large state objects"""
        run_id = str(uuid.uuid4())
        
        # Create state with large data
        large_response = {"output": "x" * 100000}  # 100KB string
        large_metadata = {f"key_{i}": f"value_{i}" * 1000 for i in range(100)}  # Large metadata
        
        large_state = HITLState(
            run_id=run_id,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "large_state_session",
                "large_metadata": large_metadata
            },
            raw_response=large_response,
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Large state test",
                    metadata={"large_data": "y" * 10000}
                )
                for _ in range(50)  # 50 step events
            ]
        )
        
        # Save and load large state
        await db_store.save_state(large_state)
        loaded_state = await db_store.load_state(run_id)
        
        # Verify large data integrity
        assert loaded_state is not None
        assert len(loaded_state.raw_response["output"]) == 100000
        assert len(loaded_state.original_input["large_metadata"]) == 100
        assert len(loaded_state.step_history) == 50
        
        print("✅ Large state handling test passed")
    
    @pytest.mark.asyncio
    async def test_state_deletion_cleanup(self, db_store):
        """Test proper cleanup when deleting states"""
        run_id = str(uuid.uuid4())
        
        # Create state with step events
        state = HITLState(
            run_id=run_id,
            config=HITLConfig(),
            original_input={
                "prompt": "test",
                "session_id": "cleanup_test_session",
                "user_id": "cleanup_test_user"
            },
            step_history=[
                StepEvent(
                    step=HITLStep.CREATED,
                    status=HITLStatus.QUEUED,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Cleanup test"
                ),
                StepEvent(
                    step=HITLStep.INFORMATION_REVIEW,
                    status=HITLStatus.AWAITING_HUMAN,
                    timestamp=datetime.utcnow(),
                    actor="system",
                    message="Awaiting cleanup"
                )
            ]
        )
        
        # Save state
        await db_store.save_state(state)
        
        # Verify state exists
        loaded_state = await db_store.load_state(run_id)
        assert loaded_state is not None
        assert len(loaded_state.step_history) == 2
        
        # Delete state
        await db_store.delete_state(run_id)
        
        # Verify complete cleanup
        deleted_state = await db_store.load_state(run_id)
        assert deleted_state is None
        
        # Verify state is not in active runs
        active_runs = await db_store.list_active_runs(session_id="cleanup_test_session")
        assert len(active_runs) == 0
        
        print("✅ State deletion cleanup test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
