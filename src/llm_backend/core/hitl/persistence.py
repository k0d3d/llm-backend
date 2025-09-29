"""
HITL state persistence layer with database and Redis support
"""

import uuid
import redis
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlparse
from abc import ABC, abstractmethod

from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from llm_backend.core.hitl.types import HITLState, StepEvent

Base = declarative_base()
logger = logging.getLogger(__name__)


def _serialize_json(value: Any) -> Any:
    """Recursively convert values into JSON-serializable structures."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, uuid.UUID):
        return str(value)
    if hasattr(value, "model_dump"):
        return _serialize_json(value.model_dump())
    if isinstance(value, dict):
        return {key: _serialize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_json(item) for item in value]
    return value


def _enum_value(value: Any) -> Any:
    """Return `.value` for Enum instances, otherwise the value itself."""
    if isinstance(value, Enum):
        return value.value
    return value


def _uuid_value(value: Any) -> Optional[uuid.UUID]:
    """Ensure values are stored as UUID objects when supported."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str):
        return uuid.UUID(value)
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        logger.warning("Unable to coerce value to UUID", extra={"value": value})
        return None


class HITLRun(Base):
    """Database model for HITL runs"""
    __tablename__ = 'hitl_runs'
    
    run_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(50), nullable=False)
    current_step = Column(String(50), nullable=False)
    provider_name = Column(String(100), nullable=False)
    session_id = Column(String(255), nullable=True, index=True)  # Critical for session-based queries
    user_id = Column(String(255), nullable=True, index=True)     # For user-based filtering
    original_input = Column(JSON, nullable=False)
    hitl_config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Step artifacts
    capabilities = Column(JSON, nullable=True)
    suggested_payload = Column(JSON, nullable=True)
    validation_issues = Column(JSON, nullable=True)
    raw_response = Column(JSON, nullable=True)
    processed_response = Column(Text, nullable=True)
    final_result = Column(Text, nullable=True)
    
    # Human interactions
    pending_actions = Column(JSON, nullable=True)
    approval_token = Column(String(255), nullable=True)
    checkpoint_context = Column(JSON, nullable=True)
    last_approval = Column(JSON, nullable=True)
    
    # Metrics
    total_execution_time_ms = Column(Integer, default=0)
    human_review_time_ms = Column(Integer, default=0)
    provider_execution_time_ms = Column(Integer, default=0)


class HITLStepEvent(Base):
    """Database model for HITL step events"""
    __tablename__ = 'hitl_step_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(PGUUID(as_uuid=True), nullable=False)
    step = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    actor = Column(String(255), nullable=False)
    message = Column(Text, nullable=True)
    event_metadata = Column(JSON, nullable=True)


class HITLPendingApproval(Base):
    """Database model for pending HITL approvals"""
    __tablename__ = 'hitl_pending_approvals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    approval_id = Column(String(36), unique=True, nullable=False, index=True)
    run_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    checkpoint_type = Column(String(100), nullable=False)
    context = Column(JSON, nullable=False)
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    status = Column(String(50), default='pending', index=True)  # pending, responded, expired, cancelled


class HITLApproval(Base):
    """Database model for HITL approvals"""
    __tablename__ = 'hitl_approvals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False)
    approval_id = Column(String(255), unique=True, nullable=False)
    checkpoint_type = Column(String(100), nullable=False)
    context = Column(JSON, nullable=False)
    response = Column(JSON, nullable=True)
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)


class HITLStateStore(ABC):
    """Abstract base class for HITL state storage"""
    
    @abstractmethod
    async def save_state(self, state: HITLState) -> None:
        pass
    
    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        pass
    
    @abstractmethod
    async def delete_state(self, run_id: str) -> None:
        pass
    
    @abstractmethod
    async def list_active_runs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def save_pending_approval(self, approval_data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    async def load_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def remove_pending_approval(self, approval_id: str) -> None:
        pass
    
    @abstractmethod
    async def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def cleanup_expired_approvals(self) -> int:
        pass


class RedisStateStore(HITLStateStore):
    """Redis-based state store for fast access"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
    
    async def save_state(self, state: HITLState) -> None:
        """Save state to Redis with TTL"""
        key = f"hitl:state:{state.run_id}"
        await self.redis.setex(key, self.ttl, state.json())
    
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        """Load state from Redis"""
        key = f"hitl:state:{run_id}"
        data = await self.redis.get(key)
        return HITLState.parse_raw(data) if data else None
    
    async def delete_state(self, run_id: str) -> None:
        """Delete state from Redis"""
        key = f"hitl:state:{run_id}"
        await self.redis.delete(key)
    
    async def list_active_runs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active runs from Redis"""
        pattern = "hitl:state:*"
        keys = await self.redis.keys(pattern)
        
        runs = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                state = HITLState.parse_raw(data)
                if not user_id or state.original_input.get("user_id") == user_id:
                    runs.append({
                        "run_id": state.run_id,
                        "status": state.status,
                        "current_step": state.current_step,
                        "created_at": state.created_at.isoformat(),
                        "expires_at": state.expires_at.isoformat() if state.expires_at else None
                    })
        
        return runs
    
    async def save_pending_approval(self, approval_data: Dict[str, Any]) -> None:
        """Save pending approval to Redis with TTL"""
        key = f"hitl:approval:{approval_data['approval_id']}"
        await self.redis.setex(key, self.ttl, json.dumps(approval_data))
    
    async def load_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Load pending approval from Redis"""
        key = f"hitl:approval:{approval_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def remove_pending_approval(self, approval_id: str) -> None:
        """Remove pending approval from Redis"""
        key = f"hitl:approval:{approval_id}"
        await self.redis.delete(key)
    
    async def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pending approvals from Redis"""
        pattern = "hitl:approval:*"
        keys = await self.redis.keys(pattern)
        
        approvals = []
        for key in keys:
            data = await self.redis.get(key)
            if data:
                approval = json.loads(data)
                if not user_id or approval.get("user_id") == user_id:
                    approvals.append(approval)
        
        return approvals
    
    async def cleanup_expired_approvals(self) -> int:
        """Redis handles expiration automatically"""
        return 0  # Redis TTL handles cleanup


class DatabaseStateStore(HITLStateStore):
    """Database-based state store for persistence"""
    
    def __init__(self, database_url: str):
        # Fix for Dokku/older systems that provide postgres:// instead of postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    async def save_state(self, state: HITLState) -> None:
        """Save state to database"""
        session = self.SessionLocal()
        try:
            # Check if run exists
            run_uuid = _uuid_value(state.run_id)
            existing_run = session.query(HITLRun).filter(HITLRun.run_id == run_uuid).first()
            
            if existing_run:
                # Update existing run
                existing_run.status = _enum_value(state.status)
                existing_run.current_step = _enum_value(state.current_step)
                existing_run.updated_at = state.updated_at
                existing_run.expires_at = state.expires_at
                existing_run.capabilities = _serialize_json(state.capabilities)
                existing_run.suggested_payload = _serialize_json(state.suggested_payload)
                existing_run.validation_issues = _serialize_json(state.validation_issues)
                existing_run.raw_response = _serialize_json(state.raw_response)
                existing_run.processed_response = state.processed_response
                existing_run.final_result = _serialize_json(state.final_result)
                existing_run.session_id = state.original_input.get("session_id")
                existing_run.user_id = state.original_input.get("user_id")
                existing_run.pending_actions = _serialize_json(state.pending_actions)
                existing_run.approval_token = state.approval_token
                existing_run.checkpoint_context = _serialize_json(state.checkpoint_context)
                existing_run.last_approval = _serialize_json(state.last_approval)
                existing_run.total_execution_time_ms = state.total_execution_time_ms
                existing_run.human_review_time_ms = state.human_review_time_ms
                existing_run.provider_execution_time_ms = state.provider_execution_time_ms
            else:
                # Create new run
                new_run = HITLRun(
                    run_id=run_uuid or uuid.uuid4(),
                    status=_enum_value(state.status),
                    current_step=_enum_value(state.current_step),
                    provider_name=state.original_input.get("provider", "unknown"),
                    session_id=state.original_input.get("session_id"),
                    user_id=state.original_input.get("user_id"),
                    original_input=_serialize_json(state.original_input),
                    hitl_config=_serialize_json(state.config),
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    expires_at=state.expires_at,
                    capabilities=_serialize_json(state.capabilities),
                    suggested_payload=_serialize_json(state.suggested_payload),
                    validation_issues=_serialize_json(state.validation_issues),
                    raw_response=_serialize_json(state.raw_response),
                    processed_response=_serialize_json(state.processed_response),
                    final_result=_serialize_json(state.final_result),
                    pending_actions=_serialize_json(state.pending_actions),
                    approval_token=state.approval_token,
                    checkpoint_context=_serialize_json(state.checkpoint_context),
                    last_approval=_serialize_json(state.last_approval),
                    total_execution_time_ms=state.total_execution_time_ms,
                    human_review_time_ms=state.human_review_time_ms,
                    provider_execution_time_ms=state.provider_execution_time_ms
                )
                session.add(new_run)
            
            # Save step events
            for event in state.step_history:
                existing_event = session.query(HITLStepEvent).filter(
                    HITLStepEvent.run_id == run_uuid,
                    HITLStepEvent.step == _enum_value(event.step),
                    HITLStepEvent.timestamp == event.timestamp
                ).first()
                
                if not existing_event:
                    new_event = HITLStepEvent(
                        run_id=run_uuid,
                        step=_enum_value(event.step),
                        status=_enum_value(event.status),
                        timestamp=event.timestamp,
                        actor=event.actor,
                        message=event.message,
                        event_metadata=_serialize_json(event.metadata)
                    )
                    session.add(new_event)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    async def pause_run(self, run_id: str, checkpoint_type: str, context: Dict[str, Any]) -> None:
        """Pause run and save checkpoint context"""
        state = await self.load_state(run_id)
        if state:
            from llm_backend.core.hitl.types import HITLStatus
            state.status = HITLStatus.AWAITING_HUMAN
            state.updated_at = datetime.utcnow()
            # Add checkpoint context to state
            state.checkpoint_context = {
                "type": checkpoint_type,
                "context": context,
                "paused_at": datetime.utcnow().isoformat()
            }
            await self.save_state(state)
            print(f"✅ DatabaseStateStore: Paused run {run_id} at checkpoint {checkpoint_type}")
    
    async def resume_run(self, run_id: str, approval_response: Dict[str, Any]) -> HITLState:
        """Resume run with human approval/edits"""
        state = await self.load_state(run_id)
        if state:
            from llm_backend.core.hitl.types import HITLStatus
            state.status = HITLStatus.RUNNING
            state.updated_at = datetime.utcnow()
            # Store approval response
            state.last_approval = approval_response
            await self.save_state(state)
            print(f"✅ DatabaseStateStore: Resumed run {run_id}")
        return state
    
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        """Load state from database"""
        session = self.SessionLocal()
        try:
            run = session.query(HITLRun).filter(HITLRun.run_id == run_id).first()
            if not run:
                return None
            
            # Load step events
            events = session.query(HITLStepEvent).filter(
                HITLStepEvent.run_id == run_id
            ).order_by(HITLStepEvent.timestamp).all()
            
            step_history = [
                StepEvent(
                    step=event.step,
                    status=event.status,
                    timestamp=event.timestamp,
                    actor=event.actor,
                    message=event.message,
                    metadata=event.event_metadata or {}
                )
                for event in events
            ]
            
            # Reconstruct HITLState
            from llm_backend.core.hitl.types import HITLConfig
            
            return HITLState(
                run_id=str(run.run_id),
                current_step=run.current_step,
                status=run.status,
                created_at=run.created_at,
                updated_at=run.updated_at,
                expires_at=run.expires_at,
                config=HITLConfig(**run.hitl_config),
                original_input=run.original_input,
                capabilities=run.capabilities,
                suggested_payload=run.suggested_payload,
                validation_issues=run.validation_issues or [],
                raw_response=run.raw_response,
                processed_response=run.processed_response,
                final_result=run.final_result,
                pending_actions=run.pending_actions or [],
                approval_token=run.approval_token,
                checkpoint_context=run.checkpoint_context,
                last_approval=run.last_approval,
                step_history=step_history,
                total_execution_time_ms=run.total_execution_time_ms,
                human_review_time_ms=run.human_review_time_ms,
                provider_execution_time_ms=run.provider_execution_time_ms
            )
            
        finally:
            session.close()
    
    async def delete_state(self, run_id: str) -> None:
        """Delete state from database"""
        session = self.SessionLocal()
        try:
            # Delete step events first
            session.query(HITLStepEvent).filter(HITLStepEvent.run_id == run_id).delete()
            
            # Delete approvals
            session.query(HITLApproval).filter(HITLApproval.run_id == run_id).delete()
            
            # Delete run
            session.query(HITLRun).filter(HITLRun.run_id == run_id).delete()
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    async def list_active_runs(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active runs from database with session_id support"""
        session = self.SessionLocal()
        try:
            query = session.query(HITLRun).filter(
                HITLRun.status.in_(["queued", "running", "awaiting_human", "paused"])
            )
            
            # Session ID is more important than user ID for filtering
            if session_id:
                query = query.filter(HITLRun.session_id == session_id)
            elif user_id:
                query = query.filter(HITLRun.user_id == user_id)
            
            runs = query.all()
            
            return [
                {
                    "run_id": str(run.run_id),
                    "status": run.status,
                    "current_step": run.current_step,
                    "provider_name": run.provider_name,
                    "session_id": run.session_id,
                    "user_id": run.user_id,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat(),
                    "expires_at": run.expires_at.isoformat() if run.expires_at else None,
                    "pending_actions": run.pending_actions
                }
                for run in runs
            ]
            
        finally:
            session.close()
    
    async def save_pending_approval(self, approval_data: Dict[str, Any]) -> None:
        """Save pending approval to database"""
        session = self.SessionLocal()
        try:
            logger.debug("Saving pending approval", extra={
                "approval_id": approval_data['approval_id'],
                "run_id": approval_data['run_id'],
                "checkpoint_type": approval_data['checkpoint_type']
            })
            pending_approval = HITLPendingApproval(
                approval_id=approval_data['approval_id'],
                run_id=_uuid_value(approval_data['run_id']),
                checkpoint_type=approval_data['checkpoint_type'],
                context=_serialize_json(approval_data['context']),
                user_id=approval_data.get('user_id'),
                session_id=approval_data.get('session_id'),
                expires_at=datetime.fromisoformat(approval_data['expires_at'])
            )
            session.add(pending_approval)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to save pending approval", exc_info=e)
            raise e
        finally:
            session.close()
    
    async def load_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Load pending approval from database"""
        session = self.SessionLocal()
        try:
            approval = session.query(HITLPendingApproval).filter(
                HITLPendingApproval.approval_id == approval_id,
                HITLPendingApproval.status == 'pending'
            ).first()
            
            if approval:
                return {
                    'approval_id': approval.approval_id,
                    'run_id': approval.run_id,
                    'checkpoint_type': approval.checkpoint_type,
                    'context': approval.context,
                    'user_id': approval.user_id,
                    'session_id': approval.session_id,
                    'created_at': approval.created_at.isoformat(),
                    'expires_at': approval.expires_at.isoformat()
                }
            return None
        finally:
            session.close()
    
    async def remove_pending_approval(self, approval_id: str) -> None:
        """Remove pending approval from database"""
        session = self.SessionLocal()
        try:
            approval = session.query(HITLPendingApproval).filter(
                HITLPendingApproval.approval_id == approval_id
            ).first()
            
            if approval:
                approval.status = 'responded'
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    async def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pending approvals from database"""
        session = self.SessionLocal()
        try:
            query = session.query(HITLPendingApproval).filter(
                HITLPendingApproval.status == 'pending',
                HITLPendingApproval.expires_at > datetime.utcnow()
            )
            
            if user_id:
                query = query.filter(HITLPendingApproval.user_id == user_id)
            
            approvals = query.all()
            
            return [{
                'approval_id': approval.approval_id,
                'run_id': approval.run_id,
                'checkpoint_type': approval.checkpoint_type,
                'context': approval.context,
                'user_id': approval.user_id,
                'session_id': approval.session_id,
                'created_at': approval.created_at.isoformat(),
                'expires_at': approval.expires_at.isoformat()
            } for approval in approvals]
        finally:
            session.close()
    
    async def cleanup_expired_approvals(self) -> int:
        """Clean up expired approvals and return count removed"""
        session = self.SessionLocal()
        try:
            count = session.query(HITLPendingApproval).filter(
                HITLPendingApproval.status == 'pending',
                HITLPendingApproval.expires_at <= datetime.utcnow()
            ).update({'status': 'expired'})
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


def create_state_manager(database_url: str) -> HITLStateStore:
    """
    Create appropriate state manager based on database URL scheme
    
    Args:
        database_url: Database connection URL
                     - redis://... -> Redis-only storage
                     - postgresql://... -> PostgreSQL-only storage
                     - hybrid://redis_url,postgres_url -> Hybrid storage
    
    Returns:
        Appropriate state manager instance
    """
    print(f"🔧 Creating state manager for URL: {database_url[:50]}...")
    parsed_url = urlparse(database_url)
    print(f"🔧 Parsed URL scheme: {parsed_url.scheme}")
    
    if parsed_url.scheme == 'redis':
        manager = RedisStateStore(database_url)
        print("✅ Created RedisStateStore")
        return manager
    elif parsed_url.scheme in ('postgresql', 'postgres'):
        manager = DatabaseStateStore(database_url)
        print("✅ Created DatabaseStateStore")
        return manager
    elif parsed_url.scheme == 'hybrid':
        # Format: hybrid://redis_url,postgres_url
        urls = parsed_url.netloc.split(',')
        if len(urls) != 2:
            raise ValueError("Hybrid URL must contain exactly two URLs separated by comma")
        redis_url = f"redis://{urls[0]}"
        postgres_url = f"postgresql://{urls[1]}"
        manager = HybridStateManager(
            RedisStateStore(redis_url),
            DatabaseStateStore(postgres_url)
        )
        print("✅ Created HybridStateManager")
        return manager
    else:
        raise ValueError(f"Unsupported database scheme: {parsed_url.scheme}")


class HybridStateManager:
    """Hybrid state manager using both Redis and Database"""
    
    def __init__(self, redis_store: RedisStateStore, db_store: DatabaseStateStore):
        self.redis_store = redis_store
        self.db_store = db_store
    
    async def save_state(self, state: HITLState) -> None:
        """Save to both Redis (fast access) and Database (persistence)"""
        await self.redis_store.save_state(state)
        await self.db_store.save_state(state)
    
    async def load_state(self, run_id: str) -> Optional[HITLState]:
        """Load from Redis first, fallback to Database"""
        # Try Redis first for speed
        state = await self.redis_store.load_state(run_id)
        if state:
            return state
        
        # Fallback to database
        state = await self.db_store.load_state(run_id)
        if state:
            # Cache in Redis for future access
            await self.redis_store.save_state(state)
        
        return state
    
    async def delete_state(self, run_id: str) -> None:
        """Delete from both stores"""
        await self.redis_store.delete_state(run_id)
        await self.db_store.delete_state(run_id)
    
    async def list_active_runs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active runs from database (source of truth)"""
        return await self.db_store.list_active_runs(user_id)
    
    async def save_pending_approval(self, approval_data: Dict[str, Any]) -> None:
        """Save pending approval to database"""
        await self.db_store.save_pending_approval(approval_data)
    
    async def load_pending_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Load pending approval from database"""
        return await self.db_store.load_pending_approval(approval_id)
    
    async def remove_pending_approval(self, approval_id: str) -> None:
        """Remove pending approval from database"""
        await self.db_store.remove_pending_approval(approval_id)
    
    async def list_pending_approvals(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pending approvals from database"""
        return await self.db_store.list_pending_approvals(user_id)
    
    async def cleanup_expired_approvals(self) -> int:
        """Clean up expired approvals and return count removed"""
        return await self.db_store.cleanup_expired_approvals()
    
    async def pause_run(self, run_id: str, checkpoint_type: str, context: Dict[str, Any]) -> None:
        """Pause run and save checkpoint context"""
        state = await self.load_state(run_id)
        if state:
            from llm_backend.core.hitl.types import HITLStatus
            state.status = HITLStatus.AWAITING_HUMAN
            state.updated_at = datetime.utcnow()
            # Add checkpoint context to metadata
            if not hasattr(state, 'checkpoint_context'):
                state.checkpoint_context = {}
            state.checkpoint_context = {
                "type": checkpoint_type,
                "context": context,
                "paused_at": datetime.utcnow().isoformat()
            }
            await self.save_state(state)
    
    async def resume_run(self, run_id: str, approval_response: Dict[str, Any]) -> HITLState:
        """Resume run with human approval/edits"""
        state = await self.load_state(run_id)
        if state:
            from llm_backend.core.hitl.types import HITLStatus
            state.status = HITLStatus.RUNNING
            state.updated_at = datetime.utcnow()
            # Store approval response
            if not hasattr(state, 'last_approval'):
                state.last_approval = {}
            state.last_approval = approval_response
            await self.save_state(state)
        
        return state
