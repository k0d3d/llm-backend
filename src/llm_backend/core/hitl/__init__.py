"""
HITL (Human-in-the-Loop) orchestration package

This package provides a complete provider-agnostic HITL system for AI model execution workflows.
It includes orchestration, state persistence, WebSocket communication, and browser integration.
"""

from .types import HITLConfig, HITLStatus, HITLStep, HITLState, StepEvent
from .orchestrator import HITLOrchestrator
from .persistence import HybridStateManager, RedisStateStore, DatabaseStateStore
from .websocket_bridge import WebSocketHITLBridge, HITLWebSocketHandler, BrowserAgentHITLIntegration

__all__ = [
    # Types and enums
    "HITLConfig",
    "HITLStatus", 
    "HITLStep",
    "HITLState",
    "StepEvent",
    
    # Core orchestration
    "HITLOrchestrator",
    
    # State persistence
    "HybridStateManager",
    "RedisStateStore", 
    "DatabaseStateStore",
    
    # WebSocket communication
    "WebSocketHITLBridge",
    "HITLWebSocketHandler",
    "BrowserAgentHITLIntegration"
]
