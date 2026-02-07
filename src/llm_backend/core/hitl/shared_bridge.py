"""
Shared WebSocket bridge instance for HITL workflows
This ensures all endpoints use the same bridge instance for consistent approval handling
"""

import os
from llm_backend.core.hitl.persistence import create_state_manager
from llm_backend.core.hitl.websocket_bridge import WebSocketHITLBridge

# Initialize shared components
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/tohju")
WEBSOCKET_URL = os.getenv("WEBSOCKET_URL", "wss://ws.tohju.com")
WEBSOCKET_API_KEY = os.getenv("WEBSOCKET_API_KEY")

# Global instances for lazy loading
_shared_state_manager = None
_shared_websocket_bridge = None

def get_shared_state_manager():
    """Get or create the shared state manager instance (lazy)"""
    global _shared_state_manager
    if _shared_state_manager is None:
        _shared_state_manager = create_state_manager(DATABASE_URL)
    return _shared_state_manager

def get_shared_websocket_bridge():
    """Get or create the shared websocket bridge instance (lazy)"""
    global _shared_websocket_bridge
    if _shared_websocket_bridge is None:
        state_mgr = get_shared_state_manager()
        _shared_websocket_bridge = WebSocketHITLBridge(WEBSOCKET_URL, WEBSOCKET_API_KEY, state_mgr)
    return _shared_websocket_bridge
