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

# Create shared state manager and websocket bridge
shared_state_manager = create_state_manager(DATABASE_URL)
shared_websocket_bridge = WebSocketHITLBridge(
    WEBSOCKET_URL, WEBSOCKET_API_KEY, shared_state_manager
)


def get_shared_state_manager():
    """Get the shared state manager instance"""
    return shared_state_manager


def get_shared_websocket_bridge():
    """Get the shared websocket bridge instance"""
    return shared_websocket_bridge
