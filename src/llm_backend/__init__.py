"""
llm_backend package initialization
Auto-register providers on import
"""

# Register all providers at module import time
from llm_backend.providers import registry_setup  # noqa: F401
