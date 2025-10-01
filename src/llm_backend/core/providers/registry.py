"""
Provider registry for managing AI provider implementations
"""

from typing import Dict, Type, Optional, List, Tuple, Any
from llm_backend.core.providers.base import AIProvider
from llm_backend.core.types.common import AgentTools


class ProviderRegistry:
    """Registry for managing AI provider implementations"""
    
    _providers: Dict[str, Type[AIProvider]] = {}
    _tool_mapping: Dict[AgentTools, str] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[AIProvider], tool_enum: Optional[AgentTools] = None):
        """Register a provider with optional tool mapping"""
        cls._providers[name] = provider_class
        if tool_enum:
            cls._tool_mapping[tool_enum] = name
    
    @classmethod
    def get_provider_for_tool(cls, tool: AgentTools, config: Dict) -> AIProvider:
        """Get provider instance for a specific tool"""
        provider_name = cls._tool_mapping.get(tool)
        if not provider_name:
            raise ValueError(f"No provider registered for tool: {tool}")
        return cls.get_provider(provider_name, config)
    
    @classmethod
    def get_provider(cls, name: str, config: Optional[Dict] = None) -> AIProvider:
        """Get provider instance by name"""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        provider_config = config or {}
        return cls._providers[name](provider_config)

    @classmethod
    def resolve_provider(
        cls,
        agent_tool_config: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Determine provider name and config from an agent tool configuration."""

        if not agent_tool_config:
            return None, {}

        mapping = getattr(cls, "_tool_mapping", {})

        for tool_key, tool_payload in agent_tool_config.items():
            try:
                tool_enum = AgentTools(tool_key)
            except ValueError:
                continue

            provider_name = mapping.get(tool_enum)
            if not provider_name:
                continue

            provider_config: Dict[str, Any] = {}
            if isinstance(tool_payload, dict):
                if isinstance(tool_payload.get("data"), dict):
                    provider_config = tool_payload["data"]
                else:
                    provider_config = tool_payload

            return provider_name, provider_config

        return None, {}
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names"""
        return list(cls._providers.keys())
    
    @classmethod
    def list_tools(cls) -> List[AgentTools]:
        """List all mapped tools"""
        return list(cls._tool_mapping.keys())
