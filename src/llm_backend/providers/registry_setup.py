"""
Registry setup and provider registration
"""

from llm_backend.core.providers.registry import ProviderRegistry
from llm_backend.providers.replicate_provider import ReplicateProvider
from llm_backend.core.types.common import AgentTools


def register_providers():
    """Register all available providers with the registry"""
    
    # Register Replicate provider
    ProviderRegistry.register(
        name="replicate",
        provider_class=ReplicateProvider,
        tool_enum=AgentTools.REPLICATETOOL
    )
    
    # Future providers can be registered here:
    # ProviderRegistry.register("openai", OpenAIProvider, AgentTools.OPENAITOOL)
    # ProviderRegistry.register("anthropic", AnthropicProvider, AgentTools.ANTHROPICTOOL)


# Auto-register providers when module is imported
register_providers()
