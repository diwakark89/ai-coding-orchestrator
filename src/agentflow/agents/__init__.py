"""Provider-independent AI agent adapter layer."""

from agentflow.agents.base import (
    EXCLUDED_MODELS,
    AdapterCapabilities,
    AgentAdapter,
    AgentRequest,
    AgentResult,
    AgentRole,
    BaseAgentAdapter,
    Provider,
    validate_model_allowed,
)
from agentflow.agents.claude import ClaudeAdapter
from agentflow.agents.codex import CodexAdapter
from agentflow.agents.gemini import GeminiAdapter
from agentflow.agents.registry import (
    AgentAdapterRegistry,
    create_default_registry,
)

__all__ = [
    "EXCLUDED_MODELS",
    "AdapterCapabilities",
    "AgentAdapter",
    "AgentAdapterRegistry",
    "AgentRequest",
    "AgentResult",
    "AgentRole",
    "BaseAgentAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "Provider",
    "create_default_registry",
    "validate_model_allowed",
]
