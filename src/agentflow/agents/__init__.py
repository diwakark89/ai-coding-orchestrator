"""Provider-independent AI agent adapter layer."""

from agentflow.agents.antigravity import AntigravityAdapter
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
from agentflow.agents.parser import (
    extract_json_block,
    extract_outer_json_string,
    parse_json_lines,
    parse_model_into_schema,
    parse_structured_json,
    strip_markdown_fences,
)
from agentflow.agents.registry import (
    AgentAdapterRegistry,
    create_default_registry,
)
from agentflow.errors import StructuredParsingError

__all__ = [
    "EXCLUDED_MODELS",
    "AdapterCapabilities",
    "AgentAdapter",
    "AgentAdapterRegistry",
    "AgentRequest",
    "AgentResult",
    "AgentRole",
    "AntigravityAdapter",
    "BaseAgentAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "Provider",
    "StructuredParsingError",
    "create_default_registry",
    "extract_json_block",
    "extract_outer_json_string",
    "parse_json_lines",
    "parse_model_into_schema",
    "parse_structured_json",
    "strip_markdown_fences",
    "validate_model_allowed",
]
