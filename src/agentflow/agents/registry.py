"""Adapter registry mapping providers to configured CLI adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentflow.agents.base import AgentAdapter, Provider
from agentflow.agents.claude import ClaudeAdapter
from agentflow.agents.codex import CodexAdapter
from agentflow.agents.gemini import GeminiAdapter
from agentflow.errors import AdapterNotFoundError
from agentflow.process.executor import ProcessExecutor

if TYPE_CHECKING:
    # config.models imports routing, which imports agents.base -- importing GlobalConfig at
    # runtime here would close that cycle back onto this module. Only needed for typing.
    from agentflow.config.models import GlobalConfig


class AgentAdapterRegistry:
    """Registry managing provider-to-adapter bindings."""

    def __init__(self) -> None:
        self._adapters: dict[Provider, AgentAdapter] = {}

    def register(self, provider: Provider, adapter: AgentAdapter) -> None:
        """Register an adapter for the given provider."""
        self._adapters[provider] = adapter

    def get(self, provider: Provider | str) -> AgentAdapter:
        """Retrieve the adapter for the given provider or raise AdapterNotFoundError."""
        if isinstance(provider, str):
            try:
                provider_enum = Provider.from_string(provider)
            except ValueError as e:
                raise AdapterNotFoundError(str(e)) from e
        else:
            provider_enum = provider

        if provider_enum not in self._adapters:
            raise AdapterNotFoundError(
                f"No adapter registered for provider: '{provider_enum.value}'. "
                f"Registered providers: {[p.value for p in self._adapters.keys()]}"
            )
        return self._adapters[provider_enum]

    def has(self, provider: Provider | str) -> bool:
        """Check if an adapter is registered for the given provider."""
        try:
            if isinstance(provider, str):
                provider_enum = Provider.from_string(provider)
            else:
                provider_enum = provider
            return provider_enum in self._adapters
        except ValueError:
            return False

    @property
    def registered_providers(self) -> list[Provider]:
        """List all currently registered providers."""
        return list(self._adapters.keys())


def create_default_registry(
    config: GlobalConfig | None = None,
    executor: ProcessExecutor | None = None,
) -> AgentAdapterRegistry:
    """Factory creating and populating the standard adapter registry."""
    registry = AgentAdapterRegistry()
    proc_executor = executor or ProcessExecutor()

    claude_cmd = config.cli.claude.command if config else "claude"
    codex_cmd = config.cli.codex.command if config else "codex"
    gemini_cmd = config.cli.gemini.command if config else "gemini"

    registry.register(
        Provider.ANTHROPIC,
        ClaudeAdapter(command=claude_cmd, executor=proc_executor),
    )
    registry.register(
        Provider.OPENAI,
        CodexAdapter(command=codex_cmd, executor=proc_executor),
    )
    registry.register(
        Provider.GOOGLE,
        GeminiAdapter(command=gemini_cmd, executor=proc_executor),
    )

    return registry
