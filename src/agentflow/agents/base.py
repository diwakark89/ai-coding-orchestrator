"""Provider-independent AI agent adapter abstractions and contracts."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentflow.errors import UnsupportedCapabilityError
from agentflow.process.executor import ProcessExecutor, ProcessResult

EXCLUDED_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.4-mini",
        "gpt-5.4 mini",
        "gpt-5.6-sol",
        "gpt-5.6 sol",
    }
)


def validate_model_allowed(model: str) -> None:
    """Validate that the requested model is not in the forbidden V1 model pool."""
    cleaned = model.strip().lower()
    if cleaned in EXCLUDED_MODELS:
        raise ValueError(
            f"Model '{model}' is explicitly excluded from AgentFlow V1. "
            f"Forbidden models: {sorted(EXCLUDED_MODELS)}"
        )


class Provider(str, Enum):
    """Supported AI agent CLI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

    @classmethod
    def from_string(cls, value: str) -> "Provider":
        """Parse provider from string case-insensitively."""
        clean = value.strip().lower()
        for item in cls:
            if item.value == clean or item.name.lower() == clean:
                return item
        raise ValueError(f"Unknown provider: '{value}'. Expected one of: {[p.value for p in cls]}")


class AgentRole(str, Enum):
    """Canonical agent workflow roles."""

    PLANNER = "planner"
    DEFAULT_PLANNER = "planner.default"
    ARCHITECTURE_PLANNER = "planner.architecture"

    IMPLEMENTER = "implementer"
    LIGHTWEIGHT_CODER = "implementation.lightweight"
    STANDARD_CODER = "implementation.standard"
    IMPLEMENTATION_ESCALATION = "implementation.escalation"

    REVIEWER = "reviewer"
    DEFAULT_REVIEWER = "review.default"
    DEEP_REVIEWER = "review.deep"
    ARCHITECTURE_REVIEWER = "review.architecture"

    DOCUMENTER = "documentation.default"


def _expand_path(v: Path | str | None) -> Path | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = Path(v)
    return v.expanduser()


class AgentRequest(BaseModel):
    """Provider-independent request contract for invoking an AI agent CLI."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    role: AgentRole | str
    prompt: str
    repository_path: Path
    working_directory: Path | None = None
    model: str
    read_only: bool = False
    timeout_seconds: float | None = None

    # Provider-specific settings live in an extensible structure
    provider_options: dict[str, Any] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent prompt cannot be empty.")
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        validate_model_allowed(v)
        return v

    @field_validator("repository_path", "working_directory", mode="before")
    @classmethod
    def _validate_paths(cls, v: Any) -> Any:
        return _expand_path(v)

    @property
    def worktree_path(self) -> Path | None:
        """Alias for working_directory."""
        return self.working_directory

    @property
    def effective_working_directory(self) -> Path:
        """Return working_directory if provided, otherwise repository_path."""
        return self.working_directory or self.repository_path


class AgentResult(BaseModel):
    """Provider-independent execution result contract from an AI agent CLI."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Provider
    model: str
    session_id: str | None = None
    exit_code: int
    text: str = ""
    raw_events: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime
    timed_out: bool = False
    stderr: str = ""

    @property
    def success(self) -> bool:
        """Return True if agent completed with exit code 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def duration_seconds(self) -> float:
        """Return total elapsed execution time in seconds."""
        return (self.completed_at - self.started_at).total_seconds()


class AdapterCapabilities(BaseModel):
    """Metadata describing capabilities supported by a specific CLI adapter."""

    model_config = ConfigDict(extra="ignore")

    supports_resume: bool = False
    supports_read_only_mode: bool = False
    supports_structured_output: bool = False
    supports_model_selection: bool = True


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol defining the standard interface for all provider CLI adapters."""

    @property
    def provider(self) -> Provider:
        """The provider enum handled by this adapter."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Capability flags supported by this adapter."""
        ...

    async def start(self, request: AgentRequest) -> AgentResult:
        """Execute an initial agent task."""
        ...

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        """Resume an existing agent session with a follow-up request."""
        ...


class BaseAgentAdapter(ABC):
    """Base class for provider CLI adapters providing execution scaffolding."""

    def __init__(
        self,
        command: str,
        executor: ProcessExecutor | None = None,
    ) -> None:
        self.command = command
        self.executor = executor or ProcessExecutor()

    @property
    @abstractmethod
    def provider(self) -> Provider:
        """Provider enum identifier."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Reported capability metadata."""
        ...

    @abstractmethod
    def build_args(self, request: AgentRequest, session_id: str | None = None) -> list[str]:
        """Build argument array for subprocess execution without shell concatenation."""
        ...

    @abstractmethod
    def parse_result(
        self,
        proc_res: ProcessResult,
        request: AgentRequest,
        session_id: str | None = None,
    ) -> AgentResult:
        """Parse raw process result into canonical AgentResult."""
        ...

    async def start(self, request: AgentRequest) -> AgentResult:
        """Start a new agent session for the given request."""
        validate_model_allowed(request.model)
        cmd_args = self.build_args(request, session_id=None)
        cwd = request.effective_working_directory

        proc_res = await self.executor.run(
            cmd_args=cmd_args,
            cwd=cwd,
            timeout=request.timeout_seconds,
        )
        return self.parse_result(proc_res, request, session_id=None)

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        """Resume an existing session if supported by the provider."""
        if not self.capabilities.supports_resume:
            raise UnsupportedCapabilityError(
                f"{self.provider.value.title()} CLI adapter does not support session resume."
            )
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty when resuming a session.")

        validate_model_allowed(request.model)
        cmd_args = self.build_args(request, session_id=session_id)
        cwd = request.effective_working_directory

        proc_res = await self.executor.run(
            cmd_args=cmd_args,
            cwd=cwd,
            timeout=request.timeout_seconds,
        )
        return self.parse_result(proc_res, request, session_id=session_id)
