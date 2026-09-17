"""Configuration data models for AgentFlow."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentflow.routing.complexity import ComplexityConfig
from agentflow.routing.rules import (
    DEFAULT_MODELS_CONFIG,
    ModelsConfig,
    RoutingRulesConfig,
    all_referenced_aliases,
)


def _expand_path(v: Path | str) -> Path:
    if isinstance(v, str):
        v = Path(v)
    return v.expanduser()


ExpandedPath = Annotated[Path, Field(validate_default=True)]


class CLICommandConfig(BaseModel):
    """Configuration for an individual provider CLI."""

    model_config = ConfigDict(extra="ignore")
    command: str


class GoogleCLIConfig(CLICommandConfig):
    """Configuration for the Google-provider CLI.

    `command` may point at the real `gemini` CLI or at Google's newer `agy` (Antigravity) CLI --
    their non-interactive flag syntax differs enough to need distinct adapters, selected by
    `dialect` rather than guessed from the binary's filename.
    """

    dialect: Literal["gemini", "antigravity"] = "gemini"


class CLIsConfig(BaseModel):
    """Configuration for all supported provider CLIs."""

    model_config = ConfigDict(extra="ignore")
    claude: CLICommandConfig = Field(default_factory=lambda: CLICommandConfig(command="claude"))
    codex: CLICommandConfig = Field(default_factory=lambda: CLICommandConfig(command="codex"))
    gemini: GoogleCLIConfig = Field(default_factory=lambda: GoogleCLIConfig(command="gemini"))

    @field_validator("claude", mode="before")
    @classmethod
    def _validate_claude(cls, v: object) -> object:
        if isinstance(v, dict) and "command" not in v:
            return {**v, "command": "claude"}
        return v

    @field_validator("codex", mode="before")
    @classmethod
    def _validate_codex(cls, v: object) -> object:
        if isinstance(v, dict) and "command" not in v:
            return {**v, "command": "codex"}
        return v

    @field_validator("gemini", mode="before")
    @classmethod
    def _validate_gemini(cls, v: object) -> object:
        if isinstance(v, dict) and "command" not in v:
            return {**v, "command": "gemini"}
        return v


class StorageConfig(BaseModel):
    """Global storage paths."""

    model_config = ConfigDict(extra="ignore")
    database: Path = Field(default_factory=lambda: Path("~/.agentflow/agentflow.db").expanduser())

    @field_validator("database", mode="before")
    @classmethod
    def _validate_database_path(cls, v: Path | str) -> Path:
        return _expand_path(v)


class WorktreesConfig(BaseModel):
    """Worktree root directory path."""

    model_config = ConfigDict(extra="ignore")
    root: Path = Field(default_factory=lambda: Path("~/.agentflow/worktrees").expanduser())

    @field_validator("root", mode="before")
    @classmethod
    def _validate_worktrees_path(cls, v: Path | str) -> Path:
        return _expand_path(v)


class LoggingConfig(BaseModel):
    """Logging directory path."""

    model_config = ConfigDict(extra="ignore")
    root: Path = Field(default_factory=lambda: Path("~/.agentflow/logs").expanduser())

    @field_validator("root", mode="before")
    @classmethod
    def _validate_logging_path(cls, v: Path | str) -> Path:
        return _expand_path(v)


class GlobalConfig(BaseModel):
    """Global AgentFlow machine configuration (~/.agentflow/config.yaml)."""

    model_config = ConfigDict(extra="ignore")
    version: int = 1
    cli: CLIsConfig = Field(default_factory=CLIsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    worktrees: WorktreesConfig = Field(default_factory=WorktreesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported configuration version: {v}. Expected version 1.")
        return v


class ProjectMetaConfig(BaseModel):
    """Project metadata within routing.yaml."""

    model_config = ConfigDict(extra="allow")
    name: str


class VerificationGroup(BaseModel):
    """A named verification group: files whose presence enables it, and commands to run."""

    model_config = ConfigDict(extra="ignore")
    detect: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)


class LimitsConfig(BaseModel):
    """Bounded-retry limits preventing infinite agent loops (TDD §33)."""

    model_config = ConfigDict(extra="ignore")
    planning_turns: int = 20
    implementation_attempts: int = 3
    lightweight_verification_failures: int = 2
    standard_failures: int = 2
    review_fix_cycles: int = 2


class DocumentationConfig(BaseModel):
    """Project documentation update rules (routing.yaml `documentation:` section)."""

    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    candidate_files: list[str] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Project-level routing configuration (.ai-orchestrator/routing.yaml)."""

    model_config = ConfigDict(extra="allow")
    version: int = 1
    project: ProjectMetaConfig

    models: ModelsConfig | None = None
    complexity: ComplexityConfig | None = None
    routing: RoutingRulesConfig | None = None
    verification: dict[str, VerificationGroup] | None = None
    limits: LimitsConfig | None = None
    documentation: DocumentationConfig | None = None

    # The §23-style nested escalation graph (Phase 7+) has no dedicated schema yet;
    # accepted here as opaque data so routing.yaml can declare it without being rejected.
    escalation: dict[str, Any] | None = None

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported configuration version: {v}. Expected version 1.")
        return v

    @model_validator(mode="after")
    def _validate_routing_model_aliases(self) -> "ProjectConfig":
        """Reject routing rules that reference an alias undefined in the models: section."""
        if self.routing is None:
            return self
        # No models: section configured; fall back to V1 defaults for alias resolution.
        models = self.models or DEFAULT_MODELS_CONFIG
        for alias in all_referenced_aliases(self.routing):
            models.resolve(alias)
        return self
