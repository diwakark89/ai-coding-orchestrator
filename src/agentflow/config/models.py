"""Configuration data models for AgentFlow."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _expand_path(v: Path | str) -> Path:
    if isinstance(v, str):
        v = Path(v)
    return v.expanduser()


ExpandedPath = Annotated[Path, Field(validate_default=True)]


class CLICommandConfig(BaseModel):
    """Configuration for an individual provider CLI."""

    model_config = ConfigDict(extra="ignore")
    command: str


class CLIsConfig(BaseModel):
    """Configuration for all supported provider CLIs."""

    model_config = ConfigDict(extra="ignore")
    claude: CLICommandConfig = Field(default_factory=lambda: CLICommandConfig(command="claude"))
    codex: CLICommandConfig = Field(default_factory=lambda: CLICommandConfig(command="codex"))
    gemini: CLICommandConfig = Field(default_factory=lambda: CLICommandConfig(command="gemini"))

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


class ProjectConfig(BaseModel):
    """Project-level routing configuration (.ai-orchestrator/routing.yaml)."""

    model_config = ConfigDict(extra="allow")
    version: int = 1
    project: ProjectMetaConfig

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported configuration version: {v}. Expected version 1.")
        return v
