"""Configuration loader functions for global and project configurations."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentflow.config.models import GlobalConfig, ProjectConfig
from agentflow.errors import ConfigurationError

DEFAULT_GLOBAL_CONFIG_PATH = Path("~/.agentflow/config.yaml")


def get_default_global_config_path() -> Path:
    """Return the resolved path to the global configuration file."""
    env_override = os.environ.get("AGENTFLOW_CONFIG_PATH")
    if env_override:
        return Path(env_override).expanduser()
    return DEFAULT_GLOBAL_CONFIG_PATH.expanduser()


def load_global_config(path: Path | str | None = None) -> GlobalConfig:
    """Load and validate the global configuration file.

    If the file does not exist, returns sensible defaults without automatically
    creating or overwriting any file.
    """
    config_path = Path(path).expanduser() if path else get_default_global_config_path()

    if not config_path.is_file():
        return GlobalConfig()

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw_data: Any = yaml.safe_load(raw_text)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read or parse global configuration YAML at '{config_path}': {e}"
        ) from e

    if raw_data is None:
        return GlobalConfig()

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            f"Global configuration at '{config_path}' must be a mapping/dictionary, "
            f"got {type(raw_data).__name__}."
        )

    try:
        return GlobalConfig.model_validate(raw_data)
    except ValidationError as e:
        error_messages = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            error_messages.append(f"  - Field '{loc}': {err['msg']}")
        joined_errors = "\n".join(error_messages)
        raise ConfigurationError(
            f"Global configuration at '{config_path}' has validation errors:\n{joined_errors}"
        ) from e


def save_global_config(config: GlobalConfig, path: Path | str | None = None) -> None:
    """Write the global configuration file, preserving every field on `config`.

    Callers that want to change one section (e.g. `models.retired`) should first
    `load_global_config()` to get the full current object, mutate that section, then pass
    the whole object here -- this always writes the complete file, never a partial merge.
    """
    config_path = Path(path).expanduser() if path else get_default_global_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", exclude_none=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_project_config(path: Path | str) -> ProjectConfig:
    """Load and validate a project routing configuration file (.ai-orchestrator/routing.yaml)."""
    config_path = Path(path).resolve()

    if not config_path.is_file():
        raise ConfigurationError(f"Project routing configuration not found at '{config_path}'.")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw_data: Any = yaml.safe_load(raw_text)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read or parse project configuration YAML at '{config_path}': {e}"
        ) from e

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            f"Project configuration at '{config_path}' must be a mapping/dictionary, "
            f"got {type(raw_data).__name__}."
        )

    try:
        return ProjectConfig.model_validate(raw_data)
    except ValidationError as e:
        error_messages = []
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            error_messages.append(f"  - Field '{loc}': {err['msg']}")
        joined_errors = "\n".join(error_messages)
        raise ConfigurationError(
            f"Project configuration at '{config_path}' has validation errors:\n{joined_errors}"
        ) from e
