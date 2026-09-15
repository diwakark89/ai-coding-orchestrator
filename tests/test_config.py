"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest

from agentflow.config.loader import load_global_config, load_project_config
from agentflow.config.models import GlobalConfig
from agentflow.errors import ConfigurationError


def test_default_global_config():
    """Verify default GlobalConfig values when no file exists."""
    config = GlobalConfig()
    assert config.version == 1
    assert config.cli.claude.command == "claude"
    assert config.cli.codex.command == "codex"
    assert config.cli.gemini.command == "gemini"
    assert "agentflow.db" in str(config.storage.database)
    assert "worktrees" in str(config.worktrees.root)
    assert "logs" in str(config.logging.root)


def test_tilde_expansion():
    """Verify that tildes (~) are expanded in storage paths."""
    config = GlobalConfig.model_validate(
        {
            "version": 1,
            "storage": {"database": "~/custom/db.sqlite"},
            "worktrees": {"root": "~/custom/worktrees"},
            "logging": {"root": "~/custom/logs"},
        }
    )
    assert not str(config.storage.database).startswith("~")
    assert not str(config.worktrees.root).startswith("~")
    assert not str(config.logging.root).startswith("~")
    assert str(config.storage.database).endswith("custom/db.sqlite".replace("/", "\\")) or str(
        config.storage.database
    ).endswith("custom/db.sqlite")


def test_load_global_config_missing_file(tmp_path: Path):
    """Loading non-existent global config returns defaults without creating file."""
    missing_file = tmp_path / "nonexistent" / "config.yaml"
    config = load_global_config(missing_file)
    assert config.version == 1
    assert not missing_file.exists()


def test_load_global_config_valid_file(tmp_path: Path):
    """Loading valid YAML configuration file populates model correctly."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
version: 1

cli:
  claude:
    command: /usr/local/bin/claude
  codex:
    command: codex-custom
  gemini:
    command: gemini-cli

storage:
  database: /data/agentflow.db

worktrees:
  root: /data/worktrees

logging:
  root: /data/logs
""",
        encoding="utf-8",
    )

    config = load_global_config(config_file)
    assert config.version == 1
    assert config.cli.claude.command == "/usr/local/bin/claude"
    assert config.cli.codex.command == "codex-custom"
    assert config.cli.gemini.command == "gemini-cli"
    assert str(config.storage.database).replace("\\", "/").endswith("/data/agentflow.db")


def test_load_global_config_invalid_version(tmp_path: Path):
    """Rejection of unsupported configuration versions."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("version: 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unsupported configuration version"):
        load_global_config(config_file)


def test_load_global_config_invalid_yaml(tmp_path: Path):
    """Rejection of malformed YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("version: [unclosed list", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Failed to read or parse global configuration"):
        load_global_config(config_file)


def test_load_global_config_not_mapping(tmp_path: Path):
    """Rejection when YAML root is not a dictionary."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be a mapping/dictionary"):
        load_global_config(config_file)


def test_load_project_config_valid(tmp_path: Path):
    """Validate loading valid project routing configuration."""
    project_config_file = tmp_path / "routing.yaml"
    project_config_file.write_text(
        """
version: 1

project:
  name: test-project
""",
        encoding="utf-8",
    )

    cfg = load_project_config(project_config_file)
    assert cfg.version == 1
    assert cfg.project.name == "test-project"


def test_load_project_config_invalid_version(tmp_path: Path):
    """Reject project config with version != 1."""
    project_config_file = tmp_path / "routing.yaml"
    project_config_file.write_text(
        """
version: 99

project:
  name: test-project
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Unsupported configuration version"):
        load_project_config(project_config_file)


def test_load_project_config_missing_file(tmp_path: Path):
    """Raise ConfigurationError when project routing config is missing."""
    missing = tmp_path / "routing.yaml"
    with pytest.raises(ConfigurationError, match="not found"):
        load_project_config(missing)


def test_load_global_config_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """AGENTFLOW_CONFIG_PATH environment variable overrides default path."""
    env_config = tmp_path / "env_config.yaml"
    env_config.write_text(
        """
version: 1
cli:
  claude:
    command: claude-env
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTFLOW_CONFIG_PATH", str(env_config))

    cfg = load_global_config()
    assert cfg.cli.claude.command == "claude-env"


def test_cli_commands_defaults_when_empty_dict_passed():
    """Verify that empty mapping for codex or gemini does not inadvertently default to claude."""
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "cli": {
                "codex": {},
                "gemini": {},
            },
        }
    )
    assert cfg.cli.claude.command == "claude"
    assert cfg.cli.codex.command == "codex"
    assert cfg.cli.gemini.command == "gemini"
