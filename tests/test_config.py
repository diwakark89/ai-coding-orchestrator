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
    assert config.cli.gemini.dialect == "gemini"
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


def test_load_global_config_antigravity_dialect(tmp_path: Path):
    """The Google-provider CLI can be pointed at Antigravity's `agy` via `dialect`."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
version: 1

cli:
  gemini:
    command: agy
    dialect: antigravity
""",
        encoding="utf-8",
    )

    config = load_global_config(config_file)
    assert config.cli.gemini.command == "agy"
    assert config.cli.gemini.dialect == "antigravity"


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


def test_load_project_config_with_full_routing_schema(tmp_path: Path):
    """A routing.yaml with models/complexity/routing sections validates end to end."""
    project_config_file = tmp_path / "routing.yaml"
    project_config_file.write_text(
        """
version: 1

project:
  name: viteprep

models:
  planner:
    default:
      provider: anthropic
      model: Claude Sonnet 5
    architecture:
      provider: anthropic
      model: Claude Opus 5
  implementation:
    lightweight:
      provider: openai
      model: GPT-5.6 Luna
    standard:
      provider: openai
      model: GPT-5.6 Terra
    escalation:
      provider: anthropic
      model: Claude Sonnet 5
  review:
    default:
      provider: google
      model: Gemini 3.8 Flash
    deep:
      provider: anthropic
      model: Claude Sonnet 5
    architecture:
      provider: anthropic
      model: Claude Opus 5
  documentation:
    default:
      provider: google
      model: Gemini 3.8 Flash

complexity:
  file_count:
    "1-3": 0
    "4-7": 1
    "8+": 2
  flags:
    payment: 3
  thresholds:
    low:
      max: 2
    medium:
      min: 3
      max: 5
    high:
      min: 6

routing:
  implementation:
    force_standard_if_any:
      - payment
    rules:
      - id: low-complexity
        when:
          complexity: low
        use: implementation.lightweight
      - id: medium-complexity
        when:
          complexity: medium
        use: implementation.standard

escalation:
  implementation:
    lightweight:
      verification_failure_limit: 2
      next: implementation.standard

verification:
  python:
    detect:
      - pyproject.toml
    commands:
      - pytest
""",
        encoding="utf-8",
    )

    cfg = load_project_config(project_config_file)
    assert cfg.models is not None
    assert cfg.models.implementation.lightweight.model == "GPT-5.6 Luna"
    assert cfg.complexity is not None
    assert cfg.complexity.flags["payment"] == 3
    assert cfg.routing is not None
    assert cfg.routing.implementation.rules[0].id == "low-complexity"
    assert cfg.escalation is not None
    assert cfg.verification is not None


def test_load_project_config_rejects_undefined_model_alias(tmp_path: Path):
    """A routing rule referencing a model alias absent from models: is rejected."""
    project_config_file = tmp_path / "routing.yaml"
    project_config_file.write_text(
        """
version: 1

project:
  name: broken-alias-project

routing:
  implementation:
    rules:
      - id: low-complexity
        when:
          complexity: low
        use: implementation.does_not_exist
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Undefined model alias"):
        load_project_config(project_config_file)


def test_load_project_config_routing_without_models_uses_v1_defaults(tmp_path: Path):
    """A routing: section referencing only the standard V1 aliases needs no models: section."""
    project_config_file = tmp_path / "routing.yaml"
    project_config_file.write_text(
        """
version: 1

project:
  name: defaults-only-project

routing:
  implementation:
    rules:
      - id: low-complexity
        when:
          complexity: low
        use: implementation.lightweight
""",
        encoding="utf-8",
    )

    cfg = load_project_config(project_config_file)
    assert cfg.models is None
    assert cfg.routing is not None


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
