"""Unit tests for `agentflow init`: starter profile generation."""

from pathlib import Path

import pytest
import yaml

from agentflow.application import Application
from agentflow.config.models import GlobalConfig, ProjectConfig
from agentflow.errors import InitError
from agentflow.project.init import build_starter_config, detect_verification_groups


def test_detect_verification_groups_root_marker(tmp_path: Path):
    """A marker file at the repo root is detected as a group keyed by its kind."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    groups = detect_verification_groups(tmp_path)

    assert "python" in groups
    assert groups["python"].detect == ["pyproject.toml"]
    assert groups["python"].commands == ["pytest"]


def test_detect_verification_groups_nested_directories(tmp_path: Path):
    """Marker files in subdirectories are detected as groups keyed by directory name."""
    (tmp_path / "front-end").mkdir()
    (tmp_path / "front-end" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "web-service").mkdir()
    (tmp_path / "web-service" / "pom.xml").write_text("<project/>", encoding="utf-8")

    groups = detect_verification_groups(tmp_path)

    assert groups["front-end"].detect == ["front-end/package.json"]
    assert groups["front-end"].commands == ["npm test"]
    assert groups["web-service"].detect == ["web-service/pom.xml"]
    assert groups["web-service"].commands == ["mvn test"]


def test_detect_verification_groups_skips_dependency_directories(tmp_path: Path):
    """node_modules/.venv/etc. are never walked into, even if they contain marker files."""
    (tmp_path / "node_modules" / "some-pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "some-pkg" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "pyproject.toml").write_text("", encoding="utf-8")

    groups = detect_verification_groups(tmp_path)

    assert groups == {}


def test_detect_verification_groups_dedupes_same_directory_name(tmp_path: Path):
    """Two directories with the same name get distinct group keys."""
    (tmp_path / "a" / "python").mkdir(parents=True)
    (tmp_path / "a" / "python" / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "b" / "python").mkdir(parents=True)
    (tmp_path / "b" / "python" / "pyproject.toml").write_text("", encoding="utf-8")

    groups = detect_verification_groups(tmp_path)

    assert "python" in groups
    assert "python-2" in groups
    assert groups["python"].detect != groups["python-2"].detect


def test_detect_verification_groups_no_markers_found(tmp_path: Path):
    """An empty repository detects no groups at all."""
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    assert detect_verification_groups(tmp_path) == {}


def test_build_starter_config_is_valid_project_config(tmp_path: Path):
    """build_starter_config produces a ProjectConfig that round-trips through validation."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    config = build_starter_config(tmp_path)

    assert isinstance(config, ProjectConfig)
    assert config.project.name == tmp_path.name
    assert config.models is not None
    assert config.routing is not None
    assert config.documentation is not None
    assert config.documentation.enabled is False
    assert config.verification is not None and "python" in config.verification

    # Round-trip through the exact serialization run_init uses.
    dumped = yaml.safe_dump(config.model_dump(mode="json", exclude_none=True), sort_keys=False)
    reloaded = ProjectConfig.model_validate(yaml.safe_load(dumped))
    assert reloaded.project.name == config.project.name


def _app(tmp_path: Path) -> Application:
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "storage": {"database": str(tmp_path / "app.db")},
            "worktrees": {"root": str(tmp_path / "worktrees")},
            "logging": {"root": str(tmp_path / "logs")},
        }
    )
    return Application(config=cfg)


def test_run_init_writes_starter_profile(tmp_path: Path):
    """agentflow init writes .ai-orchestrator/routing.yaml under the target project."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    result = _app(tmp_path).run_init(project_path=tmp_path)

    expected_path = tmp_path / ".ai-orchestrator" / "routing.yaml"
    assert result.path == expected_path
    assert expected_path.exists()
    assert result.overwritten is False
    assert "python" in result.detected_groups

    loaded = ProjectConfig.model_validate(yaml.safe_load(expected_path.read_text()))
    assert loaded.project.name == tmp_path.name


def test_run_init_refuses_to_overwrite_without_force(tmp_path: Path):
    """agentflow init refuses to clobber an existing routing.yaml unless --force is passed."""
    (tmp_path / ".git").mkdir()
    orch_dir = tmp_path / ".ai-orchestrator"
    orch_dir.mkdir()
    (orch_dir / "routing.yaml").write_text(
        "version: 1\nproject:\n  name: existing\n", encoding="utf-8"
    )

    with pytest.raises(InitError, match="already exists"):
        _app(tmp_path).run_init(project_path=tmp_path)

    # The pre-existing file must be untouched.
    assert "existing" in (orch_dir / "routing.yaml").read_text()


def test_run_init_overwrites_with_force(tmp_path: Path):
    """agentflow init --force replaces an existing routing.yaml."""
    (tmp_path / ".git").mkdir()
    orch_dir = tmp_path / ".ai-orchestrator"
    orch_dir.mkdir()
    (orch_dir / "routing.yaml").write_text(
        "version: 1\nproject:\n  name: existing\n", encoding="utf-8"
    )

    result = _app(tmp_path).run_init(project_path=tmp_path, force=True)

    assert result.overwritten is True
    loaded = ProjectConfig.model_validate(yaml.safe_load((orch_dir / "routing.yaml").read_text()))
    assert loaded.project.name == tmp_path.name
