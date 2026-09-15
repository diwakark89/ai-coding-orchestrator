"""Integration tests for Typer CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from agentflow import __version__
from agentflow.cli import app

runner = CliRunner()


def test_cli_version():
    """agentflow --version prints current version and exits with 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"agentflow {__version__}" in result.output


def test_cli_version_short():
    """agentflow -v prints current version and exits with 0."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert f"agentflow {__version__}" in result.output


def test_cli_doctor():
    """agentflow doctor executes and displays check results."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "AgentFlow Doctor" in result.output
    assert "Python runtime" in result.output


def test_cli_status():
    """agentflow status executes and displays status message."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Status:" in result.output or "Active runs" in result.output


def test_cli_with_explicit_project_flag(tmp_path: Path):
    """agentflow --project <path> status inspects specified directory."""
    (tmp_path / ".git").mkdir()
    result = runner.invoke(app, ["--project", str(tmp_path), "status"])
    assert result.exit_code == 0
    assert "No active runs found" in result.output


def test_cli_with_dash_c_flag(tmp_path: Path):
    """agentflow -C <path> doctor inspects specified directory."""
    (tmp_path / ".git").mkdir()
    result = runner.invoke(app, ["-C", str(tmp_path), "doctor"])
    assert result.exit_code == 0
    assert "AgentFlow Doctor" in result.output


def test_cli_state_isolation_between_invocations(tmp_path: Path):
    """Global option from previous invocation does not leak into subsequent invocations."""
    project_dir = tmp_path / "custom_proj"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    # Invocations 1: With -C
    res1 = runner.invoke(app, ["-C", str(project_dir), "status"])
    assert res1.exit_code == 0
    assert "custom_proj" in res1.output

    # Invocation 2: Without -C (should auto-discover from cwd, not use custom_proj)
    res2 = runner.invoke(app, ["status"])
    assert res2.exit_code == 0
    assert "custom_proj" not in res2.output


def test_cli_status_nonexistent_project_fails(tmp_path: Path):
    """status command exits with code 1 when target project does not exist."""
    missing = tmp_path / "missing_project_dir"
    result = runner.invoke(app, ["-C", str(missing), "status"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_cli_doctor_nonexistent_project_fails(tmp_path: Path):
    """doctor command exits with code 1 when target project does not exist."""
    missing = tmp_path / "missing_project_dir"
    result = runner.invoke(app, ["-C", str(missing), "doctor"])
    assert result.exit_code == 1
    assert "Critical environment requirements are missing" in result.output
