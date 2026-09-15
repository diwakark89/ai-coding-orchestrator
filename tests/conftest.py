"""Shared pytest fixtures for AgentFlow tests."""

import subprocess
from pathlib import Path

import pytest


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a minimal real Git repo with one commit, for worktree/implementation tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init"], cwd=repo)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    _git(["config", "user.name", "AgentFlow Test"], cwd=repo)
    (repo / "README.md").write_text("# Test Repo\n", encoding="utf-8")
    _git(["add", "-A"], cwd=repo)
    _git(["commit", "-m", "Initial commit"], cwd=repo)
    return repo
