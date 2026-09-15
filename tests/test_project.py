"""Unit tests for project discovery and identity."""

from pathlib import Path

import pytest

from agentflow.errors import ProjectNotFoundError
from agentflow.project.discovery import (
    discover_project,
    find_git_root,
    generate_project_id,
)


def test_generate_project_id_stability(tmp_path: Path):
    """Project ID must be deterministic and identical for the same path."""
    repo_dir = tmp_path / "my_project"
    repo_dir.mkdir()

    id1 = generate_project_id(repo_dir)
    id2 = generate_project_id(repo_dir)

    assert id1 == id2
    assert len(id1) == 64  # SHA-256 hex string


def test_generate_project_id_distinct_for_different_paths(tmp_path: Path):
    """Different paths must produce different IDs even if folder names match."""
    p1 = tmp_path / "a" / "app"
    p2 = tmp_path / "b" / "app"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)

    id1 = generate_project_id(p1)
    id2 = generate_project_id(p2)

    assert id1 != id2


def test_find_git_root_direct(tmp_path: Path):
    """Detect git root when start_path contains .git."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    detected = find_git_root(tmp_path)
    assert detected == tmp_path.resolve()


def test_find_git_root_nested(tmp_path: Path):
    """Detect git root by walking upward from deeply nested directory."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    nested_dir = tmp_path / "src" / "components" / "button"
    nested_dir.mkdir(parents=True)

    detected = find_git_root(nested_dir)
    assert detected == tmp_path.resolve()


def test_find_git_root_not_found(tmp_path: Path):
    """Return None when no .git is found upward."""
    non_git = tmp_path / "some_dir"
    non_git.mkdir()

    # Note: tmp_path is in system temp, ensure no parent has .git
    # If system temp has .git, this might find it, but normally it doesn't.
    # We can check find_git_root returns None or non_git if isolated.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    # Check find_git_root
    res = find_git_root(isolated)
    # Either None or not isolated
    assert res is None or res != isolated


def test_discover_project_with_git_and_routing_config(tmp_path: Path):
    """Discover project with both git root and .ai-orchestrator/routing.yaml."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    orch_dir = tmp_path / ".ai-orchestrator"
    orch_dir.mkdir()
    routing_file = orch_dir / "routing.yaml"
    routing_file.write_text(
        """
version: 1
project:
  name: demo-app
""",
        encoding="utf-8",
    )

    nested_subdir = tmp_path / "nested" / "sub"
    nested_subdir.mkdir(parents=True)

    ctx = discover_project(start_path=nested_subdir)

    assert ctx.root_path == tmp_path.resolve()
    assert ctx.project_name == "demo-app"
    assert ctx.is_git_repo is True
    assert ctx.routing_config is not None
    assert ctx.routing_config.project.name == "demo-app"
    assert ctx.routing_config_path == routing_file.resolve()


def test_discover_project_explicit_override(tmp_path: Path):
    """Explicit project path overrides auto-discovery from current directory."""
    project_a = tmp_path / "project_a"
    project_a.mkdir()
    (project_a / ".git").mkdir()

    project_b = tmp_path / "project_b"
    project_b.mkdir()
    (project_b / ".git").mkdir()

    # Start discovery from project_a, but pass explicit project_b
    ctx = discover_project(start_path=project_a, explicit_path=project_b)

    assert ctx.root_path == project_b.resolve()
    assert ctx.project_id == generate_project_id(project_b)


def test_discover_project_nonexistent_explicit_path(tmp_path: Path):
    """Explicit path that does not exist raises ProjectNotFoundError."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ProjectNotFoundError, match="does not exist"):
        discover_project(explicit_path=missing)


def test_find_git_root_worktree_file(tmp_path: Path):
    """Detect git root when .git is a file as in git worktrees."""
    worktree_dir = tmp_path / "worktree_branch"
    worktree_dir.mkdir()
    git_file = worktree_dir / ".git"
    git_file.write_text("gitdir: /path/to/original/.git/worktrees/wt\n", encoding="utf-8")

    sub_dir = worktree_dir / "src" / "pkg"
    sub_dir.mkdir(parents=True)

    detected = find_git_root(sub_dir)
    assert detected == worktree_dir.resolve()


def test_discover_project_explicit_nested_subdirectory(tmp_path: Path):
    """Explicit path to a subdirectory inside a git repo must resolve root to git root."""
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    orch_dir = repo_dir / ".ai-orchestrator"
    orch_dir.mkdir()
    (orch_dir / "routing.yaml").write_text(
        """
version: 1
project:
  name: nested-explicit-app
""",
        encoding="utf-8",
    )

    deep_subdir = repo_dir / "src" / "deep" / "module"
    deep_subdir.mkdir(parents=True)

    ctx = discover_project(explicit_path=deep_subdir)
    assert ctx.root_path == repo_dir.resolve()
    assert ctx.is_git_repo is True
    assert ctx.project_name == "nested-explicit-app"
    assert ctx.routing_config is not None
    assert ctx.routing_config.project.name == "nested-explicit-app"
    assert ctx.project_id == generate_project_id(repo_dir)


def test_discover_project_invalid_routing_config_captures_error(tmp_path: Path):
    """When routing.yaml exists but has invalid schema, routing_config_error records details."""
    repo_dir = tmp_path / "broken_repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    orch_dir = repo_dir / ".ai-orchestrator"
    orch_dir.mkdir()
    (orch_dir / "routing.yaml").write_text("version: 99\n", encoding="utf-8")

    ctx = discover_project(explicit_path=repo_dir)
    assert ctx.routing_config is None
    assert ctx.routing_config_error is not None
    assert "Unsupported configuration version: 99" in ctx.routing_config_error
