"""Project discovery and identity calculation."""

import hashlib
import os
from pathlib import Path

from agentflow.config.loader import load_project_config
from agentflow.errors import ProjectNotFoundError
from agentflow.project.context import ProjectContext

ROUTING_CONFIG_RELATIVE_PATH = Path(".ai-orchestrator") / "routing.yaml"


def generate_project_id(canonical_path: Path) -> str:
    """Generate a deterministic SHA-256 project identifier from canonical repository path.

    Normalizes case and separator representation to ensure cross-platform stability.
    """
    resolved = canonical_path.resolve()
    normalized = os.path.normcase(str(resolved))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def find_git_root(start_path: Path) -> Path | None:
    """Walk upward from start_path to locate the Git repository root.

    Detects both standard `.git` directories and `.git` files (such as those in worktrees).
    """
    current = start_path.resolve()
    while True:
        git_entry = current / ".git"
        if git_entry.exists():
            return current
        parent = current.parent
        if parent == current:
            # Reached root of the filesystem
            return None
        current = parent


def discover_project(
    start_path: Path | None = None,
    explicit_path: Path | None = None,
) -> ProjectContext:
    """Discover project context from explicit path or upward walk from current working directory.

    Explicit project path takes precedence over auto-discovery.
    """
    if explicit_path is not None:
        target = explicit_path.resolve()
        if not target.exists():
            raise ProjectNotFoundError(f"Specified project path does not exist: {target}")
        if not target.is_dir():
            raise ProjectNotFoundError(f"Specified project path is not a directory: {target}")

        git_root = find_git_root(target)
        if git_root is not None:
            root_path = git_root
            is_git_repo = True
        else:
            root_path = target
            is_git_repo = False
    else:
        search_start = (start_path or Path.cwd()).resolve()
        if not search_start.exists():
            raise ProjectNotFoundError(f"Start path does not exist: {search_start}")
        git_root = find_git_root(search_start)
        if git_root is not None:
            root_path = git_root
            is_git_repo = True
        else:
            root_path = search_start
            is_git_repo = False

    routing_config_file = root_path / ROUTING_CONFIG_RELATIVE_PATH
    routing_config = None
    routing_config_error = None
    project_name = root_path.name

    if routing_config_file.is_file():
        try:
            routing_config = load_project_config(routing_config_file)
            project_name = routing_config.project.name
        except Exception as e:
            routing_config = None
            routing_config_error = str(e)

    project_id = generate_project_id(root_path)

    return ProjectContext(
        root_path=root_path,
        project_id=project_id,
        project_name=project_name,
        routing_config_path=routing_config_file if routing_config_file.exists() else None,
        routing_config=routing_config,
        routing_config_error=routing_config_error,
        is_git_repo=is_git_repo,
    )
