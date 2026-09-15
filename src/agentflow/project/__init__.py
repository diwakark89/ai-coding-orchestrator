"""Project discovery and context module."""

from agentflow.project.context import ProjectContext
from agentflow.project.discovery import (
    ROUTING_CONFIG_RELATIVE_PATH,
    discover_project,
    find_git_root,
    generate_project_id,
)

__all__ = [
    "ProjectContext",
    "ROUTING_CONFIG_RELATIVE_PATH",
    "discover_project",
    "find_git_root",
    "generate_project_id",
]
