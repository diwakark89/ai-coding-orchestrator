"""Project context data model."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentflow.config.models import ProjectConfig


class ProjectContext(BaseModel):
    """Structured context representing an active or discovered project."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root_path: Path
    project_id: str
    project_name: str
    routing_config_path: Path | None = None
    routing_config: ProjectConfig | None = None
    routing_config_error: str | None = None
    is_git_repo: bool = True
