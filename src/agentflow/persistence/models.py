"""Data models for database records."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class RunStatus(str, Enum):
    """Execution status for an AgentFlow run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class ProjectRecord(BaseModel):
    """Database record representation of a tracked project."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str | None = None
    repository_path: str
    config_hash: str | None = None
    created_at: datetime
    last_used_at: datetime


class RunRecord(BaseModel):
    """Database record representation of an AgentFlow execution run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    task: str
    status: str
    created_at: datetime
    updated_at: datetime
