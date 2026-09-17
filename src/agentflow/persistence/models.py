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
    state: str = "NEW"
    created_at: datetime
    updated_at: datetime


class RunStateTransitionRecord(BaseModel):
    """Database record of a single workflow state transition for a run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    from_state: str | None
    to_state: str
    reason: str | None
    created_at: datetime


class AgentSessionRecord(BaseModel):
    """Database record of an individual agent CLI session invoked during a run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    stage: str
    provider: str
    model: str
    cli_session_id: str | None
    created_at: datetime


class DecisionRecord(BaseModel):
    """Database record of a user decision (question/answer) made during a run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    question: str
    answer: str
    created_at: datetime


class RoutingDecisionRecord(BaseModel):
    """Database record of a persisted deterministic routing decision."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    stage: str
    task_profile_json: str
    complexity_score: int
    complexity_level: str
    matched_rule: str
    provider: str
    model: str
    reason: str
    created_at: datetime


class StageRecord(BaseModel):
    """Database record tracking attempts of a single workflow stage for a run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    stage: str
    status: str
    attempt_count: int = 0
    started_at: datetime
    completed_at: datetime | None = None


class VerificationRunRecord(BaseModel):
    """Database record of a single verification command execution."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    command: str
    exit_code: int
    stdout_path: str | None
    stderr_path: str | None
    started_at: datetime
    completed_at: datetime


class CliAvailabilityRecord(BaseModel):
    """Database record of the most recent `agentflow doctor` check for one provider CLI."""

    model_config = ConfigDict(from_attributes=True)

    provider: str
    command: str
    available: bool
    last_checked_at: datetime


class EventRecord(BaseModel):
    """A safe, structured local observability event for a workflow run."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    stage: str | None = None
    event: str
    provider: str | None = None
    model: str | None = None
    attributes: dict[str, str | int | float | bool]
    created_at: datetime
