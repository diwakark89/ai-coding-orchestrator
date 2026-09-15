"""Machine-readable TaskProfile schema produced by the planning workflow."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Stage(str, Enum):
    """Workflow stage a TaskProfile primarily describes."""

    PLANNING = "PLANNING"
    IMPLEMENTATION = "IMPLEMENTATION"
    REVIEW = "REVIEW"
    DOCUMENTATION = "DOCUMENTATION"


class TaskProfile(BaseModel):
    """Structured facts describing a unit of work, used later for deterministic routing.

    The planner produces this; it must never select the exact implementation model itself.
    """

    model_config = ConfigDict(extra="forbid")

    stage: Stage

    technologies: set[str]
    affected_layers: set[str]

    estimated_files: int = Field(ge=0)

    schema_change: bool = False
    destructive_schema_change: bool = False

    api_contract_change: bool = False

    authentication: bool = False
    authorization: bool = False
    data_ownership: bool = False

    transaction_logic: bool = False
    concurrency: bool = False
    idempotency: bool = False

    payment: bool = False
    security_boundary_change: bool = False

    external_integration: bool = False
    new_dependency: bool = False

    new_service: bool = False
    new_datastore: bool = False

    architecture_change: bool = False

    ai_or_rag: bool = False
    performance_sensitive: bool = False
