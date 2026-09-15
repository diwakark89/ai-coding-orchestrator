"""RoutingDecision contract: an explainable result of the deterministic routing engine."""

from pydantic import BaseModel, ConfigDict

from agentflow.agents.base import Provider
from agentflow.routing.complexity import ComplexityLevel
from agentflow.task.profile import Stage


class RoutingDecision(BaseModel):
    """Every model selection must produce an explainable, reproducible result."""

    model_config = ConfigDict(extra="ignore")

    stage: Stage

    provider: Provider
    model: str
    role: str

    matched_rule: str
    reason: str

    complexity_score: int
    complexity: ComplexityLevel

    risk_flags: list[str]
