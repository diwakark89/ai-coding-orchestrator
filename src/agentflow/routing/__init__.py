"""Deterministic model-routing module for AgentFlow."""

from agentflow.routing.complexity import (
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityConfig,
    ComplexityLevel,
    ComplexityResult,
    ComplexityThreshold,
    score_complexity,
)
from agentflow.routing.decision import RoutingDecision
from agentflow.routing.engine import route
from agentflow.routing.matcher import match_complexity_rule
from agentflow.routing.rules import (
    DEFAULT_MODELS_CONFIG,
    DEFAULT_ROUTING_RULES,
    ComplexityRule,
    ComplexityRuleWhen,
    DocumentationModels,
    DocumentationRouting,
    ImplementationModels,
    ImplementationRouting,
    ModelRef,
    ModelsConfig,
    PlannerModels,
    PlanningRouting,
    ReviewModels,
    ReviewRouting,
    RoutingRulesConfig,
    all_referenced_aliases,
)

__all__ = [
    "DEFAULT_COMPLEXITY_CONFIG",
    "DEFAULT_MODELS_CONFIG",
    "DEFAULT_ROUTING_RULES",
    "ComplexityConfig",
    "ComplexityLevel",
    "ComplexityResult",
    "ComplexityRule",
    "ComplexityRuleWhen",
    "ComplexityThreshold",
    "DocumentationModels",
    "DocumentationRouting",
    "ImplementationModels",
    "ImplementationRouting",
    "ModelRef",
    "ModelsConfig",
    "PlannerModels",
    "PlanningRouting",
    "ReviewModels",
    "ReviewRouting",
    "RoutingDecision",
    "RoutingRulesConfig",
    "all_referenced_aliases",
    "match_complexity_rule",
    "route",
    "score_complexity",
]
