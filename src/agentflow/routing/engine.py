"""Deterministic routing engine: TaskProfile + routing configuration -> RoutingDecision.

Fixed precedence, never reordered (TDD §18):
    1. Explicit user override
    2. Hard safety/risk rule
    3. Workflow-stage rule
    4. Project-specific rule
    5. Complexity rule
    6. Default

No AI model ever decides which model executes the next stage — this module is a pure
function of its inputs and must produce an identical RoutingDecision for identical inputs.
"""

from collections.abc import Mapping

from agentflow.agents.base import validate_model_allowed
from agentflow.config.retirement import apply_retirements
from agentflow.routing.complexity import (
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityConfig,
    ComplexityLevel,
    ComplexityResult,
    score_complexity,
)
from agentflow.routing.decision import RoutingDecision
from agentflow.routing.matcher import match_complexity_rule
from agentflow.routing.rules import (
    DEFAULT_MODELS_CONFIG,
    DEFAULT_ROUTING_RULES,
    ImplementationRouting,
    ModelRef,
    ModelsConfig,
    PlanningRouting,
    ReviewRouting,
    RoutingRulesConfig,
)
from agentflow.task.profile import Stage, TaskProfile


def route(
    task_profile: TaskProfile,
    models: ModelsConfig | None = None,
    routing_rules: RoutingRulesConfig | None = None,
    complexity_config: ComplexityConfig | None = None,
    user_override: ModelRef | None = None,
    retired_models: Mapping[str, str] | None = None,
) -> RoutingDecision:
    """Deterministically select a provider/model for a TaskProfile's stage.

    Given the same TaskProfile and routing configuration, always returns an identical
    RoutingDecision (TDD §2, §24). The planner never influences this beyond the facts it
    reported in the TaskProfile itself.

    `retired_models` is the global, user-managed retirement map (`GlobalConfig.models.retired`,
    see config/retirement.py); when set, every model this function would otherwise return --
    including an explicit `user_override` -- is transparently substituted through it.
    """
    models_cfg = models or DEFAULT_MODELS_CONFIG
    rules_cfg = routing_rules or DEFAULT_ROUTING_RULES
    complexity_result = score_complexity(
        task_profile, complexity_config or DEFAULT_COMPLEXITY_CONFIG
    )

    if user_override is not None:
        ref = user_override
        if retired_models:
            substituted = apply_retirements(ref.model, retired_models)
            if substituted != ref.model:
                validate_model_allowed(substituted)
                ref = ModelRef(provider=ref.provider, model=substituted)
        return _decision(
            task_profile,
            ref,
            role="user.override",
            matched_rule="user-override",
            reason="Explicit user override takes precedence over all routing rules.",
            complexity_result=complexity_result,
            risk_flags=[],
        )

    if task_profile.stage == Stage.PLANNING:
        return _route_planning(
            task_profile, models_cfg, rules_cfg.planning, complexity_result, retired_models
        )
    if task_profile.stage == Stage.REVIEW:
        return _route_review(
            task_profile, models_cfg, rules_cfg.review, complexity_result, retired_models
        )
    if task_profile.stage == Stage.DOCUMENTATION:
        ref = models_cfg.resolve(rules_cfg.documentation.default, retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role=rules_cfg.documentation.default,
            matched_rule="documentation.default",
            reason="Documentation always uses the configured default model.",
            complexity_result=complexity_result,
            risk_flags=[],
        )

    # Stage.IMPLEMENTATION (and any future stage defaults to implementation routing).
    return _route_implementation(
        task_profile, models_cfg, rules_cfg.implementation, complexity_result, retired_models
    )


def _route_planning(
    task_profile: TaskProfile,
    models_cfg: ModelsConfig,
    planning_rules: PlanningRouting,
    complexity_result: ComplexityResult,
    retired_models: Mapping[str, str] | None = None,
) -> RoutingDecision:
    """Apply planning-stage precedence: workflow-stage architecture escalation, then default."""
    triggered = [
        flag
        for flag in planning_rules.architecture_if_any
        if getattr(task_profile, flag, False) is True
    ]
    if triggered:
        ref = models_cfg.resolve(planning_rules.architecture, retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role=planning_rules.architecture,
            matched_rule="planning.architecture-escalation",
            reason=f"Flag(s) {', '.join(triggered)} require architecture-level planning.",
            complexity_result=complexity_result,
            risk_flags=triggered,
        )

    ref = models_cfg.resolve(planning_rules.default, retired=retired_models)
    return _decision(
        task_profile,
        ref,
        role=planning_rules.default,
        matched_rule="planning.default",
        reason="No architecture-escalation flags present; using the default planner.",
        complexity_result=complexity_result,
        risk_flags=[],
    )


def _route_implementation(
    task_profile: TaskProfile,
    models_cfg: ModelsConfig,
    impl_rules: ImplementationRouting,
    complexity_result: ComplexityResult,
    retired_models: Mapping[str, str] | None = None,
) -> RoutingDecision:
    """Apply implementation precedence: hard risk, then project rule, then built-in default."""
    triggered_risks = [
        flag
        for flag in impl_rules.force_standard_if_any
        if getattr(task_profile, flag, False) is True
    ]
    if triggered_risks:
        ref = models_cfg.resolve("implementation.standard", retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role="implementation.standard",
            matched_rule="implementation.force-standard",
            reason=(
                f"Hard-risk flag(s) {', '.join(triggered_risks)} require at least the "
                "standard implementation tier."
            ),
            complexity_result=complexity_result,
            risk_flags=triggered_risks,
        )

    matched = match_complexity_rule(impl_rules.rules, complexity_result.level)
    if matched is not None:
        ref = models_cfg.resolve(matched.use, retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role=matched.use,
            matched_rule=matched.id,
            reason=(
                f"Complexity {complexity_result.level.value} matched project rule '{matched.id}'."
            ),
            complexity_result=complexity_result,
            risk_flags=complexity_result.contributing_factors,
        )

    fallback_alias = {
        ComplexityLevel.LOW: "implementation.lightweight",
        ComplexityLevel.MEDIUM: "implementation.standard",
        ComplexityLevel.HIGH: "implementation.standard",
    }[complexity_result.level]
    ref = models_cfg.resolve(fallback_alias, retired=retired_models)
    return _decision(
        task_profile,
        ref,
        role=fallback_alias,
        matched_rule="implementation.default-complexity",
        reason=(
            f"No project rule matched; applied the built-in default for "
            f"{complexity_result.level.value} complexity."
        ),
        complexity_result=complexity_result,
        risk_flags=complexity_result.contributing_factors,
    )


def _route_review(
    task_profile: TaskProfile,
    models_cfg: ModelsConfig,
    review_rules: ReviewRouting,
    complexity_result: ComplexityResult,
    retired_models: Mapping[str, str] | None = None,
) -> RoutingDecision:
    """Apply review-stage precedence: architecture escalation, then deep review, then default."""
    architecture_triggered = [
        flag
        for flag, expected in review_rules.architecture_if.items()
        if getattr(task_profile, flag, False) is expected
    ]
    if architecture_triggered:
        ref = models_cfg.resolve(review_rules.architecture, retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role=review_rules.architecture,
            matched_rule="review.architecture-escalation",
            reason=f"Flag(s) {', '.join(architecture_triggered)} require architecture review.",
            complexity_result=complexity_result,
            risk_flags=architecture_triggered,
        )

    deep_triggered = [
        flag for flag in review_rules.deep_if_any if getattr(task_profile, flag, False) is True
    ]
    if deep_triggered:
        ref = models_cfg.resolve(review_rules.deep, retired=retired_models)
        return _decision(
            task_profile,
            ref,
            role=review_rules.deep,
            matched_rule="review.deep-escalation",
            reason=f"Flag(s) {', '.join(deep_triggered)} require deeper review.",
            complexity_result=complexity_result,
            risk_flags=deep_triggered,
        )

    ref = models_cfg.resolve(review_rules.default, retired=retired_models)
    return _decision(
        task_profile,
        ref,
        role=review_rules.default,
        matched_rule="review.default",
        reason="No deep- or architecture-review flags present; using the default reviewer.",
        complexity_result=complexity_result,
        risk_flags=[],
    )


def _decision(
    task_profile: TaskProfile,
    ref: ModelRef,
    role: str,
    matched_rule: str,
    reason: str,
    complexity_result: ComplexityResult,
    risk_flags: list[str],
) -> RoutingDecision:
    """Build a RoutingDecision from a resolved model binding and the computed complexity."""
    return RoutingDecision(
        stage=task_profile.stage,
        provider=ref.provider_enum,
        model=ref.model,
        role=role,
        matched_rule=matched_rule,
        reason=reason,
        complexity_score=complexity_result.score,
        complexity=complexity_result.level,
        risk_flags=risk_flags,
    )
