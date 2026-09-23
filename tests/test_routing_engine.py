"""Table-driven tests for the deterministic routing engine (Phase 4 mandatory scenarios)."""

from agentflow.agents.base import Provider
from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.engine import route
from agentflow.routing.rules import ModelRef
from agentflow.task.profile import Stage, TaskProfile


def _profile(
    stage: Stage = Stage.IMPLEMENTATION, estimated_files: int = 0, **flags: bool
) -> TaskProfile:
    return TaskProfile(
        stage=stage,
        technologies=set(),
        affected_layers=set(),
        estimated_files=estimated_files,
        **flags,
    )


def test_case_a_low_complexity_no_risk_routes_to_luna():
    """Case A: 2 files, no risk flags -> LOW complexity -> GPT-6 Luna."""
    decision = route(_profile(estimated_files=2))
    assert decision.model == "GPT-6 Luna"
    assert decision.provider == Provider.OPENAI
    assert decision.complexity == ComplexityLevel.LOW
    assert decision.matched_rule == "low-complexity"


def test_case_b_medium_complexity_routes_to_sol():
    """Case B: file count + ordinary flags reach MEDIUM complexity -> GPT-6 Sol."""
    decision = route(_profile(estimated_files=5, schema_change=True, api_contract_change=True))
    assert decision.complexity == ComplexityLevel.MEDIUM
    assert decision.model == "GPT-6 Sol"
    assert decision.matched_rule == "medium-complexity"


def test_case_c_low_complexity_but_authorization_forces_sol():
    """Case C: 2 files + authorization=true -> hard risk forces Sol despite LOW complexity."""
    decision = route(_profile(estimated_files=2, authorization=True))
    assert decision.complexity_score == 2
    assert decision.complexity == ComplexityLevel.LOW
    assert decision.model == "GPT-6 Sol"
    assert decision.matched_rule == "implementation.force-standard"
    assert decision.risk_flags == ["authorization"]


def test_case_d_concurrency_and_idempotency_force_sol():
    """Case D: concurrency + idempotency hard-risk flags force Sol."""
    decision = route(_profile(estimated_files=1, concurrency=True, idempotency=True))
    assert decision.model == "GPT-6 Sol"
    assert decision.matched_rule == "implementation.force-standard"
    assert set(decision.risk_flags) == {"concurrency", "idempotency"}


def test_case_e_architecture_change_during_planning_routes_to_opus():
    """Case E: architecture_change=true during planning -> Claude Opus 5.5."""
    decision = route(_profile(stage=Stage.PLANNING, estimated_files=1, architecture_change=True))
    assert decision.model == "Claude Opus 5.5"
    assert decision.provider == Provider.ANTHROPIC
    assert decision.matched_rule == "planning.architecture-escalation"


def test_case_f_user_override_wins_over_hard_risk():
    """Case F: an explicit user override wins even over a hard-risk match."""
    profile = _profile(estimated_files=2, authorization=True)
    override = ModelRef(provider="anthropic", model="Claude Sonnet 5")
    decision = route(profile, user_override=override)
    assert decision.model == "Claude Sonnet 5"
    assert decision.provider == Provider.ANTHROPIC
    assert decision.matched_rule == "user-override"


def test_case_g_identical_inputs_produce_identical_decisions():
    """Case G: the same TaskProfile + config repeated 100 times yields identical decisions."""
    profile = _profile(estimated_files=5, concurrency=True)
    first = route(profile)
    for _ in range(100):
        assert route(profile) == first


def test_planning_default_when_no_escalation_flags():
    """Planning without any architecture-escalation flag uses the default planner."""
    decision = route(_profile(stage=Stage.PLANNING, estimated_files=1))
    assert decision.model == "Claude Sonnet 5"
    assert decision.matched_rule == "planning.default"


def test_review_default_when_no_flags():
    """Review without deep- or architecture-triggering flags uses the default reviewer."""
    decision = route(_profile(stage=Stage.REVIEW, estimated_files=1))
    assert decision.model == "Gemini 3.8 Flash"
    assert decision.matched_rule == "review.default"


def test_review_deep_escalation_on_authentication():
    """Review flags authentication=true -> deep reviewer (Claude Sonnet 5)."""
    decision = route(_profile(stage=Stage.REVIEW, estimated_files=1, authentication=True))
    assert decision.model == "Claude Sonnet 5"
    assert decision.matched_rule == "review.deep-escalation"


def test_review_architecture_takes_precedence_over_deep():
    """When both deep and architecture conditions are true, architecture review wins."""
    decision = route(
        _profile(
            stage=Stage.REVIEW, estimated_files=1, authentication=True, architecture_change=True
        )
    )
    assert decision.model == "Claude Opus 5.5"
    assert decision.matched_rule == "review.architecture-escalation"


def test_documentation_always_uses_default():
    """Documentation stage always routes to the configured default model."""
    decision = route(_profile(stage=Stage.DOCUMENTATION, estimated_files=1))
    assert decision.model == "Gemini 3.8 Flash"
    assert decision.matched_rule == "documentation.default"


def test_high_complexity_implementation_without_hard_risk_still_routes_to_sol():
    """HIGH complexity reached via non-hard-risk flags routes to Sol through the project rule."""
    decision = route(_profile(estimated_files=8, architecture_change=True, schema_change=True))
    assert decision.complexity_score == 6
    assert decision.complexity == ComplexityLevel.HIGH
    assert decision.model == "GPT-6 Sol"
    assert decision.matched_rule == "high-complexity"


def test_retired_models_substitute_in_implementation_routing():
    """A globally retired model is substituted transparently in implementation routing."""
    decision = route(_profile(estimated_files=2), retired_models={"gpt-6 luna": "GPT-6 Sol"})
    assert decision.model == "GPT-6 Sol"


def test_retired_models_substitute_in_planning_routing():
    """A globally retired model is substituted transparently in planning routing."""
    decision = route(
        _profile(stage=Stage.PLANNING, estimated_files=1, architecture_change=True),
        retired_models={"claude opus 5.5": "Claude Sonnet 5"},
    )
    assert decision.model == "Claude Sonnet 5"


def test_retired_models_substitute_in_review_routing():
    """A globally retired model is substituted transparently in review routing."""
    decision = route(
        _profile(stage=Stage.REVIEW, estimated_files=1),
        retired_models={"gemini 3.8 flash": "Claude Sonnet 5"},
    )
    assert decision.model == "Claude Sonnet 5"


def test_retired_models_substitute_in_documentation_routing():
    """A globally retired model is substituted transparently in documentation routing."""
    decision = route(
        _profile(stage=Stage.DOCUMENTATION, estimated_files=1),
        retired_models={"gemini 3.8 flash": "Claude Sonnet 5"},
    )
    assert decision.model == "Claude Sonnet 5"


def test_retired_models_substitute_in_user_override():
    """An explicit user override is itself subject to retirement substitution."""
    override = ModelRef(provider="openai", model="GPT-6 Luna")
    decision = route(
        _profile(estimated_files=2),
        user_override=override,
        retired_models={"gpt-6 luna": "GPT-6 Sol"},
    )
    assert decision.model == "GPT-6 Sol"
    assert decision.matched_rule == "user-override"


def test_retired_models_no_match_leaves_routing_unchanged():
    """A retirement map that doesn't mention the routed model changes nothing."""
    decision = route(
        _profile(estimated_files=2), retired_models={"claude opus 5.5": "Claude Sonnet 5"}
    )
    assert decision.model == "GPT-6 Luna"
