"""Unit tests for routing configuration schema: model aliases and rule definitions."""

import pytest
from pydantic import ValidationError

from agentflow.agents.base import Provider
from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.matcher import match_complexity_rule
from agentflow.routing.rules import (
    DEFAULT_MODELS_CONFIG,
    ComplexityRule,
    ComplexityRuleWhen,
    ModelRef,
    RoleOverride,
)
from agentflow.task.profile import Stage


def test_model_ref_resolves_provider_enum():
    """ModelRef.provider_enum resolves the string provider to a Provider member."""
    ref = ModelRef(provider="anthropic", model="Claude Sonnet 5")
    assert ref.provider_enum == Provider.ANTHROPIC


def test_model_ref_rejects_excluded_model():
    """ModelRef rejects models explicitly excluded from the V1 pool."""
    with pytest.raises(ValidationError, match="excluded"):
        ModelRef(provider="openai", model="GPT-5.4 Mini")


def test_model_ref_rejects_unknown_provider():
    """ModelRef rejects a provider string that does not map to a known Provider."""
    with pytest.raises(ValidationError):
        ModelRef(provider="not-a-real-provider", model="Claude Sonnet 5")


def test_models_config_resolve_dotted_alias():
    """ModelsConfig.resolve looks up a dotted alias like 'implementation.lightweight'."""
    ref = DEFAULT_MODELS_CONFIG.resolve("implementation.lightweight")
    assert ref.model == "GPT-6 Luna"
    assert ref.provider_enum == Provider.OPENAI


def test_models_config_resolve_undefined_alias_raises():
    """Resolving an alias not present in the models: section raises ValueError."""
    with pytest.raises(ValueError, match="Undefined model alias"):
        DEFAULT_MODELS_CONFIG.resolve("implementation.nonexistent")


def test_models_config_resolve_malformed_alias_raises():
    """An alias missing the dotted category.key shape raises ValueError."""
    with pytest.raises(ValueError, match="Undefined model alias"):
        DEFAULT_MODELS_CONFIG.resolve("not-dotted")


def test_models_config_resolve_applies_retirement():
    """ModelsConfig.resolve substitutes a retired model with its replacement."""
    ref = DEFAULT_MODELS_CONFIG.resolve(
        "implementation.lightweight", retired={"gpt-6 luna": "GPT-6 Sol"}
    )
    assert ref.model == "GPT-6 Sol"
    assert ref.provider == "openai"


def test_models_config_resolve_retirement_no_match_unchanged():
    """A retirement map that doesn't mention the resolved model changes nothing."""
    ref = DEFAULT_MODELS_CONFIG.resolve(
        "implementation.lightweight", retired={"claude sonnet 5": "Claude Opus 5.5"}
    )
    assert ref.model == "GPT-6 Luna"


def test_models_config_resolve_retirement_to_excluded_model_raises():
    """Substituting to a model excluded from the V1 pool is rejected, not silently applied."""
    with pytest.raises(ValueError, match="excluded"):
        DEFAULT_MODELS_CONFIG.resolve(
            "implementation.lightweight", retired={"gpt-6 luna": "GPT-5.6 Sol"}
        )


def test_complexity_rule_when_rejects_invalid_level():
    """ComplexityRuleWhen rejects a complexity value outside low/medium/high."""
    with pytest.raises(ValidationError):
        ComplexityRuleWhen(complexity="extreme")


def test_role_override_for_stage_returns_matching_override():
    """RoleOverride.for_stage returns the ModelRef registered for that stage."""
    ref = ModelRef(provider="openai", model="GPT-6 Sol")
    override = RoleOverride(overrides={Stage.IMPLEMENTATION: ref})
    assert override.for_stage(Stage.IMPLEMENTATION) == ref


def test_role_override_for_stage_returns_none_when_unscoped():
    """RoleOverride.for_stage returns None for a stage that wasn't overridden."""
    ref = ModelRef(provider="openai", model="GPT-6 Sol")
    override = RoleOverride(overrides={Stage.IMPLEMENTATION: ref})
    assert override.for_stage(Stage.REVIEW) is None
    assert override.for_stage(Stage.DOCUMENTATION) is None
    assert override.for_stage(Stage.PLANNING) is None


def test_role_override_defaults_to_empty():
    """A RoleOverride constructed with no overrides scopes to nothing."""
    override = RoleOverride()
    assert override.for_stage(Stage.IMPLEMENTATION) is None


def test_complexity_rule_when_normalizes_case():
    """ComplexityRuleWhen lowercases its complexity value for consistent matching."""
    when = ComplexityRuleWhen(complexity="LOW")
    assert when.complexity == "low"


def test_first_match_wins_top_to_bottom():
    """When multiple rules could match, the first one listed wins."""
    rules = [
        ComplexityRule(
            id="first", when=ComplexityRuleWhen(complexity="low"), use="implementation.lightweight"
        ),
        ComplexityRule(
            id="second", when=ComplexityRuleWhen(complexity="low"), use="implementation.standard"
        ),
    ]
    matched = match_complexity_rule(rules, ComplexityLevel.LOW)
    assert matched is not None
    assert matched.id == "first"


def test_no_matching_rule_returns_none():
    """match_complexity_rule returns None when no rule's condition matches."""
    rules = [
        ComplexityRule(
            id="only-low",
            when=ComplexityRuleWhen(complexity="low"),
            use="implementation.lightweight",
        )
    ]
    assert match_complexity_rule(rules, ComplexityLevel.HIGH) is None
