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
)


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
    assert ref.model == "GPT-5.6 Luna"
    assert ref.provider_enum == Provider.OPENAI


def test_models_config_resolve_undefined_alias_raises():
    """Resolving an alias not present in the models: section raises ValueError."""
    with pytest.raises(ValueError, match="Undefined model alias"):
        DEFAULT_MODELS_CONFIG.resolve("implementation.nonexistent")


def test_models_config_resolve_malformed_alias_raises():
    """An alias missing the dotted category.key shape raises ValueError."""
    with pytest.raises(ValueError, match="Undefined model alias"):
        DEFAULT_MODELS_CONFIG.resolve("not-dotted")


def test_complexity_rule_when_rejects_invalid_level():
    """ComplexityRuleWhen rejects a complexity value outside low/medium/high."""
    with pytest.raises(ValidationError):
        ComplexityRuleWhen(complexity="extreme")


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
