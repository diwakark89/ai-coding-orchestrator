"""Unit tests for the global model-retirement substitution function."""

import pytest

from agentflow.config.retirement import apply_retirements
from agentflow.errors import ConfigurationError


def test_apply_retirements_no_match_returns_input():
    """A model not present in the retirement map is returned unchanged."""
    assert apply_retirements("Claude Sonnet 5", {}) == "Claude Sonnet 5"
    assert apply_retirements("Claude Sonnet 5", {"gpt-5.6 terra": "GPT-6 Sol"}) == "Claude Sonnet 5"


def test_apply_retirements_simple_substitution():
    """A directly retired model resolves to its replacement."""
    assert apply_retirements("GPT-5.6 Terra", {"gpt-5.6 terra": "GPT-6 Sol"}) == "GPT-6 Sol"


def test_apply_retirements_is_case_insensitive():
    """Matching against the retirement map ignores case and surrounding whitespace."""
    assert apply_retirements("  GPT-5.6 TERRA  ", {"gpt-5.6 terra": "GPT-6 Sol"}) == "GPT-6 Sol"


def test_apply_retirements_chains():
    """X -> Y -> Z resolves X (and Y) all the way to Z."""
    retired = {"x": "Y", "y": "Z"}
    assert apply_retirements("X", retired) == "Z"
    assert apply_retirements("Y", retired) == "Z"


def test_apply_retirements_detects_cycle():
    """A cycle in the retirement map raises ConfigurationError instead of looping forever."""
    retired = {"x": "Y", "y": "X"}
    with pytest.raises(ConfigurationError, match="cycle"):
        apply_retirements("X", retired)


def test_apply_retirements_detects_self_cycle():
    """A model retired to itself is treated as a cycle."""
    with pytest.raises(ConfigurationError, match="cycle"):
        apply_retirements("X", {"x": "X"})
