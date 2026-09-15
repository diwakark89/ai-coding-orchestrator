"""Unit tests for deterministic complexity scoring."""

import pytest
from pydantic import ValidationError

from agentflow.routing.complexity import (
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityConfig,
    ComplexityLevel,
    ComplexityThreshold,
    score_complexity,
)
from agentflow.task.profile import Stage, TaskProfile


def _profile(**overrides: object) -> TaskProfile:
    defaults: dict[str, object] = {
        "stage": Stage.IMPLEMENTATION,
        "technologies": set(),
        "affected_layers": set(),
        "estimated_files": 0,
    }
    defaults.update(overrides)
    return TaskProfile(**defaults)  # type: ignore[arg-type]


def test_low_file_count_band_scores_zero():
    """1-3 files with no risk flags scores 0 and is LOW complexity."""
    result = score_complexity(_profile(estimated_files=2))
    assert result.score == 0
    assert result.level == ComplexityLevel.LOW


def test_mid_file_count_band_contributes_score():
    """4-7 files alone contributes +1 to the score."""
    result = score_complexity(_profile(estimated_files=5))
    assert result.score == 1
    assert result.level == ComplexityLevel.LOW


def test_high_file_count_band_contributes_score():
    """8+ files alone contributes +2 to the score."""
    result = score_complexity(_profile(estimated_files=10))
    assert result.score == 2
    assert result.level == ComplexityLevel.LOW


def test_flags_and_file_count_combine_to_medium():
    """File count band plus multiple low-weight flags can reach MEDIUM complexity."""
    result = score_complexity(
        _profile(estimated_files=5, schema_change=True, api_contract_change=True)
    )
    assert result.score == 3
    assert result.level == ComplexityLevel.MEDIUM
    assert "estimated_files(4-7)=+1" in result.contributing_factors
    assert "schema_change=+1" in result.contributing_factors
    assert "api_contract_change=+1" in result.contributing_factors


def test_heavy_flags_reach_high_complexity():
    """Multiple heavily-weighted flags push complexity to HIGH."""
    result = score_complexity(
        _profile(estimated_files=1, payment=True, security_boundary_change=True)
    )
    assert result.score == 6
    assert result.level == ComplexityLevel.HIGH


def test_score_beyond_configured_thresholds_falls_back_to_high():
    """A score exceeding every configured upper bound fails safe to HIGH."""
    narrow_config = ComplexityConfig(
        file_count={"1-3": 0},
        flags={"payment": 3},
        thresholds={"LOW": ComplexityThreshold(min=0, max=1)},
    )
    result = score_complexity(_profile(estimated_files=1, payment=True), config=narrow_config)
    assert result.score == 3
    assert result.level == ComplexityLevel.HIGH


def test_unmapped_flag_in_config_is_ignored():
    """A flag configured with a score but absent/false on the TaskProfile contributes nothing."""
    result = score_complexity(_profile(estimated_files=2))
    assert not any("concurrency" in factor for factor in result.contributing_factors)


def test_default_complexity_config_matches_tdd_example():
    """The built-in default configuration mirrors the TDD reference example exactly."""
    assert DEFAULT_COMPLEXITY_CONFIG.file_count == {"1-3": 0, "4-7": 1, "8+": 2}
    assert DEFAULT_COMPLEXITY_CONFIG.flags["payment"] == 3
    assert DEFAULT_COMPLEXITY_CONFIG.thresholds["HIGH"].min == 6
    assert DEFAULT_COMPLEXITY_CONFIG.thresholds["HIGH"].max is None


def test_invalid_file_count_band_rejected():
    """A malformed file_count band key is rejected at config validation time."""
    with pytest.raises(ValidationError):
        ComplexityConfig(file_count={"not-a-band": 1})


def test_threshold_keys_normalized_case_insensitively():
    """Threshold section keys from YAML (lowercase) are normalized for lookup."""
    config = ComplexityConfig(
        file_count={"1-3": 1},
        thresholds={"low": ComplexityThreshold(min=0, max=0), "high": ComplexityThreshold(min=1)},
    )
    result = score_complexity(_profile(estimated_files=1), config=config)
    assert result.score == 1
    assert result.level == ComplexityLevel.HIGH
