"""Deterministic complexity scoring for a TaskProfile.

The orchestrator computes complexity from objective signals (file count band + configured
risk-flag weights) — the planner must never simply assert a complexity level.
"""

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentflow.task.profile import TaskProfile

_BAND_PATTERN = re.compile(r"^(\d+)(?:-(\d+)|\+)$")


class ComplexityLevel(str, Enum):
    """Computed complexity tier for a task."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ComplexityThreshold(BaseModel):
    """Inclusive score range mapping to a ComplexityLevel."""

    model_config = ConfigDict(extra="ignore")

    min: int = 0
    max: int | None = None


class ComplexityConfig(BaseModel):
    """Project-configurable complexity scoring rules (routing.yaml `complexity:` section)."""

    model_config = ConfigDict(extra="ignore")

    file_count: dict[str, int] = Field(default_factory=dict)
    flags: dict[str, int] = Field(default_factory=dict)
    thresholds: dict[str, ComplexityThreshold] = Field(default_factory=dict)

    @field_validator("file_count")
    @classmethod
    def _validate_file_count_bands(cls, v: dict[str, int]) -> dict[str, int]:
        for key in v:
            _parse_band(key)
        return v

    @field_validator("thresholds", mode="before")
    @classmethod
    def _normalize_threshold_keys(cls, v: object) -> object:
        if isinstance(v, dict):
            return {str(k).upper(): val for k, val in v.items()}
        return v


class ComplexityResult(BaseModel):
    """Result of scoring a TaskProfile: numeric score, tier, and contributing factors."""

    model_config = ConfigDict(extra="ignore")

    score: int
    level: ComplexityLevel
    contributing_factors: list[str]


def _parse_band(key: str) -> tuple[int, int | None]:
    """Parse a file-count band key such as '1-3' or '8+' into an inclusive (min, max) range."""
    match = _BAND_PATTERN.match(key.strip())
    if not match:
        raise ValueError(f"Invalid file_count band: '{key}'. Expected a format like '1-3' or '8+'.")
    lo = int(match.group(1))
    hi = int(match.group(2)) if match.group(2) else None
    return lo, hi


DEFAULT_COMPLEXITY_CONFIG = ComplexityConfig(
    file_count={"1-3": 0, "4-7": 1, "8+": 2},
    flags={
        "schema_change": 1,
        "api_contract_change": 1,
        "transaction_logic": 2,
        "concurrency": 2,
        "idempotency": 2,
        "authentication": 2,
        "authorization": 2,
        "data_ownership": 2,
        "payment": 3,
        "security_boundary_change": 3,
        "external_integration": 1,
        "new_dependency": 1,
        "architecture_change": 3,
        "ai_or_rag": 2,
        "performance_sensitive": 1,
    },
    thresholds={
        "LOW": ComplexityThreshold(min=0, max=2),
        "MEDIUM": ComplexityThreshold(min=3, max=5),
        "HIGH": ComplexityThreshold(min=6, max=None),
    },
)


def score_complexity(
    task_profile: TaskProfile, config: ComplexityConfig | None = None
) -> ComplexityResult:
    """Deterministically score a TaskProfile's complexity from file count band + risk flags."""
    cfg = config or DEFAULT_COMPLEXITY_CONFIG
    contributing: list[str] = []
    score = 0

    for band_key, band_score in cfg.file_count.items():
        lo, hi = _parse_band(band_key)
        if task_profile.estimated_files >= lo and (
            hi is None or task_profile.estimated_files <= hi
        ):
            score += band_score
            if band_score:
                contributing.append(f"estimated_files({band_key})=+{band_score}")
            break

    for flag_name, flag_score in cfg.flags.items():
        if getattr(task_profile, flag_name, False) is True:
            score += flag_score
            contributing.append(f"{flag_name}=+{flag_score}")

    level = _resolve_level(score, cfg)
    return ComplexityResult(score=score, level=level, contributing_factors=contributing)


def _resolve_level(score: int, config: ComplexityConfig) -> ComplexityLevel:
    """Map a numeric score to a ComplexityLevel using configured (or default) thresholds."""
    thresholds = config.thresholds or DEFAULT_COMPLEXITY_CONFIG.thresholds
    for level in (ComplexityLevel.LOW, ComplexityLevel.MEDIUM, ComplexityLevel.HIGH):
        bounds = thresholds.get(level.value)
        if bounds is None:
            continue
        if score >= bounds.min and (bounds.max is None or score <= bounds.max):
            return level
    # Score exceeds every configured upper bound: fail safe toward the highest tier.
    return ComplexityLevel.HIGH
