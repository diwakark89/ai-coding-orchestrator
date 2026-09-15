"""Unit tests for the TaskProfile machine-readable schema."""

import pytest
from pydantic import ValidationError

from agentflow.task.profile import Stage, TaskProfile


def test_task_profile_minimal_required_fields():
    """TaskProfile accepts only the required fields, defaulting all risk flags to False."""
    profile = TaskProfile(
        stage=Stage.IMPLEMENTATION,
        technologies={"python"},
        affected_layers={"backend"},
        estimated_files=2,
    )
    assert profile.stage == Stage.IMPLEMENTATION
    assert profile.schema_change is False
    assert profile.destructive_schema_change is False
    assert profile.authentication is False
    assert profile.authorization is False
    assert profile.payment is False
    assert profile.architecture_change is False


def test_task_profile_accepts_stage_as_string():
    """TaskProfile validates a plain string against the Stage enum."""
    profile = TaskProfile(
        stage="PLANNING",  # type: ignore[arg-type]
        technologies=set(),
        affected_layers=set(),
        estimated_files=0,
    )
    assert profile.stage == Stage.PLANNING


def test_task_profile_missing_required_field_rejected():
    """TaskProfile validation fails when a required field is missing."""
    with pytest.raises(ValidationError):
        TaskProfile(  # type: ignore[call-arg]
            technologies={"python"}, affected_layers={"backend"}, estimated_files=1
        )


def test_task_profile_rejects_unknown_fields():
    """TaskProfile must reject malformed/unexpected output rather than silently ignoring it."""
    with pytest.raises(ValidationError):
        TaskProfile(
            stage=Stage.IMPLEMENTATION,
            technologies=set(),
            affected_layers=set(),
            estimated_files=1,
            made_up_field=True,  # type: ignore[call-arg]
        )


def test_task_profile_rejects_negative_estimated_files():
    """estimated_files must be non-negative."""
    with pytest.raises(ValidationError):
        TaskProfile(
            stage=Stage.IMPLEMENTATION,
            technologies=set(),
            affected_layers=set(),
            estimated_files=-1,
        )


def test_task_profile_round_trips_through_json():
    """TaskProfile serializes and deserializes losslessly via JSON."""
    profile = TaskProfile(
        stage=Stage.IMPLEMENTATION,
        technologies={"python", "sqlite"},
        affected_layers={"backend", "persistence"},
        estimated_files=5,
        concurrency=True,
        idempotency=True,
    )
    restored = TaskProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
