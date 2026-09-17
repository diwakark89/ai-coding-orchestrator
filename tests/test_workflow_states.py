"""Unit tests for the AgentFlow workflow state machine."""

from pathlib import Path

import pytest

from agentflow.errors import InvalidStateTransitionError
from agentflow.persistence.database import DatabaseManager
from agentflow.workflow.states import WorkflowState, transition_run_state, validate_transition


def test_valid_transition_new_to_project_ready():
    """NEW may transition to PROJECT_READY."""
    validate_transition(WorkflowState.NEW, WorkflowState.PROJECT_READY)


def test_valid_planning_question_loop_transitions():
    """PLANNING <-> WAITING_FOR_USER round trip is legal."""
    validate_transition(WorkflowState.PLANNING, WorkflowState.WAITING_FOR_USER)
    validate_transition(WorkflowState.WAITING_FOR_USER, WorkflowState.PLANNING)


def test_valid_plan_ready_to_plan_approved():
    """PLAN_READY may transition to PLAN_APPROVED once the user approves."""
    validate_transition(WorkflowState.PLAN_READY, WorkflowState.PLAN_APPROVED)


def test_valid_plan_approved_to_task_classified():
    """PLAN_APPROVED may transition to TASK_CLASSIFIED once the TaskProfile is generated."""
    validate_transition(WorkflowState.PLAN_APPROVED, WorkflowState.TASK_CLASSIFIED)


def test_valid_continue_planning_from_plan_ready():
    """PLAN_READY may transition back to PLANNING when the user requests changes."""
    validate_transition(WorkflowState.PLAN_READY, WorkflowState.PLANNING)


def test_invalid_transition_skips_planning():
    """NEW cannot jump directly to PLAN_APPROVED, skipping planning entirely."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.NEW, WorkflowState.PLAN_APPROVED)


def test_invalid_transition_from_cancelled():
    """CANCELLED is terminal and accepts no further transitions."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.CANCELLED, WorkflowState.PLANNING)


def test_invalid_self_transition_not_implicitly_allowed():
    """A state is not implicitly allowed to transition to itself."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.PLANNING, WorkflowState.PLANNING)


def test_valid_task_classified_through_implementation_pipeline():
    """TASK_CLASSIFIED -> WORKTREE_READY -> IMPLEMENTING -> VERIFYING is a legal chain."""
    validate_transition(WorkflowState.TASK_CLASSIFIED, WorkflowState.WORKTREE_READY)
    validate_transition(WorkflowState.WORKTREE_READY, WorkflowState.IMPLEMENTING)
    validate_transition(WorkflowState.IMPLEMENTING, WorkflowState.VERIFYING)


def test_valid_verify_repair_loop():
    """VERIFYING <-> REPAIRING round trip is legal for the bounded repair loop."""
    validate_transition(WorkflowState.VERIFYING, WorkflowState.REPAIRING)
    validate_transition(WorkflowState.REPAIRING, WorkflowState.VERIFYING)


def test_invalid_transition_skips_implementation():
    """TASK_CLASSIFIED cannot jump directly to VERIFYING, skipping WORKTREE_READY/IMPLEMENTING."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.TASK_CLASSIFIED, WorkflowState.VERIFYING)


def test_invalid_transition_from_worktree_ready_to_repairing():
    """WORKTREE_READY cannot jump directly to REPAIRING without an implementation attempt."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.WORKTREE_READY, WorkflowState.REPAIRING)


def test_valid_verification_to_review_pipeline():
    """VERIFYING -> REVIEWING -> REVIEW_APPROVED -> READY_FOR_APPROVAL -> COMPLETED is legal."""
    validate_transition(WorkflowState.VERIFYING, WorkflowState.REVIEWING)
    validate_transition(WorkflowState.REVIEWING, WorkflowState.REVIEW_APPROVED)
    validate_transition(WorkflowState.REVIEW_APPROVED, WorkflowState.READY_FOR_APPROVAL)
    validate_transition(WorkflowState.READY_FOR_APPROVAL, WorkflowState.COMPLETED)


def test_valid_review_fix_loop():
    """REVIEWING -> REVIEW_FIXING -> VERIFYING -> REVIEWING is a legal mandatory-fix round trip."""
    validate_transition(WorkflowState.REVIEWING, WorkflowState.REVIEW_FIXING)
    validate_transition(WorkflowState.REVIEW_FIXING, WorkflowState.VERIFYING)
    validate_transition(WorkflowState.VERIFYING, WorkflowState.REVIEWING)


def test_valid_ready_for_approval_to_cancelled():
    """READY_FOR_APPROVAL may transition to CANCELLED if the user declines at final approval."""
    validate_transition(WorkflowState.READY_FOR_APPROVAL, WorkflowState.CANCELLED)


def test_invalid_transition_skips_review():
    """VERIFYING cannot jump directly to REVIEW_APPROVED, skipping the actual review."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.VERIFYING, WorkflowState.REVIEW_APPROVED)


def test_invalid_transition_from_completed():
    """COMPLETED is terminal and accepts no further transitions."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.COMPLETED, WorkflowState.CANCELLED)


def test_invalid_transition_skips_final_approval():
    """REVIEW_APPROVED cannot jump directly to COMPLETED, skipping READY_FOR_APPROVAL."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.REVIEW_APPROVED, WorkflowState.COMPLETED)


def test_valid_documentation_pipeline():
    """REVIEW_APPROVED -> DOCUMENTING -> READY_FOR_APPROVAL is legal when docs run."""
    validate_transition(WorkflowState.REVIEW_APPROVED, WorkflowState.DOCUMENTING)
    validate_transition(WorkflowState.DOCUMENTING, WorkflowState.READY_FOR_APPROVAL)


def test_valid_documentation_skip_still_legal():
    """REVIEW_APPROVED -> READY_FOR_APPROVAL stays legal when documentation is skipped."""
    validate_transition(WorkflowState.REVIEW_APPROVED, WorkflowState.READY_FOR_APPROVAL)


def test_valid_documentation_blocked():
    """DOCUMENTING may transition to BLOCKED if the documentation agent fails."""
    validate_transition(WorkflowState.DOCUMENTING, WorkflowState.BLOCKED)


def test_invalid_transition_skips_documenting_and_approval():
    """REVIEWING cannot jump directly to DOCUMENTING, skipping REVIEW_APPROVED."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.REVIEWING, WorkflowState.DOCUMENTING)


def test_transition_run_state_persists_and_validates(tmp_path: Path):
    """transition_run_state validates against the run's current state and persists the change."""
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", "/path/to/one")
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")

    transition_run_state(db, "run_1", WorkflowState.PROJECT_READY, "discovered")
    run = db.get_run("run_1")
    assert run is not None
    assert run.state == "PROJECT_READY"


def test_transition_run_state_rejects_illegal_transition(tmp_path: Path):
    """transition_run_state raises InvalidStateTransitionError for an illegal jump."""
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", "/path/to/one")
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")

    with pytest.raises(InvalidStateTransitionError):
        transition_run_state(db, "run_1", WorkflowState.VERIFYING, "skip ahead")
