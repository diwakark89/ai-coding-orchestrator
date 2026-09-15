"""Unit tests for the AgentFlow workflow state machine."""

import pytest

from agentflow.errors import InvalidStateTransitionError
from agentflow.workflow.states import WorkflowState, validate_transition


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


def test_invalid_transition_from_terminal_state():
    """TASK_CLASSIFIED is terminal for Phase 3 and accepts no further transitions."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.TASK_CLASSIFIED, WorkflowState.PLANNING)


def test_invalid_transition_from_cancelled():
    """CANCELLED is terminal and accepts no further transitions."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.CANCELLED, WorkflowState.PLANNING)


def test_invalid_self_transition_not_implicitly_allowed():
    """A state is not implicitly allowed to transition to itself."""
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(WorkflowState.PLANNING, WorkflowState.PLANNING)
