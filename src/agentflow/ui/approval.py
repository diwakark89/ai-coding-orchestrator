"""Interactive plan approval prompt."""

from enum import Enum

from rich.console import Console
from rich.prompt import Prompt

from agentflow.ui.console import console


class ApprovalDecision(str, Enum):
    """User decision when presented with a ready plan."""

    APPROVE = "approve"
    CONTINUE_PLANNING = "revise"
    CANCEL = "cancel"


def ask_plan_approval(console_instance: Console | None = None) -> ApprovalDecision:
    """Prompt the user to approve, revise, or cancel the current plan."""
    c = console_instance or console
    choice = Prompt.ask(
        "\n[bold]Approve this plan?[/bold] (approve / revise / cancel)",
        choices=["approve", "revise", "cancel"],
        default="approve",
        console=c,
    )
    return ApprovalDecision(choice)


def ask_plan_feedback(console_instance: Console | None = None) -> str:
    """Prompt the user for feedback to send back to the planner."""
    c = console_instance or console
    return Prompt.ask("What should the planner change?", console=c)


class FinalApprovalDecision(str, Enum):
    """User decision when presented with a run's final summary (Phase 9)."""

    APPROVE = "approve"
    KEEP_WORKTREE = "keep_worktree"
    CANCEL = "cancel"


def ask_final_approval(console_instance: Console | None = None) -> FinalApprovalDecision:
    """Prompt the user to approve completion, keep the worktree for inspection, or cancel.

    V1 never auto-pushes, auto-merges, or auto-deploys -- a human always decides here.
    """
    c = console_instance or console
    choice = Prompt.ask(
        "\n[bold]Approve completion?[/bold] (approve / keep_worktree / cancel)",
        choices=["approve", "keep_worktree", "cancel"],
        default="approve",
        console=c,
    )
    return FinalApprovalDecision(choice)
