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

    MERGE = "merge"
    DIFF = "diff"
    KEEP_WORKTREE = "keep_worktree"
    CANCEL = "cancel"


def ask_final_approval(
    console_instance: Console | None = None, merge_available: bool = True
) -> FinalApprovalDecision:
    """Prompt the user to merge, view the diff, keep the worktree for inspection, or cancel.

    Merging happens only on this explicit choice; AgentFlow never pushes or deploys.
    """
    c = console_instance or console
    choices = [
        *(["merge"] if merge_available else []),
        "diff",
        "keep_worktree",
        "cancel",
    ]
    choice = Prompt.ask(
        f"\n[bold]Approve completion?[/bold] ({' / '.join(choices)})",
        choices=choices,
        default=choices[0] if merge_available else "keep_worktree",
        console=c,
    )
    return FinalApprovalDecision(choice)


def ask_commit_subject(default: str, console_instance: Console | None = None) -> str:
    """Let the user confirm or edit the commit subject for a merge."""
    c = console_instance or console
    return Prompt.ask("Commit subject", default=default, console=c).strip() or default


def confirm_merge(console_instance: Console | None = None) -> bool:
    """Confirmation for `agentflow merge`."""
    c = console_instance or console
    return Prompt.ask("Merge now?", choices=["y", "n"], default="y", console=c) == "y"
