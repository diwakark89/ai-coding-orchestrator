"""Interactive plan approval prompt."""

from enum import Enum

from rich.console import Console
from rich.prompt import Prompt

from agentflow.ui.console import console


class ApprovalDecision(str, Enum):
    """User decision when presented with a ready plan."""

    APPROVE = "approve"
    CONTINUE_PLANNING = "continue"
    CANCEL = "cancel"


def ask_plan_approval(console_instance: Console | None = None) -> ApprovalDecision:
    """Prompt the user to approve, continue refining, or cancel the current plan."""
    c = console_instance or console
    choice = Prompt.ask(
        "\n[bold]Approve this plan?[/bold] (approve / continue / cancel)",
        choices=["approve", "continue", "cancel"],
        default="approve",
        console=c,
    )
    return ApprovalDecision(choice)


def ask_plan_feedback(console_instance: Console | None = None) -> str:
    """Prompt the user for feedback to send back to the planner."""
    c = console_instance or console
    return Prompt.ask("What should the planner change?", console=c)
