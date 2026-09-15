"""Interactive terminal prompts for planner-generated clarifying questions."""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Prompt

from agentflow.ui.console import console

if TYPE_CHECKING:
    from agentflow.workflow.planning import PlannerQuestion


def ask_question(question: "PlannerQuestion", console_instance: Console | None = None) -> str:
    """Render a planner question and return the user's chosen or freely typed answer."""
    c = console_instance or console
    c.print(f"\n[bold cyan]Planner:[/bold cyan] {question.question}")

    if question.options:
        for idx, option in enumerate(question.options, start=1):
            marker = " [dim](recommended)[/dim]" if option == question.recommended else ""
            c.print(f"  [bold]{idx}.[/bold] {option}{marker}")

        choices = [str(i) for i in range(1, len(question.options) + 1)]
        default = None
        if question.recommended in question.options:
            default = str(question.options.index(question.recommended) + 1)

        selected = Prompt.ask("Choose", choices=choices, default=default, console=c) or choices[0]
        return question.options[int(selected) - 1]

    return Prompt.ask("Your answer", console=c)
