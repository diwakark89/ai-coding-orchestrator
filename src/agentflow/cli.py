"""Command line interface for AgentFlow."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from agentflow import __version__
from agentflow.application import Application
from agentflow.errors import AgentFlowError, ProjectNotFoundError
from agentflow.ui.console import console
from agentflow.workflow.states import WorkflowState

app = typer.Typer(
    name="agentflow",
    help="AgentFlow — Reusable Local Multi-Agent Coding Orchestrator",
    no_args_is_help=True,
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Callback for --version flag."""
    if value:
        console.print(f"agentflow {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show AgentFlow version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project repository path (overrides auto-discovery).",
        ),
    ] = None,
) -> None:
    """AgentFlow CLI — Local multi-agent coding orchestrator."""
    ctx.ensure_object(dict)
    ctx.obj["project"] = project


@app.command(name="doctor")
def doctor_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
) -> None:
    """Verify toolchains, environment requirements, and persistent storage."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    report = app_instance.run_doctor(project_path=target_project)
    app_instance.render_doctor_report(report)
    if report.has_critical_failures:
        raise typer.Exit(code=1)


@app.command(name="status")
def status_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
) -> None:
    """Show orchestration status and any active runs for the project."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        msg = app_instance.get_status_message(project_path=target_project)
        app_instance.ui.console.print(msg)
    except ProjectNotFoundError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1)


@app.command(name="run")
def run_cmd(
    ctx: typer.Context,
    task: Annotated[
        str,
        typer.Argument(help="Natural-language description of the requested change."),
    ],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
) -> None:
    """Start an interactive planning run for a task (Phase 3: planning only, no code changes)."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()

    try:
        outcome = asyncio.run(
            app_instance.run_planning(task_description=task, project_path=target_project)
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    if outcome.state == WorkflowState.TASK_CLASSIFIED:
        app_instance.ui.print_success(
            f"Run {outcome.run_id} reached TASK_CLASSIFIED.\n"
            f"  Plan:        {outcome.plan_path}\n"
            f"  TaskProfile: {outcome.task_profile_path}"
        )
        return

    if outcome.state == WorkflowState.CANCELLED:
        app_instance.ui.print_warning(f"Run {outcome.run_id} cancelled by user.")
        raise typer.Exit(code=1)

    detail = f": {outcome.blocker_reason}" if outcome.blocker_reason else ""
    app_instance.ui.print_error(
        f"Run {outcome.run_id} ended in state {outcome.state.value}{detail}"
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
