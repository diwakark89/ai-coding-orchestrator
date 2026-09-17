"""Command line interface for AgentFlow."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from agentflow import __version__
from agentflow.application import Application
from agentflow.errors import AgentFlowError, ProjectNotFoundError
from agentflow.routing.rules import ModelRef, RoleOverride
from agentflow.task.profile import Stage
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


_OVERRIDE_STAGE_CHOICES: dict[str, Stage] = {
    "implementation": Stage.IMPLEMENTATION,
    "review": Stage.REVIEW,
    "documentation": Stage.DOCUMENTATION,
}


def _build_role_override(
    app_instance: Application,
    override_provider: str | None,
    override_model: str | None,
    override_stage: str | None,
) -> RoleOverride | None:
    """Build a RoleOverride from --override-provider/--override-model/--override-stage.

    --override-stage defaults to "implementation" when omitted -- this is the one stage the
    override has ever actually reached, so omitting the flag preserves prior behavior exactly.
    """
    if not (override_provider or override_model):
        return None
    if not (override_provider and override_model):
        app_instance.ui.print_error(
            "--override-provider and --override-model must both be provided together."
        )
        raise typer.Exit(code=1)
    stage_key = (override_stage or "implementation").strip().lower()
    stage = _OVERRIDE_STAGE_CHOICES.get(stage_key)
    if stage is None:
        app_instance.ui.print_error(
            f"Invalid --override-stage '{override_stage}'. Expected one of: "
            f"{', '.join(sorted(_OVERRIDE_STAGE_CHOICES))}."
        )
        raise typer.Exit(code=1)
    try:
        model_ref = ModelRef(provider=override_provider, model=override_model)
    except ValidationError as e:
        app_instance.ui.print_error(f"Invalid model override: {e}")
        raise typer.Exit(code=1) from e
    return RoleOverride(overrides={stage: model_ref})


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


@app.command(name="init")
def init_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing .ai-orchestrator/routing.yaml."),
    ] = False,
) -> None:
    """Generate a starter .ai-orchestrator/routing.yaml, seeded from detected project structure.

    Detection is a heuristic starting point only -- always review the generated file before
    running real tasks against it.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        result = app_instance.run_init(project_path=target_project, force=force)
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    if result.detected_groups:
        groups_text = ", ".join(result.detected_groups)
        app_instance.ui.print_info(f"Detected verification groups: {groups_text}")
    else:
        app_instance.ui.print_info(
            "No recognized project markers detected -- verification: left empty."
        )
    action = "Overwrote" if result.overwritten else "Wrote"
    app_instance.ui.print_success(
        f"{action} starter profile at {result.path}.\n"
        "Review it -- especially the verification: commands and documentation: section -- "
        "before running real tasks against this project."
    )


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


@app.command(name="route")
def route_cmd(
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
    override_provider: Annotated[
        str | None,
        typer.Option(
            "--override-provider",
            help="Force a specific provider (anthropic/openai/google), bypassing routing rules.",
        ),
    ] = None,
    override_model: Annotated[
        str | None,
        typer.Option(
            "--override-model",
            help="Force a specific model, bypassing routing rules. Requires --override-provider.",
        ),
    ] = None,
    override_stage: Annotated[
        str | None,
        typer.Option(
            "--override-stage",
            help=(
                "Which stage the override applies to: implementation (default), review, "
                "or documentation."
            ),
        ),
    ] = None,
) -> None:
    """Classify a task and display its deterministic routing decision (no code changes)."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_routing(
                task_description=task, project_path=target_project, role_override=role_override
            )
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    planning = outcome.planning_outcome
    if planning.state != WorkflowState.TASK_CLASSIFIED or outcome.decision is None:
        detail = f": {planning.blocker_reason}" if planning.blocker_reason else ""
        app_instance.ui.print_error(
            f"Run {planning.run_id} ended in state {planning.state.value}{detail}"
        )
        raise typer.Exit(code=1)

    app_instance.render_routing_decision(outcome.decision)
    app_instance.ui.print_success(f"Routing decision persisted to {outcome.decision_path}")


@app.command(name="implement")
def implement_cmd(
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
    override_provider: Annotated[
        str | None,
        typer.Option(
            "--override-provider",
            help="Force a specific provider (anthropic/openai/google), bypassing routing rules.",
        ),
    ] = None,
    override_model: Annotated[
        str | None,
        typer.Option(
            "--override-model",
            help="Force a specific model, bypassing routing rules. Requires --override-provider.",
        ),
    ] = None,
    override_stage: Annotated[
        str | None,
        typer.Option(
            "--override-stage",
            help=(
                "Which stage the override applies to: implementation (default), review, "
                "or documentation."
            ),
        ),
    ] = None,
) -> None:
    """Plan, route, implement in an isolated worktree, then verify and repair.

    The primary working tree is never modified — all code changes happen in a disposable
    Git worktree under the configured worktrees root.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_implementation(
                task_description=task, project_path=target_project, role_override=role_override
            )
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    if (
        outcome.state == WorkflowState.VERIFYING
        and outcome.verification is not None
        and outcome.verification.success
    ):
        worktree_path = (
            outcome.implementation.worktree.path
            if outcome.implementation and outcome.implementation.worktree
            else None
        )
        app_instance.ui.print_success(
            f"Run {outcome.planning_outcome.run_id}: implementation verified successfully.\n"
            f"  Worktree:    {worktree_path}\n"
            f"  Verification: {outcome.verification_path}"
        )
        return

    detail = f": {outcome.blocker_reason}" if outcome.blocker_reason else ""
    app_instance.ui.print_error(
        f"Run {outcome.planning_outcome.run_id} ended in state {outcome.state.value}{detail}"
    )
    raise typer.Exit(code=1)


@app.command(name="complete")
def complete_cmd(
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
    override_provider: Annotated[
        str | None,
        typer.Option(
            "--override-provider",
            help="Force a specific provider (anthropic/openai/google), bypassing routing rules.",
        ),
    ] = None,
    override_model: Annotated[
        str | None,
        typer.Option(
            "--override-model",
            help="Force a specific model, bypassing routing rules. Requires --override-provider.",
        ),
    ] = None,
    override_stage: Annotated[
        str | None,
        typer.Option(
            "--override-stage",
            help=(
                "Which stage the override applies to: implementation (default), review, "
                "or documentation."
            ),
        ),
    ] = None,
) -> None:
    """Run the full pipeline: plan, route, implement, verify/repair, review, document, approve.

    V1 never auto-pushes, auto-merges, or deploys -- completion always requires human approval.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_pipeline(
                task_description=task, project_path=target_project, role_override=role_override
            )
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    run_id = outcome.implementation_outcome.planning_outcome.run_id

    if outcome.state == WorkflowState.COMPLETED:
        app_instance.ui.print_success(
            f"Run {run_id} COMPLETED.\n  Final summary: {outcome.final_summary_path}"
        )
        return

    if outcome.state == WorkflowState.CANCELLED:
        app_instance.ui.print_warning(f"Run {run_id} cancelled by user at final approval.")
        raise typer.Exit(code=1)

    detail = f": {outcome.blocker_reason}" if outcome.blocker_reason else ""
    app_instance.ui.print_error(f"Run {run_id} ended in state {outcome.state.value}{detail}")
    raise typer.Exit(code=1)


@app.command(name="resume")
def resume_cmd(
    ctx: typer.Context,
    run_id: Annotated[
        str,
        typer.Argument(help="Run ID to resume, e.g. RUN-AB12CD34 (see `agentflow runs`)."),
    ],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
    override_provider: Annotated[
        str | None,
        typer.Option(
            "--override-provider",
            help="Force a specific provider (anthropic/openai/google), bypassing routing rules.",
        ),
    ] = None,
    override_model: Annotated[
        str | None,
        typer.Option(
            "--override-model",
            help="Force a specific model, bypassing routing rules. Requires --override-provider.",
        ),
    ] = None,
    override_stage: Annotated[
        str | None,
        typer.Option(
            "--override-stage",
            help=(
                "Which stage the override applies to: implementation (default), review, "
                "or documentation. Check `agentflow runs`/`status` for the Stage a BLOCKED "
                "run was in and pass that here, or the override won't reach it."
            ),
        ),
    ] = None,
) -> None:
    """Resume a run interrupted by a crash, or a BLOCKED run, and continue it safely.

    Reuses any existing worktree changes rather than blindly redoing completed work. A BLOCKED
    run (e.g. a coding-agent CLI hit a usage limit) is retried from the stage it was in when it
    blocked -- pass --override-provider/--override-model (and --override-stage matching that
    stage) to route the retry elsewhere. Refuses to resume a run still actively controlled by
    another live process, or one that already ended in COMPLETED/FAILED/CANCELLED.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_resume(
                run_id, project_path=target_project, role_override=role_override
            )
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    if outcome.state == WorkflowState.COMPLETED:
        app_instance.ui.print_success(
            f"Run {run_id} COMPLETED.\n  Final summary: {outcome.final_summary_path}"
        )
        return

    if outcome.state == WorkflowState.CANCELLED:
        app_instance.ui.print_warning(f"Run {run_id} cancelled by user at final approval.")
        raise typer.Exit(code=1)

    detail = f": {outcome.blocker_reason}" if outcome.blocker_reason else ""
    app_instance.ui.print_error(f"Run {run_id} ended in state {outcome.state.value}{detail}")
    raise typer.Exit(code=1)


@app.command(name="runs")
def runs_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
    all_projects: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="List runs across all tracked projects instead of just the current one.",
        ),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of runs to show.")] = 20,
) -> None:
    """List recent runs with their status, current stage, and timestamps."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        app_instance.render_runs(project_path=target_project, limit=limit, show_all=all_projects)
    except ProjectNotFoundError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1)


@app.command(name="cleanup")
def cleanup_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Restrict cleanup to a single project (default: all tracked projects).",
        ),
    ] = None,
) -> None:
    """Remove worktrees/logs for completed, blocked, or cancelled runs, and clear stale locks.

    Never removes a worktree still guarded by a live writer lock, regardless of run status.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        report = asyncio.run(app_instance.run_cleanup(project_path=target_project))
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e
    app_instance.render_cleanup_report(report)


@app.command(name="stats")
def stats_cmd(
    ctx: typer.Context,
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
    all_projects: Annotated[
        bool,
        typer.Option("--all", "-a", help="Report across all tracked projects."),
    ] = False,
) -> None:
    """Show local routing and workflow metrics; this never changes routing rules."""
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        report = app_instance.get_statistics(project_path=target_project, show_all=all_projects)
    except ProjectNotFoundError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e
    app_instance.render_statistics(report)


if __name__ == "__main__":
    app()
