"""Command line interface for AgentFlow."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from agentflow import __version__
from agentflow.agents.base import Provider
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
    epilog="""Examples:
  agentflow init
  agentflow complete "Add a health check endpoint"

Run 'agentflow COMMAND --help' for that command's options and examples.""",
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

# Friendly CLI names -> provider, for --providers. agy/antigravity are Google's Antigravity CLI
# (a distinct binary/dialect from `gemini`, but the same routing provider -- see `dialect:` in
# .ai-orchestrator/routing.yaml for which one actually gets invoked).
_PROVIDER_ALIASES: dict[str, Provider] = {
    "claude": Provider.ANTHROPIC,
    "anthropic": Provider.ANTHROPIC,
    "codex": Provider.OPENAI,
    "openai": Provider.OPENAI,
    "gemini": Provider.GOOGLE,
    "agy": Provider.GOOGLE,
    "antigravity": Provider.GOOGLE,
    "google": Provider.GOOGLE,
}


def _parse_providers_flag(app_instance: Application, raw: str | None) -> set[Provider] | None:
    """Parse --providers into a set of Provider, or None to trigger auto-detection."""
    if raw is None:
        return None
    providers: set[Provider] = set()
    for name in raw.split(","):
        key = name.strip().lower()
        if not key:
            continue
        provider = _PROVIDER_ALIASES.get(key)
        if provider is None:
            app_instance.ui.print_error(
                f"Invalid --providers entry '{name}'. Expected one of: "
                f"{', '.join(sorted(_PROVIDER_ALIASES))}."
            )
            raise typer.Exit(code=1)
        providers.add(provider)
    if not providers:
        app_instance.ui.print_error("--providers was given but named no valid provider.")
        raise typer.Exit(code=1)
    return providers


def _resolve_task_description(
    app_instance: Application, task: str | None, file: Path | None
) -> str:
    """Resolve the task description from the TASK argument or --file/-f, not both.

    --file is the escape hatch for a task description that's long, multi-paragraph, or
    otherwise awkward to type as a single shell argument -- point it at a plain-text or
    Markdown file and its full contents become the task description.
    """
    if task is not None and file is not None:
        app_instance.ui.print_error("Provide the task as an argument or via --file, not both.")
        raise typer.Exit(code=1)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as e:
            app_instance.ui.print_error(f"Could not read --file '{file}': {e}")
            raise typer.Exit(code=1) from e
        text = text.strip()
        if not text:
            app_instance.ui.print_error(f"--file '{file}' is empty.")
            raise typer.Exit(code=1)
        return text
    if task is None or not task.strip():
        app_instance.ui.print_error("Provide a task description, or use --file/-f.")
        raise typer.Exit(code=1)
    return task


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
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
) -> None:
    """AgentFlow CLI — Local multi-agent coding orchestrator."""
    ctx.ensure_object(dict)
    ctx.obj["project"] = project


@app.command(
    name="doctor",
    epilog="""Examples:
  agentflow doctor
  agentflow doctor --project ../other-repo""",
)
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


@app.command(
    name="init",
    epilog="""Examples:
  agentflow init
  agentflow init --providers claude,codex --force""",
)
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
    providers: Annotated[
        str | None,
        typer.Option(
            "--providers",
            help=(
                "Comma-separated coding-agent CLIs to generate the profile for "
                "(claude/codex/gemini, or agy/antigravity, or anthropic/openai/google). "
                "Default: auto-detect whichever are actually on PATH."
            ),
        ),
    ] = None,
) -> None:
    """Generate a starter .ai-orchestrator/routing.yaml, seeded from detected project structure.

    Detection is a heuristic starting point only -- always review the generated file before
    running real tasks against it. Without --providers, only CLIs actually found on PATH are
    used, so the profile is runnable even if you don't have all of claude/codex/gemini installed.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    resolved_providers = _parse_providers_flag(app_instance, providers)
    try:
        result = app_instance.run_init(
            project_path=target_project, force=force, providers=resolved_providers
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e

    app_instance.ui.print_info(f"Providers: {', '.join(result.providers)}")
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


@app.command(
    name="status",
    epilog="""Examples:
  agentflow status
  agentflow status --project ../other-repo""",
)
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


@app.command(
    name="plan",
    epilog="""Examples:
  agentflow plan "Add a health check endpoint"
  agentflow plan "Add rate limiting to the login route" --project ../other-repo
  agentflow plan --file task.md""",
)
def plan_cmd(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(help="Natural-language description of the requested change."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Read the task description from this file instead of the TASK argument.",
        ),
    ] = None,
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
    task_description = _resolve_task_description(app_instance, task, file)

    try:
        outcome = asyncio.run(
            app_instance.run_planning(
                task_description=task_description, project_path=target_project
            )
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
        app_instance.ui.print_info(
            "Next step -- continue this run through implementation, verification, review, "
            "documentation, and a final approval prompt:\n"
            f"    agentflow resume {outcome.run_id}\n"
            "Or run the full pipeline end-to-end from scratch in one command next time:\n"
            f'    agentflow complete "{task_description}"'
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


@app.command(
    name="route",
    epilog="""Examples:
  agentflow route "Add a health check endpoint"
  agentflow route "Fix flaky test" --override-provider openai --override-model 'GPT-6 Sol'
  agentflow route --file task.md""",
)
def route_cmd(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(help="Natural-language description of the requested change."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Read the task description from this file instead of the TASK argument.",
        ),
    ] = None,
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
    task_description = _resolve_task_description(app_instance, task, file)
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_routing(
                task_description=task_description,
                project_path=target_project,
                role_override=role_override,
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


@app.command(
    name="implement",
    epilog="""Examples:
  agentflow implement "Add a health check endpoint"
  agentflow implement "Fix flaky test" --override-provider openai --override-model 'GPT-6 Sol'
  agentflow implement --file task.md""",
)
def implement_cmd(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(help="Natural-language description of the requested change."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Read the task description from this file instead of the TASK argument.",
        ),
    ] = None,
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
    task_description = _resolve_task_description(app_instance, task, file)
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_implementation(
                task_description=task_description,
                project_path=target_project,
                role_override=role_override,
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


@app.command(
    name="complete",
    epilog="""Examples:
  agentflow complete "Add a health check endpoint"
  agentflow complete "Fix flaky test" --override-provider openai --override-model 'GPT-6 Sol'
  agentflow complete --file task.md""",
)
def complete_cmd(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(help="Natural-language description of the requested change."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Read the task description from this file instead of the TASK argument.",
        ),
    ] = None,
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

    Merges only when you choose `merge` at the final approval prompt; never pushes or deploys.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    task_description = _resolve_task_description(app_instance, task, file)
    role_override = _build_role_override(
        app_instance, override_provider, override_model, override_stage
    )

    try:
        outcome = asyncio.run(
            app_instance.run_pipeline(
                task_description=task_description,
                project_path=target_project,
                role_override=role_override,
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


@app.command(
    name="resume",
    epilog="""Examples:
  agentflow resume RUN-AB12CD34
  agentflow resume RUN-AB12CD34 --override-provider openai --override-model 'GPT-6 Sol'""",
)
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


@app.command(
    name="merge",
    epilog="""Examples:
  agentflow merge RUN-AB12CD34
  agentflow merge RUN-AB12CD34 --yes""",
)
def merge_cmd(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Completed run whose changes to merge.")],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            "-C",
            help="Target project directory path (overrides auto-discovery).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Merge without the confirmation prompt."),
    ] = False,
) -> None:
    """Squash a completed run into one commit on the branch it started from.

    Your checkout must be on that branch with no uncommitted changes to tracked files.
    On conflicts nothing is changed and the worktree is kept. Never pushes.
    """
    global_project = ctx.obj.get("project") if ctx.obj else None
    target_project = project or global_project
    app_instance = Application()
    try:
        outcome = asyncio.run(
            app_instance.merge_completed_run(
                run_id,
                project_path=target_project,
                confirm=(lambda: True) if yes else None,
            )
        )
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e
    if not outcome.merged:
        raise typer.Exit(code=1)


@app.command(
    name="runs",
    epilog="""Examples:
  agentflow runs
  agentflow runs --all --limit 50""",
)
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


@app.command(
    name="cleanup",
    epilog="""Examples:
  agentflow cleanup
  agentflow cleanup --project ../other-repo""",
)
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


@app.command(
    name="stats",
    epilog="""Examples:
  agentflow stats
  agentflow stats --all""",
)
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


@app.command(
    name="retire",
    epilog="""Examples:
  agentflow retire "GPT-5.6 Terra" "GPT-6 Sol"
  agentflow retire --list""",
)
def retire_cmd(
    old_model: Annotated[
        str | None,
        typer.Argument(help="Model name to retire, e.g. 'GPT-5.6 Terra'."),
    ] = None,
    new_model: Annotated[
        str | None,
        typer.Argument(help="Replacement model name, e.g. 'GPT-6 Sol'."),
    ] = None,
    list_: Annotated[
        bool,
        typer.Option("--list", help="List current model retirements."),
    ] = False,
    remove: Annotated[
        str | None,
        typer.Option("--remove", help="Un-retire this model name."),
    ] = None,
) -> None:
    """Retire a model globally: every project transparently uses the replacement from now on,
    with no need to edit any project's routing.yaml."""
    app_instance = Application()

    if list_:
        app_instance.render_retirements(app_instance.list_retirements())
        return

    if remove is not None:
        removed = app_instance.remove_retirement(remove)
        if removed:
            app_instance.ui.print_success(f"'{remove}' is no longer retired.")
        else:
            app_instance.ui.print_warning(f"'{remove}' was not retired; nothing to remove.")
        return

    if old_model is None or new_model is None:
        app_instance.ui.print_error(
            "Provide both OLD_MODEL and NEW_MODEL, or use --list / --remove."
        )
        raise typer.Exit(code=1)

    try:
        report = app_instance.run_retire(old_model, new_model)
    except AgentFlowError as e:
        app_instance.ui.print_error(str(e))
        raise typer.Exit(code=1) from e
    app_instance.ui.print_success(f"'{old_model}' retired; replaced by '{new_model}'.")
    app_instance.render_retirements(report)


if __name__ == "__main__":
    app()
