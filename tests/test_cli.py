"""Integration tests for Typer CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from agentflow import __version__
from agentflow.agents.base import Provider
from agentflow.application import (
    Application,
    CleanupReport,
    ImplementationRunOutcome,
    InitResult,
    PipelineOutcome,
    RetirementReport,
    RoutingOutcome,
)
from agentflow.cli import app
from agentflow.errors import ConfigurationError
from agentflow.observability.metrics import StatisticsReport
from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.decision import RoutingDecision
from agentflow.task.profile import Stage
from agentflow.workflow.planning import PlanningOutcome
from agentflow.workflow.states import WorkflowState
from agentflow.workflow.verification import VerificationResult, VerificationStatus

runner = CliRunner()


def test_cli_version():
    """agentflow --version prints current version and exits with 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"agentflow {__version__}" in result.output


def test_cli_version_short():
    """agentflow -v prints current version and exits with 0."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert f"agentflow {__version__}" in result.output


def test_cli_help_shows_examples():
    """Top-level --help includes a runnable example, not just the command list."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert "agentflow init" in result.output


def test_cli_complete_help_shows_examples():
    """agentflow complete --help includes runnable examples covering the override flags."""
    result = runner.invoke(app, ["complete", "--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert 'agentflow complete "Add a health check endpoint"' in result.output


def test_cli_resume_help_shows_examples():
    """agentflow resume --help includes a run-id example and an override example."""
    result = runner.invoke(app, ["resume", "--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert "agentflow resume RUN-AB12CD34" in result.output


def test_cli_retire_help_shows_examples():
    """agentflow retire --help includes both the add and --list forms."""
    result = runner.invoke(app, ["retire", "--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert "agentflow retire --list" in result.output


def test_cli_doctor():
    """agentflow doctor executes and displays check results."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "AgentFlow Doctor" in result.output
    assert "Python runtime" in result.output


def test_cli_status():
    """agentflow status executes and displays status message."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Status:" in result.output or "Active runs" in result.output


def test_cli_stats_reports_local_metrics(monkeypatch):
    """agentflow stats delegates to informational local reporting without changing routing."""
    observed: dict[str, bool] = {}

    def fake_get_statistics(self, project_path=None, show_all=False):
        observed["show_all"] = show_all
        return StatisticsReport(total_runs=0)

    def fake_render_statistics(self, report):
        self.ui.console.print(f"Stats: {report.total_runs}")

    monkeypatch.setattr(Application, "get_statistics", fake_get_statistics)
    monkeypatch.setattr(Application, "render_statistics", fake_render_statistics)

    result = runner.invoke(app, ["stats", "--all"])

    assert result.exit_code == 0
    assert observed["show_all"]
    assert "Stats: 0" in result.output


def test_cli_with_explicit_project_flag(tmp_path: Path):
    """agentflow --project <path> status inspects specified directory."""
    (tmp_path / ".git").mkdir()
    result = runner.invoke(app, ["--project", str(tmp_path), "status"])
    assert result.exit_code == 0
    assert "No active runs found" in result.output


def test_cli_with_dash_c_flag(tmp_path: Path):
    """agentflow -C <path> doctor inspects specified directory."""
    (tmp_path / ".git").mkdir()
    result = runner.invoke(app, ["-C", str(tmp_path), "doctor"])
    assert result.exit_code == 0
    assert "AgentFlow Doctor" in result.output


def test_cli_state_isolation_between_invocations(tmp_path: Path):
    """Global option from previous invocation does not leak into subsequent invocations."""
    project_dir = tmp_path / "custom_proj"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    # Invocations 1: With -C
    res1 = runner.invoke(app, ["-C", str(project_dir), "status"])
    assert res1.exit_code == 0
    assert "custom_proj" in res1.output

    # Invocation 2: Without -C (should auto-discover from cwd, not use custom_proj)
    res2 = runner.invoke(app, ["status"])
    assert res2.exit_code == 0
    assert "custom_proj" not in res2.output


def test_cli_status_nonexistent_project_fails(tmp_path: Path):
    """status command exits with code 1 when target project does not exist."""
    missing = tmp_path / "missing_project_dir"
    result = runner.invoke(app, ["-C", str(missing), "status"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_cli_doctor_nonexistent_project_fails(tmp_path: Path):
    """doctor command exits with code 1 when target project does not exist."""
    missing = tmp_path / "missing_project_dir"
    result = runner.invoke(app, ["-C", str(missing), "doctor"])
    assert result.exit_code == 1
    assert "Critical environment requirements are missing" in result.output


def test_cli_init_reports_success(monkeypatch, tmp_path: Path):
    """agentflow init prints the providers used and the written path, and exits 0."""

    def fake_run_init(self, project_path=None, force=False, providers=None):
        return InitResult(
            path=tmp_path / ".ai-orchestrator" / "routing.yaml",
            detected_groups=["python"],
            overwritten=False,
            providers=["anthropic"],
        )

    monkeypatch.setattr(Application, "run_init", fake_run_init)
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "Providers: anthropic" in result.output
    assert "routing.yaml" in result.output


def test_cli_init_passes_parsed_providers_flag(monkeypatch, tmp_path: Path):
    """--providers is parsed into Provider values and forwarded to Application.run_init."""
    observed: dict[str, object] = {}

    def fake_run_init(self, project_path=None, force=False, providers=None):
        observed["providers"] = providers
        return InitResult(
            path=tmp_path / ".ai-orchestrator" / "routing.yaml",
            providers=sorted(p.value for p in (providers or [])),
        )

    monkeypatch.setattr(Application, "run_init", fake_run_init)
    # "claude" and "agy" are friendly aliases for anthropic and google respectively.
    result = runner.invoke(app, ["init", "--providers", "claude,agy"])

    assert result.exit_code == 0
    assert observed["providers"] == {Provider.ANTHROPIC, Provider.GOOGLE}


def test_cli_init_invalid_provider_exits_nonzero():
    """An unrecognized --providers entry fails fast with a clear error, not a stack trace."""
    result = runner.invoke(app, ["init", "--providers", "not-a-real-provider"])

    assert result.exit_code == 1
    assert "Invalid --providers entry" in result.output


def test_cli_plan_reports_task_classified(monkeypatch):
    """agentflow plan prints artifact paths and exits 0 when planning reaches TASK_CLASSIFIED."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(
            run_id="RUN-TEST01",
            state=WorkflowState.TASK_CLASSIFIED,
            plan_path=Path("approved-plan.md"),
            task_profile_path=Path("task-profile.json"),
        )

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["plan", "Add a health check endpoint"])

    assert result.exit_code == 0
    assert "RUN-TEST01" in result.output
    assert "TASK_CLASSIFIED" in result.output


def test_cli_plan_blocked_exits_nonzero(monkeypatch):
    """agentflow plan exits with code 1 and surfaces the blocker reason when planning is blocked."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(
            run_id="RUN-TEST02",
            state=WorkflowState.BLOCKED,
            blocker_reason="Planner CLI not authenticated.",
        )

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["plan", "Do something impossible"])

    assert result.exit_code == 1
    assert "not authenticated" in result.output


def test_cli_plan_cancelled_exits_nonzero(monkeypatch):
    """agentflow plan exits with code 1 when the user cancels plan approval."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(run_id="RUN-TEST03", state=WorkflowState.CANCELLED)

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["plan", "Add a feature"])

    assert result.exit_code == 1
    assert "cancelled" in result.output.lower()


def test_cli_plan_file_reads_task_description(monkeypatch, tmp_path: Path):
    """agentflow plan --file reads the task description from disk, not the TASK argument."""
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a health check endpoint.\n\nDetails here.\n", encoding="utf-8")
    observed: dict[str, object] = {}

    async def fake_run_planning(self, task_description, project_path=None):
        observed["task_description"] = task_description
        return PlanningOutcome(run_id="RUN-TEST04", state=WorkflowState.TASK_CLASSIFIED)

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["plan", "--file", str(task_file)])

    assert result.exit_code == 0
    assert observed["task_description"] == "Add a health check endpoint.\n\nDetails here."


def test_cli_plan_rejects_task_and_file_together(tmp_path: Path):
    """Providing both the TASK argument and --file is rejected, not silently resolved."""
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a feature.", encoding="utf-8")
    result = runner.invoke(app, ["plan", "Add a feature", "--file", str(task_file)])

    assert result.exit_code == 1
    assert "not both" in result.output


def test_cli_plan_rejects_neither_task_nor_file():
    """Providing neither the TASK argument nor --file fails fast with a clear error."""
    result = runner.invoke(app, ["plan"])

    assert result.exit_code == 1
    assert "Provide a task description" in result.output


def test_cli_plan_file_missing_exits_nonzero(tmp_path: Path):
    """--file pointing at a nonexistent path surfaces a clear error, not a stack trace."""
    result = runner.invoke(app, ["plan", "--file", str(tmp_path / "nonexistent.md")])

    assert result.exit_code == 1
    assert "Could not read --file" in result.output


def test_cli_plan_file_empty_exits_nonzero(tmp_path: Path):
    """--file pointing at an empty (or whitespace-only) file is rejected."""
    task_file = tmp_path / "task.md"
    task_file.write_text("   \n\n", encoding="utf-8")
    result = runner.invoke(app, ["plan", "--file", str(task_file)])

    assert result.exit_code == 1
    assert "is empty" in result.output


def test_cli_implement_file_short_flag_reads_task_description(monkeypatch, tmp_path: Path):
    """agentflow implement -f is the same option as --file on every task-taking command."""
    task_file = tmp_path / "task.md"
    task_file.write_text("Fix flaky test.", encoding="utf-8")
    observed: dict[str, object] = {}

    async def fake_run_implementation(
        self, task_description, project_path=None, role_override=None
    ):
        observed["task_description"] = task_description
        return ImplementationRunOutcome(
            planning_outcome=PlanningOutcome(
                run_id="RUN-TEST05", state=WorkflowState.TASK_CLASSIFIED
            ),
            state=WorkflowState.BLOCKED,
            blocker_reason="stopped for test",
        )

    monkeypatch.setattr(Application, "run_implementation", fake_run_implementation)
    runner.invoke(app, ["implement", "-f", str(task_file)])

    assert observed["task_description"] == "Fix flaky test."


def test_cli_route_displays_decision_and_exits_zero(monkeypatch):
    """agentflow route prints the routing decision and exits 0 on success."""

    async def fake_run_routing(self, task_description, project_path=None, role_override=None):
        planning = PlanningOutcome(run_id="RUN-ROUTE01", state=WorkflowState.TASK_CLASSIFIED)
        decision = RoutingDecision(
            stage=Stage.IMPLEMENTATION,
            provider=Provider.OPENAI,
            model="GPT-6 Sol",
            role="implementation.standard",
            matched_rule="implementation.force-standard",
            reason="Authorization requires the standard tier.",
            complexity_score=2,
            complexity=ComplexityLevel.LOW,
            risk_flags=["authorization"],
        )
        return RoutingOutcome(
            planning_outcome=planning,
            decision=decision,
            decision_path=Path("routing-decision.json"),
        )

    monkeypatch.setattr(Application, "run_routing", fake_run_routing)
    result = runner.invoke(app, ["route", "Add a secured endpoint"])

    assert result.exit_code == 0
    assert "GPT-6 Sol" in result.output
    assert "implementation.force-standard" in result.output


def test_cli_route_blocked_exits_nonzero(monkeypatch):
    """agentflow route exits 1 when planning does not reach TASK_CLASSIFIED."""

    async def fake_run_routing(self, task_description, project_path=None, role_override=None):
        planning = PlanningOutcome(
            run_id="RUN-ROUTE02", state=WorkflowState.BLOCKED, blocker_reason="nope"
        )
        return RoutingOutcome(planning_outcome=planning)

    monkeypatch.setattr(Application, "run_routing", fake_run_routing)
    result = runner.invoke(app, ["route", "Do something impossible"])

    assert result.exit_code == 1
    assert "nope" in result.output


def test_cli_route_requires_both_override_flags():
    """agentflow route rejects a lone --override-model without --override-provider."""
    result = runner.invoke(app, ["route", "Add a feature", "--override-model", "Claude Sonnet 5"])

    assert result.exit_code == 1
    assert "must both be provided" in result.output


def test_cli_implement_reports_success(monkeypatch, tmp_path: Path):
    """agentflow implement reports success and exits 0 when verification passes."""

    async def fake_run_implementation(
        self, task_description, project_path=None, role_override=None
    ):
        from agentflow.git.worktree import WorktreeHandle
        from agentflow.workflow.implementation import ImplementationOutcome

        planning = PlanningOutcome(run_id="RUN-IMPL01", state=WorkflowState.TASK_CLASSIFIED)
        worktree = WorktreeHandle(
            path=tmp_path / "worktree", branch_name="agentflow/RUN-IMPL01", repository_path=tmp_path
        )
        implementation = ImplementationOutcome(
            run_id="RUN-IMPL01", state=WorkflowState.VERIFYING, worktree=worktree
        )
        verification = VerificationResult(
            status=VerificationStatus.PASSED, groups_run=[], command_results=[]
        )
        return ImplementationRunOutcome(
            planning_outcome=planning,
            state=WorkflowState.VERIFYING,
            implementation=implementation,
            verification=verification,
            verification_path=tmp_path / "verification.json",
        )

    monkeypatch.setattr(Application, "run_implementation", fake_run_implementation)
    result = runner.invoke(app, ["implement", "Add a feature"])

    assert result.exit_code == 0
    assert "RUN-IMPL01" in result.output
    assert "verified successfully" in result.output


def test_cli_implement_blocked_exits_nonzero(monkeypatch):
    """agentflow implement exits 1 and surfaces the reason when implementation is blocked."""

    async def fake_run_implementation(
        self, task_description, project_path=None, role_override=None
    ):
        planning = PlanningOutcome(run_id="RUN-IMPL02", state=WorkflowState.TASK_CLASSIFIED)
        return ImplementationRunOutcome(
            planning_outcome=planning,
            state=WorkflowState.BLOCKED,
            blocker_reason="Codex CLI not authenticated.",
        )

    monkeypatch.setattr(Application, "run_implementation", fake_run_implementation)
    result = runner.invoke(app, ["implement", "Add a feature"])

    assert result.exit_code == 1
    assert "not authenticated" in result.output


def test_cli_complete_reports_success(monkeypatch, tmp_path: Path):
    """agentflow complete reports COMPLETED and exits 0 when the full pipeline succeeds."""

    async def fake_run_pipeline(
        self, task_description, project_path=None, role_override=None, final_approval_prompt=None
    ):
        planning = PlanningOutcome(run_id="RUN-COMPLETE01", state=WorkflowState.TASK_CLASSIFIED)
        implementation_outcome = ImplementationRunOutcome(
            planning_outcome=planning, state=WorkflowState.VERIFYING
        )
        return PipelineOutcome(
            implementation_outcome=implementation_outcome,
            state=WorkflowState.COMPLETED,
            final_summary_path=tmp_path / "final-summary.md",
        )

    monkeypatch.setattr(Application, "run_pipeline", fake_run_pipeline)
    result = runner.invoke(app, ["complete", "Add a feature"])

    assert result.exit_code == 0
    assert "RUN-COMPLETE01" in result.output
    assert "COMPLETED" in result.output


def test_cli_complete_blocked_exits_nonzero(monkeypatch):
    """agentflow complete exits 1 and surfaces the reason when the pipeline is blocked."""

    async def fake_run_pipeline(
        self, task_description, project_path=None, role_override=None, final_approval_prompt=None
    ):
        planning = PlanningOutcome(run_id="RUN-COMPLETE02", state=WorkflowState.TASK_CLASSIFIED)
        implementation_outcome = ImplementationRunOutcome(
            planning_outcome=planning, state=WorkflowState.BLOCKED
        )
        return PipelineOutcome(
            implementation_outcome=implementation_outcome,
            state=WorkflowState.BLOCKED,
            blocker_reason="Reviewer exited with code 1.",
        )

    monkeypatch.setattr(Application, "run_pipeline", fake_run_pipeline)
    result = runner.invoke(app, ["complete", "Add a feature"])

    assert result.exit_code == 1
    assert "Reviewer exited" in result.output


def test_cli_complete_cancelled_exits_nonzero(monkeypatch):
    """agentflow complete exits 1 when the user cancels at final approval."""

    async def fake_run_pipeline(
        self, task_description, project_path=None, role_override=None, final_approval_prompt=None
    ):
        planning = PlanningOutcome(run_id="RUN-COMPLETE03", state=WorkflowState.TASK_CLASSIFIED)
        implementation_outcome = ImplementationRunOutcome(
            planning_outcome=planning, state=WorkflowState.VERIFYING
        )
        return PipelineOutcome(
            implementation_outcome=implementation_outcome, state=WorkflowState.CANCELLED
        )

    monkeypatch.setattr(Application, "run_pipeline", fake_run_pipeline)
    result = runner.invoke(app, ["complete", "Add a feature"])

    assert result.exit_code == 1
    assert "cancelled" in result.output.lower()


def test_cli_resume_reports_completion(monkeypatch, tmp_path: Path):
    """agentflow resume reports a recovered run's final summary when it completes."""

    async def fake_run_resume(
        self, run_id, project_path=None, role_override=None, final_approval_prompt=None
    ):
        planning = PlanningOutcome(run_id=run_id, state=WorkflowState.TASK_CLASSIFIED)
        implementation_outcome = ImplementationRunOutcome(
            planning_outcome=planning, state=WorkflowState.VERIFYING
        )
        return PipelineOutcome(
            implementation_outcome=implementation_outcome,
            state=WorkflowState.COMPLETED,
            final_summary_path=tmp_path / "final-summary.md",
        )

    monkeypatch.setattr(Application, "run_resume", fake_run_resume)
    result = runner.invoke(app, ["resume", "RUN-RESUME01"])

    assert result.exit_code == 0
    assert "RUN-RESUME01" in result.output
    assert "COMPLETED" in result.output


def test_cli_resume_errors_exit_nonzero(monkeypatch):
    """agentflow resume surfaces recovery errors with a nonzero exit code."""

    async def fake_run_resume(
        self, run_id, project_path=None, role_override=None, final_approval_prompt=None
    ):
        from agentflow.errors import ResumeError

        raise ResumeError("Run has no approved plan")

    monkeypatch.setattr(Application, "run_resume", fake_run_resume)
    result = runner.invoke(app, ["resume", "RUN-BAD"])

    assert result.exit_code == 1
    assert "no approved plan" in result.output


def test_cli_runs_passes_all_and_limit_to_application(monkeypatch):
    """agentflow runs forwards --all and --limit to the application renderer."""
    received: dict[str, object] = {}

    def fake_render_runs(self, project_path=None, limit=20, show_all=False):
        received.update(project_path=project_path, limit=limit, show_all=show_all)

    monkeypatch.setattr(Application, "render_runs", fake_render_runs)
    result = runner.invoke(app, ["runs", "--all", "--limit", "7"])

    assert result.exit_code == 0
    assert received == {"project_path": None, "limit": 7, "show_all": True}


def test_cli_cleanup_renders_report(monkeypatch):
    """agentflow cleanup runs the async cleanup service and renders its result."""
    received: dict[str, object] = {}

    async def fake_run_cleanup(self, project_path=None):
        received["project_path"] = project_path
        return CleanupReport(worktrees_removed=["RUN-OLD"], locks_cleared=["run:RUN-OLD"])

    def fake_render_cleanup_report(self, report):
        received["report"] = report

    monkeypatch.setattr(Application, "run_cleanup", fake_run_cleanup)
    monkeypatch.setattr(Application, "render_cleanup_report", fake_render_cleanup_report)
    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0
    assert received["project_path"] is None
    report = received["report"]
    assert isinstance(report, CleanupReport)
    assert report.worktrees_removed == ["RUN-OLD"]


def test_cli_retire_adds_mapping(monkeypatch):
    """agentflow retire OLD NEW adds the mapping and prints a confirmation table."""
    observed: dict[str, object] = {}

    def fake_run_retire(self, old_model, new_model):
        observed["old_model"] = old_model
        observed["new_model"] = new_model
        return RetirementReport(retired={"gpt-5.6 terra": "GPT-6 Sol"})

    monkeypatch.setattr(Application, "run_retire", fake_run_retire)
    result = runner.invoke(app, ["retire", "GPT-5.6 Terra", "GPT-6 Sol"])

    assert result.exit_code == 0
    assert observed == {"old_model": "GPT-5.6 Terra", "new_model": "GPT-6 Sol"}
    assert "retired" in result.output.lower()
    assert "GPT-6 Sol" in result.output


def test_cli_retire_rejects_excluded_replacement(monkeypatch):
    """A replacement excluded from the V1 pool surfaces a clear error, not a stack trace."""

    def fake_run_retire(self, old_model, new_model):
        raise ConfigurationError(f"Model '{new_model}' is explicitly excluded from AgentFlow V1.")

    monkeypatch.setattr(Application, "run_retire", fake_run_retire)
    result = runner.invoke(app, ["retire", "Foo", "GPT-5.6 Sol"])

    assert result.exit_code == 1
    assert "explicitly excluded" in result.output


def test_cli_retire_rejects_cycle(monkeypatch):
    """A retirement that would create a cycle is rejected with a clear error."""

    def fake_run_retire(self, old_model, new_model):
        raise ConfigurationError(f"Cannot retire '{old_model}' to '{new_model}': cycle.")

    monkeypatch.setattr(Application, "run_retire", fake_run_retire)
    result = runner.invoke(app, ["retire", "GPT-6 Sol", "GPT-6 Luna"])

    assert result.exit_code == 1
    assert "cycle" in result.output


def test_cli_retire_list(monkeypatch):
    """agentflow retire --list renders current retirements without mutating anything."""

    def fake_list_retirements(self):
        return RetirementReport(retired={"gpt-5.6 terra": "GPT-6 Sol"})

    monkeypatch.setattr(Application, "list_retirements", fake_list_retirements)
    result = runner.invoke(app, ["retire", "--list"])

    assert result.exit_code == 0
    assert "gpt-5.6 terra" in result.output
    assert "GPT-6 Sol" in result.output


def test_cli_retire_list_empty(monkeypatch):
    """agentflow retire --list reports when nothing is retired."""
    monkeypatch.setattr(Application, "list_retirements", lambda self: RetirementReport())
    result = runner.invoke(app, ["retire", "--list"])

    assert result.exit_code == 0
    assert "No models are currently retired" in result.output


def test_cli_retire_remove(monkeypatch):
    """agentflow retire --remove un-retires a model and confirms."""
    observed: dict[str, object] = {}

    def fake_remove(self, old_model):
        observed["old_model"] = old_model
        return True

    monkeypatch.setattr(Application, "remove_retirement", fake_remove)
    result = runner.invoke(app, ["retire", "--remove", "GPT-5.6 Terra"])

    assert result.exit_code == 0
    assert observed["old_model"] == "GPT-5.6 Terra"
    assert "no longer retired" in result.output


def test_cli_retire_remove_not_found(monkeypatch):
    """Removing a retirement that doesn't exist warns instead of erroring."""
    monkeypatch.setattr(Application, "remove_retirement", lambda self, old_model: False)
    result = runner.invoke(app, ["retire", "--remove", "Nonexistent Model"])

    assert result.exit_code == 0
    assert "was not retired" in result.output


def test_cli_retire_missing_arguments_exits_nonzero():
    """agentflow retire with neither a pair of models nor --list/--remove fails fast."""
    result = runner.invoke(app, ["retire"])

    assert result.exit_code == 1
    assert "Provide both" in result.output
