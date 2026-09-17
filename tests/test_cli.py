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
    RoutingOutcome,
)
from agentflow.cli import app
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


def test_cli_run_reports_task_classified(monkeypatch):
    """agentflow run prints artifact paths and exits 0 when planning reaches TASK_CLASSIFIED."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(
            run_id="RUN-TEST01",
            state=WorkflowState.TASK_CLASSIFIED,
            plan_path=Path("approved-plan.md"),
            task_profile_path=Path("task-profile.json"),
        )

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["run", "Add a health check endpoint"])

    assert result.exit_code == 0
    assert "RUN-TEST01" in result.output
    assert "TASK_CLASSIFIED" in result.output


def test_cli_run_blocked_exits_nonzero(monkeypatch):
    """agentflow run exits with code 1 and surfaces the blocker reason when planning is blocked."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(
            run_id="RUN-TEST02",
            state=WorkflowState.BLOCKED,
            blocker_reason="Planner CLI not authenticated.",
        )

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["run", "Do something impossible"])

    assert result.exit_code == 1
    assert "not authenticated" in result.output


def test_cli_run_cancelled_exits_nonzero(monkeypatch):
    """agentflow run exits with code 1 when the user cancels plan approval."""

    async def fake_run_planning(self, task_description, project_path=None):
        return PlanningOutcome(run_id="RUN-TEST03", state=WorkflowState.CANCELLED)

    monkeypatch.setattr(Application, "run_planning", fake_run_planning)
    result = runner.invoke(app, ["run", "Add a feature"])

    assert result.exit_code == 1
    assert "cancelled" in result.output.lower()


def test_cli_route_displays_decision_and_exits_zero(monkeypatch):
    """agentflow route prints the routing decision and exits 0 on success."""

    async def fake_run_routing(self, task_description, project_path=None, role_override=None):
        planning = PlanningOutcome(run_id="RUN-ROUTE01", state=WorkflowState.TASK_CLASSIFIED)
        decision = RoutingDecision(
            stage=Stage.IMPLEMENTATION,
            provider=Provider.OPENAI,
            model="GPT-5.6 Terra",
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
    assert "GPT-5.6 Terra" in result.output
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
