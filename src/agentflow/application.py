"""Application container and workflow orchestration core."""

import platform
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.table import Table

from agentflow.agents.base import Provider, validate_model_allowed
from agentflow.agents.registry import AgentAdapterRegistry, create_default_registry
from agentflow.concurrency.run_lock import RunLock
from agentflow.config.loader import load_global_config, save_global_config
from agentflow.config.models import GlobalConfig, ProjectConfig
from agentflow.config.retirement import apply_retirements
from agentflow.errors import (
    ConfigurationError,
    InitError,
    ProjectNotFoundError,
    ResumeError,
    WorktreeError,
)
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeHandle, WorktreeManager
from agentflow.observability.metrics import StatisticsReport, StatisticsService
from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.models import RunStatus
from agentflow.process.executor import ProcessExecutor
from agentflow.project.context import ProjectContext
from agentflow.project.discovery import ROUTING_CONFIG_RELATIVE_PATH, discover_project
from agentflow.project.init import build_starter_config
from agentflow.routing.decision import RoutingDecision
from agentflow.routing.engine import route
from agentflow.routing.rules import DEFAULT_MODELS_CONFIG, DEFAULT_ROUTING_RULES, RoleOverride
from agentflow.task.profile import Stage, TaskProfile
from agentflow.ui.approval import FinalApprovalDecision, ask_final_approval
from agentflow.ui.console import CHECKMARK, CROSSMARK, WARNINGMARK, ConsoleUI
from agentflow.workflow.completion import FinalSummary, write_final_summary_artifact
from agentflow.workflow.documentation import DocumentationOutcome, DocumentationWorkflow
from agentflow.workflow.implementation import ImplementationOutcome, ImplementationWorkflow
from agentflow.workflow.planning import PlanningOutcome, PlanningWorkflow
from agentflow.workflow.repair import RepairOutcome, RepairWorkflow
from agentflow.workflow.review import ReviewOutcome, ReviewReport, ReviewWorkflow
from agentflow.workflow.states import WorkflowState, transition_run_state
from agentflow.workflow.verification import VerificationResult, VerificationRunner

_TERMINAL_STATES: frozenset[WorkflowState] = frozenset(
    {
        WorkflowState.COMPLETED,
        WorkflowState.BLOCKED,
        WorkflowState.FAILED,
        WorkflowState.CANCELLED,
    }
)
_PLANNING_ONLY_STATES: frozenset[WorkflowState] = frozenset(
    {
        WorkflowState.PROJECT_READY,
        WorkflowState.PLANNING,
        WorkflowState.WAITING_FOR_USER,
        WorkflowState.PLAN_READY,
        WorkflowState.PLAN_APPROVED,
    }
)


@dataclass
class DoctorCheckItem:
    """Individual health check evaluated by agentflow doctor."""

    name: str
    passed: bool
    critical: bool
    details: str = ""
    is_warning: bool = False
    regressed: bool = False


@dataclass
class DoctorReport:
    """Consolidated doctor diagnosis report."""

    items: list[DoctorCheckItem] = field(default_factory=list)

    @property
    def has_critical_failures(self) -> bool:
        """Return True if any critical check failed."""
        return any(item.critical and not item.passed for item in self.items)

    @property
    def exit_code(self) -> int:
        """Exit code for the doctor command: 1 on critical failure, 0 otherwise."""
        return 1 if self.has_critical_failures else 0


@dataclass
class RoutingOutcome:
    """Result of classifying a task and computing its deterministic routing decision."""

    planning_outcome: PlanningOutcome
    decision: RoutingDecision | None = None
    decision_path: Path | None = None


@dataclass
class ImplementationRunOutcome:
    """Result of planning, routing, implementing, and verifying/repairing a task end to end."""

    planning_outcome: PlanningOutcome
    state: WorkflowState
    routing_decision: RoutingDecision | None = None
    implementation: ImplementationOutcome | None = None
    verification: VerificationResult | None = None
    verification_path: Path | None = None
    repair: RepairOutcome | None = None
    blocker_reason: str | None = None


@dataclass
class PipelineOutcome:
    """Result of the full pipeline: plan/route/implement/verify/repair/review/document/complete."""

    implementation_outcome: ImplementationRunOutcome
    state: WorkflowState
    review: ReviewOutcome | None = None
    documentation: DocumentationOutcome | None = None
    final_summary: FinalSummary | None = None
    final_summary_path: Path | None = None
    blocker_reason: str | None = None


@dataclass
class InitResult:
    """Result of `agentflow init`: where the starter profile was written, and what it found."""

    path: Path
    detected_groups: list[str] = field(default_factory=list)
    overwritten: bool = False
    providers: list[str] = field(default_factory=list)


@dataclass
class RetirementReport:
    """Current `agentflow retire` state: retired model name -> replacement model name."""

    retired: dict[str, str] = field(default_factory=dict)


@dataclass
class CleanupReport:
    """Result of `agentflow cleanup`: what was removed, and what was left untouched."""

    worktrees_removed: list[str] = field(default_factory=list)
    locks_cleared: list[str] = field(default_factory=list)
    logs_cleared: list[str] = field(default_factory=list)
    skipped_active: list[str] = field(default_factory=list)


class Application:
    """Core AgentFlow application container providing dependency injection and operations."""

    def __init__(
        self,
        config: GlobalConfig | None = None,
        db_manager: DatabaseManager | None = None,
        executor: ProcessExecutor | None = None,
        console_ui: ConsoleUI | None = None,
        agent_registry: AgentAdapterRegistry | None = None,
    ) -> None:
        self.config = config or load_global_config()
        self.db_manager = db_manager or DatabaseManager(self.config.storage.database)
        self.executor = executor or ProcessExecutor()
        self.ui = console_ui or ConsoleUI()
        self.agent_registry = agent_registry or create_default_registry(
            config=self.config, executor=self.executor
        )

    def initialize(self) -> None:
        """Initialize required directories and persistence database."""
        self.config.worktrees.root.mkdir(parents=True, exist_ok=True)
        self.config.logging.root.mkdir(parents=True, exist_ok=True)
        self.db_manager.initialize()

    def check_python(self) -> DoctorCheckItem:
        """Check Python version compatibility (>= 3.12 required)."""
        current_version = sys.version_info
        passed = current_version >= (3, 12)
        ver_str = platform.python_version()
        details = f"v{ver_str} (>= 3.12 required)"
        return DoctorCheckItem(
            name="Python runtime",
            passed=passed,
            critical=True,
            details=details,
        )

    def check_git(self) -> DoctorCheckItem:
        """Check if Git CLI is available and executable."""
        git_path = shutil.which("git")
        if not git_path:
            return DoctorCheckItem(
                name="Git",
                passed=False,
                critical=True,
                details="Not found in PATH",
            )

        try:
            res = self.executor.run_sync(["git", "--version"])
            if res.exit_code == 0:
                version_str = res.stdout.strip()
                return DoctorCheckItem(
                    name="Git",
                    passed=True,
                    critical=True,
                    details=f"{version_str} ({git_path})",
                )
            return DoctorCheckItem(
                name="Git",
                passed=False,
                critical=True,
                details=f"Command failed: {res.stderr.strip()}",
            )
        except Exception as e:
            return DoctorCheckItem(
                name="Git",
                passed=False,
                critical=True,
                details=f"Command execution error: {e}",
            )

    def check_cli(self, provider_name: str, command: str) -> DoctorCheckItem:
        """Check availability and functionality of a provider CLI using a lightweight command.

        Compares the fresh result against the last recorded `doctor` check for this provider
        (persisted in the `cli_availability` table) so a CLI that used to work and now doesn't
        can be flagged distinctly from an ordinary first-time-missing warning.
        """
        cli_path = shutil.which(command)
        if not cli_path:
            result = DoctorCheckItem(
                name=f"{provider_name} CLI",
                passed=False,
                critical=False,
                is_warning=True,
                details=f"Executable '{command}' not found in PATH",
            )
        else:
            # Validate using a lightweight command (no model requests sent)
            try:
                res = self.executor.run_sync([command, "--version"], timeout=5.0)
                if res.exit_code == 0:
                    first_line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
                    if not first_line and res.stderr.strip():
                        first_line = res.stderr.strip().splitlines()[0]
                    details = (
                        f"Found: {cli_path} ({first_line})" if first_line else f"Found: {cli_path}"
                    )
                    result = DoctorCheckItem(
                        name=f"{provider_name} CLI",
                        passed=True,
                        critical=False,
                        details=details,
                    )
                else:
                    result = DoctorCheckItem(
                        name=f"{provider_name} CLI",
                        passed=False,
                        critical=False,
                        is_warning=True,
                        details=(
                            f"Executable at {cli_path} failed '--version' check "
                            f"(exit code {res.exit_code})"
                        ),
                    )
            except Exception as e:
                result = DoctorCheckItem(
                    name=f"{provider_name} CLI",
                    passed=False,
                    critical=False,
                    is_warning=True,
                    details=f"Executable at {cli_path} could not be executed: {e}",
                )

        try:
            previous = self.db_manager.get_cli_availability(provider_name)
            if previous is not None and previous.available and not result.passed:
                result.regressed = True
                result.details = f"Previously available, now missing: {result.details}"
            self.db_manager.upsert_cli_availability(
                provider_name,
                command,
                result.passed,
                datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            # Caching is a diagnostic nicety, never a reason to fail the underlying CLI check.
            pass

        return result

    def check_storage_writable(self) -> DoctorCheckItem:
        """Check if the AgentFlow global storage and worktrees directories are writable."""
        directories = [
            ("Database dir", self.config.storage.database.parent),
            ("Worktrees dir", self.config.worktrees.root),
            ("Logs dir", self.config.logging.root),
        ]
        failed_dirs: list[str] = []

        for label, directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".test_write_") as tmp:
                    tmp.write(b"ok")
                    tmp.flush()
            except Exception as e:
                failed_dirs.append(f"{label} ({directory}): {e}")

        if failed_dirs:
            return DoctorCheckItem(
                name="Global data storage",
                passed=False,
                critical=True,
                details="; ".join(failed_dirs),
            )

        return DoctorCheckItem(
            name="Global data storage",
            passed=True,
            critical=True,
            details=f"Writable at {self.config.storage.database.parent}",
        )

    def check_sqlite(self) -> DoctorCheckItem:
        """Verify SQLite initialization and read/write capabilities."""
        try:
            self.db_manager.initialize()
            with self.db_manager.connection() as conn:
                cursor = conn.execute("SELECT count(*) FROM schema_migrations;")
                count = cursor.fetchone()[0]
            return DoctorCheckItem(
                name="SQLite database",
                passed=True,
                critical=True,
                details=f"{self.config.storage.database} (applied migrations: {count})",
            )
        except Exception as e:
            return DoctorCheckItem(
                name="SQLite database",
                passed=False,
                critical=True,
                details=f"Failed to initialize: {e}",
            )

    def check_project(
        self, project_path: Path | None = None
    ) -> tuple[DoctorCheckItem, DoctorCheckItem]:
        """Verify Git repository context and routing configuration for active directory."""
        try:
            ctx = discover_project(explicit_path=project_path)
            if ctx.is_git_repo:
                git_check = DoctorCheckItem(
                    name="Git repository",
                    passed=True,
                    critical=False,
                    details=f"Detected at {ctx.root_path} (ID: {ctx.project_id[:12]}...)",
                )
            else:
                git_check = DoctorCheckItem(
                    name="Git repository",
                    passed=False,
                    critical=False,
                    is_warning=True,
                    details=f"Directory '{ctx.root_path}' is not a Git repository",
                )

            if ctx.routing_config is not None:
                config_check = DoctorCheckItem(
                    name="Routing configuration",
                    passed=True,
                    critical=False,
                    details=f"Loaded from {ctx.routing_config_path} (Project: {ctx.project_name})",
                )
            elif ctx.routing_config_path and ctx.routing_config_path.exists():
                err_details = ctx.routing_config_error or "Invalid configuration"
                config_check = DoctorCheckItem(
                    name="Routing configuration",
                    passed=False,
                    critical=True,
                    is_warning=False,
                    details=f"Invalid configuration at {ctx.routing_config_path}: {err_details}",
                )
            else:
                config_check = DoctorCheckItem(
                    name="Routing configuration",
                    passed=False,
                    critical=False,
                    is_warning=True,
                    details="No .ai-orchestrator/routing.yaml found (optional for global commands)",
                )

            return git_check, config_check
        except ProjectNotFoundError as e:
            is_explicit = project_path is not None
            return (
                DoctorCheckItem(
                    name="Git repository",
                    passed=False,
                    critical=is_explicit,
                    is_warning=not is_explicit,
                    details=str(e),
                ),
                DoctorCheckItem(
                    name="Routing configuration",
                    passed=False,
                    critical=is_explicit,
                    is_warning=not is_explicit,
                    details=str(e),
                ),
            )
        except Exception as e:
            is_explicit = project_path is not None
            return (
                DoctorCheckItem(
                    name="Git repository",
                    passed=False,
                    critical=is_explicit,
                    is_warning=not is_explicit,
                    details=f"Discovery error: {e}",
                ),
                DoctorCheckItem(
                    name="Routing configuration",
                    passed=False,
                    critical=is_explicit,
                    is_warning=not is_explicit,
                    details=f"Discovery error: {e}",
                ),
            )

    def run_doctor(self, project_path: Path | None = None) -> DoctorReport:
        """Run all environment health checks and collect DoctorReport."""
        report = DoctorReport()

        try:
            # Ensure `cli_availability` (and every other table) exists before check_cli needs
            # it; check_sqlite below re-verifies this properly and reports any real failure.
            self.db_manager.initialize()
        except Exception:
            pass

        report.items.append(self.check_python())
        report.items.append(self.check_git())
        report.items.append(self.check_cli("Claude", self.config.cli.claude.command))
        report.items.append(self.check_cli("Codex", self.config.cli.codex.command))
        report.items.append(self.check_cli("Antigravity/Gemini", self.config.cli.gemini.command))
        report.items.append(self.check_storage_writable())
        report.items.append(self.check_sqlite())

        repo_check, config_check = self.check_project(project_path)
        report.items.append(repo_check)
        report.items.append(config_check)

        return report

    def render_doctor_report(self, report: DoctorReport) -> None:
        """Render a rich table for doctor results."""
        self.ui.print_header(
            "AgentFlow Doctor",
            "Environment, toolchain, and storage health check",
        )

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Status", width=8, justify="center")
        table.add_column("Component", style="bold", width=26)
        table.add_column("Details", style="dim")

        for item in report.items:
            if item.regressed:
                status_text = f"[bold red]{WARNINGMARK}[/bold red]"
            elif item.passed:
                status_text = f"[bold green]{CHECKMARK}[/bold green]"
            elif item.is_warning:
                status_text = f"[bold yellow]{WARNINGMARK}[/bold yellow]"
            else:
                status_text = f"[bold red]{CROSSMARK}[/bold red]"

            table.add_row(status_text, item.name, item.details)

        self.ui.console.print(table)

        if report.has_critical_failures:
            self.ui.print_error(
                "Critical environment requirements are missing. Please fix the marked errors above."
            )
        else:
            self.ui.print_success("All critical environment checks passed.")

    def _detect_available_providers(self) -> set[Provider]:
        """Detect which coding-agent CLIs are actually on PATH, using each provider's
        currently-configured command name (respects e.g. a `cli.gemini.command: agy` override)."""
        found: set[Provider] = set()
        if shutil.which(self.config.cli.claude.command):
            found.add(Provider.ANTHROPIC)
        if shutil.which(self.config.cli.codex.command):
            found.add(Provider.OPENAI)
        if shutil.which(self.config.cli.gemini.command):
            found.add(Provider.GOOGLE)
        return found

    def run_init(
        self,
        project_path: Path | None = None,
        force: bool = False,
        providers: set[Provider] | None = None,
    ) -> InitResult:
        """Generate a starter `.ai-orchestrator/routing.yaml` for a project.

        Detects common per-directory test setups (Node/Python/Java) as a starting point only --
        always review the generated file before running real tasks against it. Refuses to
        overwrite an existing profile unless `force` is set. `providers` selects which coding-
        agent CLIs the generated `models:` section may route to; if omitted, detects whichever
        of claude/codex/gemini are actually on PATH so the profile is usable even when not all
        three are installed.
        """
        ctx = discover_project(explicit_path=project_path)
        routing_path = ctx.root_path / ROUTING_CONFIG_RELATIVE_PATH
        already_existed = routing_path.exists()
        if already_existed and not force:
            raise InitError(f"{routing_path} already exists. Pass --force to overwrite it.")

        resolved_providers = (
            providers if providers is not None else self._detect_available_providers()
        )
        if not resolved_providers:
            raise InitError(
                "No coding-agent CLI detected (claude/codex/gemini) and none specified via "
                "--providers; agentflow needs at least one to do anything. Install one, or "
                "pass --providers explicitly."
            )

        config = build_starter_config(ctx.root_path, resolved_providers)
        routing_path.parent.mkdir(parents=True, exist_ok=True)
        routing_path.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return InitResult(
            path=routing_path,
            detected_groups=sorted(config.verification or {}),
            overwritten=already_existed,
            providers=sorted(p.value for p in resolved_providers),
        )

    def run_retire(self, old_model: str, new_model: str) -> RetirementReport:
        """Persist a global model retirement: every project's routing transparently substitutes
        `new_model` for `old_model` from now on (routing/rules.py `ModelsConfig.resolve`),
        without touching any project's `.ai-orchestrator/routing.yaml`.
        """
        try:
            validate_model_allowed(new_model)
        except ValueError as e:
            raise ConfigurationError(str(e)) from e
        retired = dict(self.config.models.retired)
        old_key = old_model.strip().lower()

        # Reject a mapping that would create a cycle: if new_model already chain-resolves
        # back to old_model, adding old_model -> new_model would close the loop.
        probe = dict(retired)
        probe[old_key] = new_model
        try:
            apply_retirements(old_model, probe)
        except ConfigurationError as e:
            raise ConfigurationError(
                f"Cannot retire '{old_model}' to '{new_model}': this would create a "
                f"retirement cycle."
            ) from e

        retired[old_key] = new_model
        self.config.models.retired = retired
        save_global_config(self.config)
        return RetirementReport(retired=dict(self.config.models.retired))

    def list_retirements(self) -> RetirementReport:
        """Return the current global model retirements."""
        return RetirementReport(retired=dict(self.config.models.retired))

    def remove_retirement(self, old_model: str) -> bool:
        """Un-retire `old_model`. Returns False (no-op) if it wasn't retired."""
        old_key = old_model.strip().lower()
        retired = dict(self.config.models.retired)
        if old_key not in retired:
            return False
        del retired[old_key]
        self.config.models.retired = retired
        save_global_config(self.config)
        return True

    def render_retirements(self, report: RetirementReport) -> None:
        """Render the current global model retirements as a table."""
        self.ui.print_header("Model Retirements", "Applied globally across every project.")
        if not report.retired:
            self.ui.print_info("No models are currently retired.")
            return
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Retired model")
        table.add_column("Replacement")
        for old, new in sorted(report.retired.items()):
            table.add_row(old, new)
        self.ui.console.print(table)

    def get_status_message(self, project_path: Path | None = None) -> str:
        """Get status message for active runs."""
        ctx = discover_project(explicit_path=project_path)
        self.db_manager.initialize()
        active_runs = self.db_manager.get_active_runs(project_id=ctx.project_id)

        if not active_runs:
            return (
                f"Project: {ctx.project_name} (ID: {ctx.project_id[:12]})\n"
                "Status: No active runs found."
            )

        runs_summary = "\n".join(
            f"  - Run {r.id}: Task='{r.task}', Status={r.status}, Stage={r.state}, "
            f"Started={r.created_at.isoformat()}, Updated={r.updated_at.isoformat()}"
            for r in active_runs
        )
        return (
            f"Project: {ctx.project_name} (ID: {ctx.project_id[:12]})\n"
            f"Active runs ({len(active_runs)}):\n{runs_summary}"
        )

    async def run_planning(
        self, task_description: str, project_path: Path | None = None
    ) -> PlanningOutcome:
        """Run interactive planning for a task through TaskProfile generation.

        The planner's provider/model are read from the project's routing.yaml
        (`models.planner.*`), not hardcoded to any one CLI.
        """
        self.initialize()
        ctx = discover_project(explicit_path=project_path)
        run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
        run_lock = RunLock(run_id, self._locks_root())
        run_lock.acquire()
        routing_config = ctx.routing_config
        models_cfg = (routing_config.models if routing_config else None) or DEFAULT_MODELS_CONFIG
        routing_rules_cfg = (
            routing_config.routing if routing_config else None
        ) or DEFAULT_ROUTING_RULES
        workflow = PlanningWorkflow(
            db_manager=self.db_manager,
            agent_registry=self.agent_registry,
            models_config=models_cfg,
            routing_rules=routing_rules_cfg,
            console_ui=self.ui,
            retired_models=self.config.models.retired,
        )
        try:
            return await workflow.run(task_description, ctx, run_id=run_id)
        finally:
            run_lock.release()

    def _route_and_persist(
        self,
        planning_outcome: PlanningOutcome,
        project_context: ProjectContext,
        role_override: RoleOverride | None,
    ) -> RoutingDecision:
        """Compute a deterministic RoutingDecision for a classified task and persist it."""
        assert planning_outcome.task_profile is not None
        routing_config = project_context.routing_config

        decision = route(
            planning_outcome.task_profile,
            models=routing_config.models if routing_config else None,
            routing_rules=routing_config.routing if routing_config else None,
            complexity_config=routing_config.complexity if routing_config else None,
            user_override=(
                role_override.for_stage(Stage.IMPLEMENTATION) if role_override else None
            ),
            retired_models=self.config.models.retired,
        )

        if planning_outcome.plan_path is not None:
            decision_path = planning_outcome.plan_path.parent / "routing-decision.json"
            decision_path.write_text(decision.model_dump_json(indent=2), encoding="utf-8")

        self.db_manager.record_routing_decision(
            str(uuid.uuid4()),
            planning_outcome.run_id,
            decision.stage.value,
            planning_outcome.task_profile.model_dump_json(),
            decision.complexity_score,
            decision.complexity.value,
            decision.matched_rule,
            decision.provider.value,
            decision.model,
            decision.reason,
        )
        return decision

    async def run_routing(
        self,
        task_description: str,
        project_path: Path | None = None,
        role_override: RoleOverride | None = None,
    ) -> RoutingOutcome:
        """Classify a task via planning, then compute and persist its routing decision.

        Never creates a worktree or modifies application files (routing is decision-only).
        """
        planning_outcome = await self.run_planning(task_description, project_path=project_path)
        if (
            planning_outcome.state != WorkflowState.TASK_CLASSIFIED
            or planning_outcome.task_profile is None
        ):
            return RoutingOutcome(planning_outcome=planning_outcome)

        ctx = discover_project(explicit_path=project_path)
        decision = self._route_and_persist(planning_outcome, ctx, role_override)

        decision_path = None
        if planning_outcome.plan_path is not None:
            decision_path = planning_outcome.plan_path.parent / "routing-decision.json"

        return RoutingOutcome(
            planning_outcome=planning_outcome, decision=decision, decision_path=decision_path
        )

    async def _plan_route_implement_and_verify(
        self,
        task_description: str,
        project_path: Path | None = None,
        role_override: RoleOverride | None = None,
    ) -> ImplementationRunOutcome:
        """Shared core for run_implementation and run_pipeline.

        Plans, routes, implements inside an isolated worktree, then verifies and repairs. Does
        NOT mark the run COMPLETED on success -- callers decide the final RunStatus, since
        run_pipeline continues into review/documentation/approval afterward.
        """
        planning_outcome = await self.run_planning(task_description, project_path=project_path)
        if (
            planning_outcome.state != WorkflowState.TASK_CLASSIFIED
            or planning_outcome.task_profile is None
            or planning_outcome.plan_markdown is None
        ):
            return ImplementationRunOutcome(
                planning_outcome=planning_outcome,
                state=planning_outcome.state,
                blocker_reason=planning_outcome.blocker_reason,
            )

        ctx = discover_project(explicit_path=project_path)
        run_lock = RunLock(planning_outcome.run_id, self._locks_root())
        run_lock.acquire()
        try:
            return await self._implement_and_verify_classified(
                planning_outcome, task_description, ctx, role_override
            )
        finally:
            run_lock.release()

    async def _implement_and_verify_classified(
        self,
        planning_outcome: PlanningOutcome,
        task_description: str,
        ctx: ProjectContext,
        role_override: RoleOverride | None,
    ) -> ImplementationRunOutcome:
        """Implement and verify a classified task while its run-level lock is held."""
        assert planning_outcome.plan_markdown is not None
        assert planning_outcome.task_profile is not None
        decision = self._route_and_persist(planning_outcome, ctx, role_override)
        routing_config = ctx.routing_config

        worktree_manager = WorktreeManager(self.config.worktrees.root, executor=self.executor)
        implementation_workflow = ImplementationWorkflow(
            self.db_manager, self.agent_registry, worktree_manager, self.ui
        )
        implementation = await implementation_workflow.run(
            planning_outcome.run_id,
            task_description,
            ctx,
            planning_outcome.plan_markdown,
            planning_outcome.task_profile,
            decision,
        )
        if implementation.state != WorkflowState.VERIFYING or implementation.worktree is None:
            self.db_manager.update_run_status(planning_outcome.run_id, RunStatus.BLOCKED.value)
            return ImplementationRunOutcome(
                planning_outcome=planning_outcome,
                state=implementation.state,
                routing_decision=decision,
                implementation=implementation,
                blocker_reason=implementation.blocker_reason,
            )

        verification, verification_path, repair = await self._verify_and_repair(
            planning_outcome.run_id,
            ctx,
            implementation.worktree.path,
            planning_outcome.plan_markdown,
            planning_outcome.task_profile,
            routing_config,
        )

        final_state = WorkflowState.VERIFYING if verification.success else WorkflowState.BLOCKED
        if not verification.success:
            self.db_manager.update_run_status(planning_outcome.run_id, RunStatus.BLOCKED.value)

        return ImplementationRunOutcome(
            planning_outcome=planning_outcome,
            state=final_state,
            routing_decision=decision,
            implementation=implementation,
            verification=verification,
            verification_path=verification_path,
            repair=repair,
            blocker_reason=repair.blocker_reason if repair else None,
        )

    async def _verify_and_repair(
        self,
        run_id: str,
        ctx: ProjectContext,
        worktree_path: Path,
        plan_markdown: str,
        task_profile: TaskProfile,
        routing_config: ProjectConfig | None,
    ) -> tuple[VerificationResult, Path, RepairOutcome | None]:
        """Run verification, then a bounded repair/re-verify loop if it fails.

        Shared by fresh implementation runs and `run_resume` (Milestone 7): identical
        deterministic verify/repair behavior regardless of how the worktree's contents came
        to exist.
        """
        cfg = routing_config
        verification_runner = VerificationRunner(
            self.db_manager, executor=self.executor, console_ui=self.ui
        )
        verification_config = cfg.verification if cfg else None
        log_dir = ctx.root_path / ".ai-orchestrator" / "runs" / run_id / "verification"
        verification = await verification_runner.run(
            run_id, worktree_path, verification_config, log_dir=log_dir
        )

        repair: RepairOutcome | None = None
        if not verification.success:
            models_cfg = (cfg.models if cfg else None) or DEFAULT_MODELS_CONFIG
            limits = cfg.limits if cfg else None
            repair_workflow = RepairWorkflow(
                self.db_manager,
                self.agent_registry,
                verification_runner,
                models_cfg,
                limits,
                self.ui,
                retired_models=self.config.models.retired,
            )
            repair = await repair_workflow.run(
                run_id,
                worktree_path,
                ctx,
                plan_markdown,
                task_profile,
                verification_config,
                verification,
            )
            verification = repair.verification_result

        verification_path = verification_runner.write_artifact(ctx.root_path, run_id, verification)
        return verification, verification_path, repair

    async def run_implementation(
        self,
        task_description: str,
        project_path: Path | None = None,
        role_override: RoleOverride | None = None,
    ) -> ImplementationRunOutcome:
        """Plan, route, implement inside an isolated worktree, then verify and repair.

        The primary working tree is never modified; all code changes happen in a disposable
        Git worktree under the configured worktrees root.
        """
        outcome = await self._plan_route_implement_and_verify(
            task_description, project_path=project_path, role_override=role_override
        )
        if outcome.verification is not None and outcome.verification.success:
            self.db_manager.update_run_status(
                outcome.planning_outcome.run_id, RunStatus.COMPLETED.value
            )
        return outcome

    def render_routing_decision(self, decision: RoutingDecision) -> None:
        """Render an explainable summary of a routing decision to the console."""
        self.ui.print_header("Routing Decision")
        self.ui.console.print(f"[bold]Stage:[/bold]             {decision.stage.value}")
        self.ui.console.print(f"[bold]Complexity score:[/bold]  {decision.complexity_score}")
        self.ui.console.print(f"[bold]Complexity:[/bold]        {decision.complexity.value}")

        if decision.risk_flags:
            self.ui.console.print("\n[bold]Risk:[/bold]")
            for flag in decision.risk_flags:
                self.ui.console.print(f"  [bold green]{CHECKMARK}[/bold green] {flag}")

        self.ui.console.print(f"\n[bold]Matched rule:[/bold]\n{decision.matched_rule}")
        self.ui.console.print(
            f"\n[bold]Selected:[/bold]\n"
            f"Provider: {decision.provider.value}\n"
            f"Model:    {decision.model}"
        )
        self.ui.console.print(f"\n[bold]Reason:[/bold]\n{decision.reason}")

    async def run_pipeline(
        self,
        task_description: str,
        project_path: Path | None = None,
        role_override: RoleOverride | None = None,
        final_approval_prompt: Callable[[], FinalApprovalDecision] | None = None,
    ) -> PipelineOutcome:
        """Plan, route, implement, verify/repair, review, document, then await human approval.

        V1 never auto-pushes, auto-merges, or deploys -- only a human decides completion.
        """
        impl_outcome = await self._plan_route_implement_and_verify(
            task_description, project_path=project_path, role_override=role_override
        )
        run_id = impl_outcome.planning_outcome.run_id

        if (
            impl_outcome.verification is None
            or not impl_outcome.verification.success
            or impl_outcome.implementation is None
            or impl_outcome.implementation.worktree is None
            or impl_outcome.routing_decision is None
        ):
            return PipelineOutcome(
                implementation_outcome=impl_outcome,
                state=impl_outcome.state,
                blocker_reason=impl_outcome.blocker_reason,
            )

        ctx = discover_project(explicit_path=project_path)
        worktree_path = impl_outcome.implementation.worktree.path
        plan_markdown = impl_outcome.planning_outcome.plan_markdown
        task_profile = impl_outcome.planning_outcome.task_profile
        assert plan_markdown is not None
        assert task_profile is not None

        run_lock = RunLock(run_id, self._locks_root())
        run_lock.acquire()
        try:
            return await self._review_document_and_approve(
                run_id,
                ctx,
                worktree_path,
                impl_outcome.implementation.worktree.branch_name,
                task_description,
                plan_markdown,
                task_profile,
                impl_outcome.routing_decision,
                impl_outcome.verification,
                impl_outcome,
                final_approval_prompt,
                role_override,
            )
        finally:
            run_lock.release()

    async def _review_document_and_approve(
        self,
        run_id: str,
        ctx: ProjectContext,
        worktree_path: Path,
        branch_name: str,
        task_description: str,
        plan_markdown: str,
        task_profile: TaskProfile,
        implementation_decision: RoutingDecision,
        verification: VerificationResult,
        impl_outcome: ImplementationRunOutcome,
        final_approval_prompt: Callable[[], FinalApprovalDecision] | None,
        role_override: RoleOverride | None = None,
    ) -> PipelineOutcome:
        """Review, document, and await human approval for an already-verified worktree.

        Shared tail for `run_pipeline` (fresh runs) and `run_resume` (Milestone 7): once a
        worktree's contents are verified, the remaining gates are identical regardless of how
        that worktree came to exist.
        """
        routing_config = ctx.routing_config
        verification_config = routing_config.verification if routing_config else None
        models_cfg = (routing_config.models if routing_config else None) or DEFAULT_MODELS_CONFIG
        routing_rules_cfg = (
            routing_config.routing if routing_config else None
        ) or DEFAULT_ROUTING_RULES
        limits = routing_config.limits if routing_config else None

        verification_runner = VerificationRunner(
            self.db_manager, executor=self.executor, console_ui=self.ui
        )
        worktree_manager = WorktreeManager(self.config.worktrees.root, executor=self.executor)

        transition_run_state(
            self.db_manager, run_id, WorkflowState.REVIEWING, "Verification passed; starting review"
        )
        review_workflow = ReviewWorkflow(
            self.db_manager,
            self.agent_registry,
            verification_runner,
            worktree_manager,
            models_cfg,
            routing_rules_cfg,
            limits,
            self.ui,
            role_override=role_override,
            retired_models=self.config.models.retired,
        )
        review_outcome = await review_workflow.run(
            run_id,
            worktree_path,
            ctx,
            plan_markdown,
            task_profile,
            verification_config,
            implementation_decision,
            verification,
        )
        if review_outcome.state != WorkflowState.REVIEW_APPROVED:
            self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
            return PipelineOutcome(
                implementation_outcome=impl_outcome,
                state=review_outcome.state,
                review=review_outcome,
                blocker_reason=review_outcome.blocker_reason,
            )

        final_verification = review_outcome.verification_result or verification
        pre_doc_diff = await worktree_manager.capture_changes(worktree_path)

        doc_config = routing_config.documentation if routing_config else None
        will_run_documentation = bool(
            doc_config and doc_config.enabled and doc_config.candidate_files
        )
        if will_run_documentation:
            transition_run_state(
                self.db_manager,
                run_id,
                WorkflowState.DOCUMENTING,
                "Review approved; starting documentation sync",
            )

        documentation_workflow = DocumentationWorkflow(
            self.db_manager,
            self.agent_registry,
            worktree_manager,
            verification_runner,
            models_cfg,
            routing_rules_cfg,
            self.ui,
            role_override=role_override,
            retired_models=self.config.models.retired,
        )
        documentation_outcome = await documentation_workflow.run(
            run_id,
            worktree_path,
            ctx,
            plan_markdown,
            task_profile,
            pre_doc_diff.diff_text,
            final_verification,
            review_outcome.findings,
            routing_config.documentation if routing_config else None,
            verification_config,
        )
        if documentation_outcome.blocked:
            reason = documentation_outcome.blocker_reason or "Documentation workflow blocked."
            transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
            self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
            return PipelineOutcome(
                implementation_outcome=impl_outcome,
                state=WorkflowState.BLOCKED,
                review=review_outcome,
                documentation=documentation_outcome,
                blocker_reason=reason,
            )
        if documentation_outcome.verification_result is not None:
            final_verification = documentation_outcome.verification_result

        transition_run_state(
            self.db_manager, run_id, WorkflowState.READY_FOR_APPROVAL, "All gates passed"
        )

        final_diff = await worktree_manager.capture_changes(worktree_path)
        summary = FinalSummary(
            task=task_description,
            plan_markdown=plan_markdown,
            routing_decisions=self.db_manager.list_routing_decisions(run_id),
            changed_files=final_diff.changed_files,
            verification_result=final_verification,
            review_findings=review_outcome.findings,
            documentation_updated_files=documentation_outcome.updated_files,
            worktree_path=worktree_path,
            branch_name=branch_name,
        )
        summary_path = write_final_summary_artifact(ctx.root_path, run_id, summary)
        self.render_final_summary(summary)

        prompt = final_approval_prompt or ask_final_approval
        approval = prompt()
        self.db_manager.record_decision(str(uuid.uuid4()), run_id, "final_approval", approval.value)

        if approval == FinalApprovalDecision.CANCEL:
            transition_run_state(
                self.db_manager, run_id, WorkflowState.CANCELLED, "User cancelled at final approval"
            )
            self.db_manager.update_run_status(run_id, RunStatus.CANCELLED.value)
            return PipelineOutcome(
                implementation_outcome=impl_outcome,
                state=WorkflowState.CANCELLED,
                review=review_outcome,
                documentation=documentation_outcome,
                final_summary=summary,
                final_summary_path=summary_path,
            )

        reason = (
            "User approved completion"
            if approval == FinalApprovalDecision.APPROVE
            else "User approved completion; keeping worktree for manual inspection"
        )
        transition_run_state(self.db_manager, run_id, WorkflowState.COMPLETED, reason)
        self.db_manager.update_run_status(run_id, RunStatus.COMPLETED.value)

        return PipelineOutcome(
            implementation_outcome=impl_outcome,
            state=WorkflowState.COMPLETED,
            review=review_outcome,
            documentation=documentation_outcome,
            final_summary=summary,
            final_summary_path=summary_path,
        )

    def render_final_summary(self, summary: FinalSummary) -> None:
        """Render the final run summary to the console."""
        self.ui.print_header("Final Summary")
        self.ui.console.print(summary.render_markdown())

    def _locks_root(self) -> Path:
        """Global directory holding run-level lock files, alongside the shared database."""
        return self.config.storage.database.parent / "locks"

    @staticmethod
    def _read_artifact_text(path: Path) -> str | None:
        """Read a run artifact file's text, or None if it does not exist."""
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _read_task_artifact(self, run_dir: Path, fallback: str) -> str:
        """Recover the original task description, preferring the on-disk task.md artifact.

        Falls back to `fallback` (the durable `runs.task` DB column) if the artifact is
        missing -- task.md is a human-readable export, not the source of truth, so its
        absence must never block a resume when the database still has the description.
        """
        text = self._read_artifact_text(run_dir / "task.md")
        if text is None:
            return fallback
        # Undo the "# Task\n\n<description>\n" wrapper written by PlanningWorkflow._write_artifact.
        return text.removeprefix("# Task\n\n").rstrip("\n")

    def _read_task_profile_artifact(self, run_dir: Path) -> TaskProfile | None:
        """Load the persisted TaskProfile artifact, if present."""
        text = self._read_artifact_text(run_dir / "task-profile.json")
        if text is None:
            return None
        return TaskProfile.model_validate_json(text)

    def _read_verification_artifact(self, run_dir: Path) -> VerificationResult | None:
        """Load the persisted verification.json artifact, if present."""
        text = self._read_artifact_text(run_dir / "verification.json")
        if text is None:
            return None
        return VerificationResult.model_validate_json(text)

    def _read_review_findings_artifact(self, run_dir: Path) -> ReviewReport | None:
        """Load the persisted review-findings.json artifact, if present."""
        text = self._read_artifact_text(run_dir / "review-findings.json")
        if text is None:
            return None
        return ReviewReport.model_validate_json(text)

    def _report_stale_artifacts(self, project_root: Path, run_id: str) -> None:
        """Surface leftover verification/review artifacts from a run being resumed after BLOCKED.

        Informational only -- resume always re-runs verification and review regardless, so this
        never changes control flow. It exists so these persisted artifacts are actually read by
        something instead of sitting on disk unconsumed.
        """
        run_dir = project_root / ".ai-orchestrator" / "runs" / run_id
        verification = self._read_verification_artifact(run_dir)
        if verification is not None:
            self.ui.print_info(
                f"Note: a prior verification.json exists (status: {verification.status.value}) "
                "-- verification will be re-run."
            )
        review = self._read_review_findings_artifact(run_dir)
        if review is not None:
            self.ui.print_info(
                f"Note: a prior review-findings.json exists (status: {review.status}) "
                "-- review will be re-run."
            )

    def _load_persisted_routing_decision(
        self,
        run_dir: Path,
        planning_outcome: PlanningOutcome,
        ctx: ProjectContext,
        role_override: RoleOverride | None = None,
    ) -> RoutingDecision:
        """Reload the original implementation RoutingDecision artifact, recomputing as a
        fallback (routing is deterministic and reproducible from the persisted TaskProfile)."""
        text = self._read_artifact_text(run_dir / "routing-decision.json")
        if text is None:
            return self._route_and_persist(planning_outcome, ctx, role_override)
        return RoutingDecision.model_validate_json(text)

    def _recover_pre_blocked_state(self, run_id: str) -> WorkflowState:
        """Resolve the stage a BLOCKED run was in immediately before it blocked, and force the
        persisted state back to it so `_resume_locked` can re-enter the pipeline there.

        Reuses the same unvalidated `update_run_state` bypass `_resume_reimplement` already uses
        for crash recovery: the run's last-persisted state isn't a legal `transition_run_state`
        predecessor of the recovery action about to be taken, so the strict validator doesn't
        apply here -- the same reasoning applies to a run blocked by an agent-CLI failure.
        """
        transitions = self.db_manager.list_state_transitions(run_id)
        recovered = next(
            (
                t.from_state
                for t in reversed(transitions)
                if t.to_state == WorkflowState.BLOCKED.value
            ),
            None,
        )
        if not recovered or recovered == WorkflowState.NEW.value:
            raise ResumeError(
                f"Run '{run_id}' is BLOCKED with no recoverable prior stage; "
                "start a new run instead."
            )
        recovered_state = WorkflowState(recovered)
        self.db_manager.update_run_state(
            run_id,
            recovered_state.value,
            f"Resuming BLOCKED run: retrying from {recovered_state.value}",
        )
        return recovered_state

    def _recover_stale_worktree_lock(self, worktree_path: Path) -> None:
        """Clear a worktree writer lock left behind by a dead process; refuse to resume if the
        lock's owner is still alive (Phase 10 Step 7: never silently delete an active lock)."""
        lock = WorktreeLock(worktree_path)
        if not lock.is_locked():
            return
        info = lock.read_info()
        if lock.is_stale():
            owner = f"{info.owner} (pid {info.pid})" if info else "unknown owner"
            self.ui.print_warning(f"Clearing stale writer lock on {worktree_path} (was {owner}).")
            lock.acquire(owner="agentflow-recovery", break_stale=True)
            lock.release()
            return
        pid = info.pid if info else "unknown"
        raise ResumeError(
            f"Worktree {worktree_path} is actively locked by a live process (pid={pid}); "
            "cannot resume until it finishes or is stopped."
        )

    async def run_resume(
        self,
        run_id: str,
        project_path: Path | None = None,
        role_override: RoleOverride | None = None,
        final_approval_prompt: Callable[[], FinalApprovalDecision] | None = None,
    ) -> PipelineOutcome:
        """Resume a run interrupted by a crash, or a BLOCKED run, and continue it safely.

        Never blindly repeats completed work: if the worktree already holds implementation
        changes, they are re-verified and carried forward instead of re-invoking the
        implementation agent; only a worktree with no durable changes is recreated. A BLOCKED
        run (any coding-agent CLI failure, including a usage-limit error) is recovered by
        resolving which stage it was in immediately before it blocked and retrying from there;
        pass `role_override` to route that retry to a different provider/model.
        """
        self.initialize()
        run = self.db_manager.get_run(run_id)
        if run is None:
            raise ResumeError(f"Run '{run_id}' not found.")

        current_state = WorkflowState(run.state)
        if current_state == WorkflowState.NEW:
            raise ResumeError(
                f"Run '{run_id}' never advanced past creation; nothing to resume. "
                "Start a new run instead."
            )
        resumed_from_blocked = current_state == WorkflowState.BLOCKED
        if resumed_from_blocked:
            current_state = self._recover_pre_blocked_state(run_id)
        elif current_state in _TERMINAL_STATES:
            raise ResumeError(
                f"Run '{run_id}' already ended in state {current_state.value}; nothing to resume."
            )

        project_record = self.db_manager.get_project(run.project_id)
        if project_record is None:
            raise ResumeError(f"Project for run '{run_id}' not found in the local database.")
        ctx = discover_project(explicit_path=project_path or Path(project_record.repository_path))
        recorded_repository = Path(project_record.repository_path).resolve()
        if ctx.project_id != run.project_id and ctx.root_path != recorded_repository:
            raise ResumeError(
                f"Run '{run_id}' belongs to project at '{project_record.repository_path}', "
                f"not '{ctx.root_path}'. Pass --project to target the correct repository."
            )

        if resumed_from_blocked:
            self._report_stale_artifacts(ctx.root_path, run_id)

        run_lock = RunLock(run_id, self._locks_root())
        run_lock.acquire()
        try:
            return await self._resume_locked(
                run_id,
                run.task,
                current_state,
                ctx,
                role_override,
                final_approval_prompt,
            )
        finally:
            run_lock.release()

    async def _resume_locked(
        self,
        run_id: str,
        persisted_task_description: str,
        current_state: WorkflowState,
        ctx: ProjectContext,
        role_override: RoleOverride | None,
        final_approval_prompt: Callable[[], FinalApprovalDecision] | None,
    ) -> PipelineOutcome:
        """Reconstruct persisted context and re-enter the pipeline at the correct stage."""
        run_dir = ctx.root_path / ".ai-orchestrator" / "runs" / run_id

        if current_state in _PLANNING_ONLY_STATES:
            routing_config = ctx.routing_config
            models_cfg = (
                routing_config.models if routing_config else None
            ) or DEFAULT_MODELS_CONFIG
            routing_rules_cfg = (
                routing_config.routing if routing_config else None
            ) or DEFAULT_ROUTING_RULES
            planning_workflow = PlanningWorkflow(
                self.db_manager,
                self.agent_registry,
                models_cfg,
                routing_rules_cfg,
                self.ui,
                retired_models=self.config.models.retired,
            )
            planning_outcome = await planning_workflow.resume(
                run_id, persisted_task_description, ctx, current_state
            )
            if planning_outcome.state != WorkflowState.TASK_CLASSIFIED:
                return PipelineOutcome(
                    implementation_outcome=ImplementationRunOutcome(
                        planning_outcome=planning_outcome,
                        state=planning_outcome.state,
                        blocker_reason=planning_outcome.blocker_reason,
                    ),
                    state=planning_outcome.state,
                    blocker_reason=planning_outcome.blocker_reason,
                )
            current_state = WorkflowState.TASK_CLASSIFIED

        task_description = self._read_task_artifact(run_dir, persisted_task_description)

        plan_markdown = self._read_artifact_text(run_dir / "approved-plan.md")
        task_profile = self._read_task_profile_artifact(run_dir)
        if plan_markdown is None or task_profile is None:
            raise ResumeError(
                f"Run '{run_id}' is missing its approved plan or TaskProfile artifact under "
                f"{run_dir}; cannot safely resume. Start a new run instead."
            )

        planning_outcome = PlanningOutcome(
            run_id=run_id,
            state=WorkflowState.TASK_CLASSIFIED,
            plan_markdown=plan_markdown,
            task_profile=task_profile,
            plan_path=run_dir / "approved-plan.md",
            task_profile_path=run_dir / "task-profile.json",
        )

        routing_config = ctx.routing_config
        worktree_manager = WorktreeManager(self.config.worktrees.root, executor=self.executor)
        worktree_path = worktree_manager.worktree_path_for(ctx.project_name, run_id)
        branch_name = worktree_manager.branch_name_for(run_id)

        existing_diff = None
        if worktree_path.exists():
            self._recover_stale_worktree_lock(worktree_path)
            existing_diff = await worktree_manager.capture_changes(worktree_path)

        if existing_diff is None or existing_diff.is_empty:
            decision, implementation = await self._resume_reimplement(
                run_id,
                ctx,
                worktree_manager,
                worktree_path,
                branch_name,
                task_description,
                planning_outcome,
                plan_markdown,
                task_profile,
                role_override,
                existing_diff is not None,
            )
            if implementation.state != WorkflowState.VERIFYING or implementation.worktree is None:
                self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
                return PipelineOutcome(
                    implementation_outcome=ImplementationRunOutcome(
                        planning_outcome=planning_outcome,
                        state=implementation.state,
                        routing_decision=decision,
                        implementation=implementation,
                        blocker_reason=implementation.blocker_reason,
                    ),
                    state=implementation.state,
                    blocker_reason=implementation.blocker_reason,
                )
            worktree_handle = implementation.worktree
        else:
            self.ui.print_info(
                f"Run '{run_id}': found {len(existing_diff.changed_files)} previously-changed "
                "file(s) already in the worktree; resuming from verification instead of "
                "re-running implementation."
            )
            decision = self._load_persisted_routing_decision(
                run_dir, planning_outcome, ctx, role_override
            )
            worktree_handle = WorktreeHandle(
                path=worktree_path, branch_name=branch_name, repository_path=ctx.root_path
            )
            self.db_manager.update_run_state(
                run_id,
                WorkflowState.VERIFYING.value,
                "Resumed: existing worktree changes found; re-verifying",
            )
            implementation = ImplementationOutcome(
                run_id=run_id,
                state=WorkflowState.VERIFYING,
                worktree=worktree_handle,
                diff=existing_diff,
                model=decision.model,
            )

        verification, verification_path, repair = await self._verify_and_repair(
            run_id, ctx, worktree_handle.path, plan_markdown, task_profile, routing_config
        )
        impl_outcome = ImplementationRunOutcome(
            planning_outcome=planning_outcome,
            state=WorkflowState.VERIFYING if verification.success else WorkflowState.BLOCKED,
            routing_decision=decision,
            implementation=implementation,
            verification=verification,
            verification_path=verification_path,
            repair=repair,
            blocker_reason=repair.blocker_reason if repair else None,
        )
        if not verification.success:
            self.db_manager.update_run_status(run_id, RunStatus.BLOCKED.value)
            return PipelineOutcome(
                implementation_outcome=impl_outcome,
                state=WorkflowState.BLOCKED,
                blocker_reason=impl_outcome.blocker_reason,
            )

        return await self._review_document_and_approve(
            run_id,
            ctx,
            worktree_handle.path,
            branch_name,
            task_description,
            plan_markdown,
            task_profile,
            decision,
            verification,
            impl_outcome,
            final_approval_prompt,
            role_override,
        )

    async def _resume_reimplement(
        self,
        run_id: str,
        ctx: ProjectContext,
        worktree_manager: WorktreeManager,
        worktree_path: Path,
        branch_name: str,
        task_description: str,
        planning_outcome: PlanningOutcome,
        plan_markdown: str,
        task_profile: TaskProfile,
        role_override: RoleOverride | None,
        worktree_existed: bool,
    ) -> tuple[RoutingDecision, ImplementationOutcome]:
        """Recreate a worktree with no durable changes and re-run implementation from scratch.

        Only reached when the prior worktree (if any) has an empty diff -- nothing produced by
        a prior attempt is discarded, since there was nothing to discard.
        """
        if worktree_existed:
            self.ui.print_info(
                f"Run '{run_id}': existing worktree has no changes; recreating it before "
                "re-running implementation."
            )
            stale_handle = WorktreeHandle(
                path=worktree_path, branch_name=branch_name, repository_path=ctx.root_path
            )
            await worktree_manager.remove(stale_handle, force=True)
            await worktree_manager.delete_branch(ctx.root_path, branch_name)

        # Force the persisted state back to a known-good checkpoint before re-entering the
        # normal (validated) transition graph -- intentional, since a crash's persisted state
        # is "last completed transition before the process died," not a legal predecessor of
        # the recovery action we are about to take.
        self.db_manager.update_run_state(
            run_id,
            WorkflowState.TASK_CLASSIFIED.value,
            "Resumed: no prior implementation changes found; restarting implementation",
        )

        decision = self._route_and_persist(planning_outcome, ctx, role_override)
        implementation_workflow = ImplementationWorkflow(
            self.db_manager, self.agent_registry, worktree_manager, self.ui
        )
        implementation = await implementation_workflow.run(
            run_id, task_description, ctx, plan_markdown, task_profile, decision
        )
        return decision, implementation

    def render_runs(
        self, project_path: Path | None = None, limit: int = 20, show_all: bool = False
    ) -> None:
        """Render a table of recent runs: ID, project, task, status, stage, and timestamps."""
        self.initialize()
        project_id = None
        if not show_all:
            ctx = discover_project(explicit_path=project_path)
            project_id = ctx.project_id
        runs = self.db_manager.list_runs(project_id=project_id, limit=limit)

        self.ui.print_header("AgentFlow Runs")
        if not runs:
            self.ui.print_info("No runs found.")
            return

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Run ID", style="bold")
        table.add_column("Project")
        table.add_column("Task", overflow="fold")
        table.add_column("Status")
        table.add_column("Stage")
        table.add_column("Created")
        table.add_column("Updated")

        project_names: dict[str, str] = {}
        for r in runs:
            if r.project_id not in project_names:
                proj = self.db_manager.get_project(r.project_id)
                project_names[r.project_id] = (proj.name or proj.repository_path) if proj else "?"
            table.add_row(
                r.id,
                project_names[r.project_id],
                r.task,
                r.status,
                r.state,
                r.created_at.isoformat(timespec="seconds"),
                r.updated_at.isoformat(timespec="seconds"),
            )
        self.ui.console.print(table)

    def get_statistics(
        self, project_path: Path | None = None, show_all: bool = False
    ) -> StatisticsReport:
        """Build an informational report from persisted local data without changing routing."""
        self.initialize()
        project_id = None
        if not show_all:
            ctx = discover_project(explicit_path=project_path)
            project_id = ctx.project_id
        return StatisticsService(self.db_manager).build_report(project_id=project_id)

    def render_statistics(self, report: StatisticsReport) -> None:
        """Render reproducible local routing analytics; they are intentionally informational."""
        self.ui.print_header("AgentFlow Statistics", "Informational only — routing remains manual.")
        self.ui.console.print(f"[bold]Tracked runs:[/bold] {report.total_runs}")
        if not report.models:
            self.ui.print_info("No implementation routing decisions have been recorded yet.")
            return

        model_table = Table(show_header=True, header_style="bold magenta")
        model_table.add_column("Model")
        model_table.add_column("Tasks", justify="right")
        model_table.add_column("Routing selections", justify="right")
        model_table.add_column("First-pass verification", justify="right")
        model_table.add_column("Implementation attempts", justify="right")
        model_table.add_column("Verification passes", justify="right")
        model_table.add_column("Escalations", justify="right")
        model_table.add_column("Review findings", justify="right")
        for metric in report.models:
            first_pass = (
                "N/A" if metric.first_pass_rate is None else f"{metric.first_pass_rate:.0%}"
            )
            model_table.add_row(
                metric.model,
                str(metric.tasks),
                str(metric.routing_selections),
                first_pass,
                str(metric.implementation_attempts),
                str(metric.verification_attempts),
                str(metric.escalations),
                str(metric.review_findings),
            )
        self.ui.console.print(model_table)

        if report.routing_rule_frequency:
            self.ui.console.print("\n[bold]Matched routing rules:[/bold]")
            for rule, count in report.routing_rule_frequency.items():
                self.ui.console.print(f"  {rule}: {count}")
        if report.review_findings_by_severity:
            self.ui.console.print("\n[bold]Review findings by severity:[/bold]")
            for severity, count in report.review_findings_by_severity.items():
                self.ui.console.print(f"  {severity}: {count}")
        if report.average_stage_duration_seconds:
            self.ui.console.print("\n[bold]Average stage duration:[/bold]")
            for stage, duration in report.average_stage_duration_seconds.items():
                self.ui.console.print(f"  {stage}: {duration:.1f}s")

    async def run_cleanup(self, project_path: Path | None = None) -> CleanupReport:
        """Remove worktrees/logs for terminal runs and clear stale lock files.

        Conservative by design: a worktree is only removed once its run has reached a terminal
        state AND its writer lock is either absent or provably stale (owning process dead).
        Never removes a worktree still guarded by a live writer lock, regardless of run state.
        """
        self.initialize()
        report = CleanupReport()

        locks_root = self._locks_root()
        if locks_root.exists():
            for lock_file in locks_root.glob("*.lock"):
                lock = RunLock(lock_file.stem, locks_root)
                if lock.is_stale():
                    lock.acquire(owner="agentflow-cleanup", break_stale=True)
                    lock.release()
                    report.locks_cleared.append(f"run:{lock_file.stem}")

        scoped_project_id: str | None = None
        if project_path is not None:
            ctx = discover_project(explicit_path=project_path)
            scoped_project_id = ctx.project_id

        worktree_manager = WorktreeManager(self.config.worktrees.root, executor=self.executor)
        for project in self.db_manager.list_projects():
            if scoped_project_id is not None and project.id != scoped_project_id:
                continue
            repository_path = Path(project.repository_path)
            if not repository_path.exists():
                continue

            for run in self.db_manager.list_runs(project_id=project.id, limit=10_000):
                state = WorkflowState(run.state)
                worktree_path = worktree_manager.worktree_path_for(
                    project.name or project.id, run.id
                )

                if worktree_path.exists():
                    writer_lock = WorktreeLock(worktree_path)
                    if writer_lock.is_locked() and not writer_lock.is_stale():
                        report.skipped_active.append(run.id)
                        continue
                    if writer_lock.is_locked():
                        writer_lock.acquire(owner="agentflow-cleanup", break_stale=True)
                        writer_lock.release()
                        report.locks_cleared.append(f"worktree:{run.id}")

                    if state not in _TERMINAL_STATES:
                        report.skipped_active.append(run.id)
                        continue

                    handle = WorktreeHandle(
                        path=worktree_path,
                        branch_name=worktree_manager.branch_name_for(run.id),
                        repository_path=repository_path,
                    )
                    try:
                        await worktree_manager.remove(handle, force=True)
                        await worktree_manager.delete_branch(repository_path, handle.branch_name)
                        report.worktrees_removed.append(run.id)
                    except WorktreeError:
                        report.skipped_active.append(run.id)
                        continue
                elif state not in _TERMINAL_STATES:
                    continue

                if state in _TERMINAL_STATES:
                    verification_log_dir = (
                        repository_path / ".ai-orchestrator" / "runs" / run.id / "verification"
                    )
                    if verification_log_dir.exists():
                        shutil.rmtree(verification_log_dir, ignore_errors=True)
                        report.logs_cleared.append(run.id)

        return report

    def render_cleanup_report(self, report: CleanupReport) -> None:
        """Render a summary of what `agentflow cleanup` removed."""
        self.ui.print_header("AgentFlow Cleanup")
        self.ui.print_success(f"Worktrees removed: {len(report.worktrees_removed)}")
        self.ui.print_success(f"Locks cleared: {len(report.locks_cleared)}")
        self.ui.print_success(f"Log directories cleared: {len(report.logs_cleared)}")
        if report.skipped_active:
            self.ui.print_info(
                f"Skipped (still active): {len(report.skipped_active)} -- "
                f"{', '.join(report.skipped_active)}"
            )
