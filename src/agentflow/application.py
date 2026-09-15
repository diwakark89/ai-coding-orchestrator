"""Application container and workflow orchestration core."""

import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table

from agentflow.agents.registry import AgentAdapterRegistry, create_default_registry
from agentflow.config.loader import load_global_config
from agentflow.config.models import GlobalConfig
from agentflow.errors import ProjectNotFoundError
from agentflow.persistence.database import DatabaseManager
from agentflow.process.executor import ProcessExecutor
from agentflow.project.discovery import discover_project
from agentflow.ui.console import CHECKMARK, CROSSMARK, WARNINGMARK, ConsoleUI
from agentflow.workflow.planning import PlanningOutcome, PlanningWorkflow


@dataclass
class DoctorCheckItem:
    """Individual health check evaluated by agentflow doctor."""

    name: str
    passed: bool
    critical: bool
    details: str = ""
    is_warning: bool = False


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
        """Check availability and functionality of a provider CLI using a lightweight command."""
        cli_path = shutil.which(command)
        if not cli_path:
            return DoctorCheckItem(
                name=f"{provider_name} CLI",
                passed=False,
                critical=False,
                is_warning=True,
                details=f"Executable '{command}' not found in PATH",
            )

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
                return DoctorCheckItem(
                    name=f"{provider_name} CLI",
                    passed=True,
                    critical=False,
                    details=details,
                )
            return DoctorCheckItem(
                name=f"{provider_name} CLI",
                passed=False,
                critical=False,
                is_warning=True,
                details=(
                    f"Executable at {cli_path} failed '--version' check (exit code {res.exit_code})"
                ),
            )
        except Exception as e:
            return DoctorCheckItem(
                name=f"{provider_name} CLI",
                passed=False,
                critical=False,
                is_warning=True,
                details=f"Executable at {cli_path} could not be executed: {e}",
            )

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

        report.items.append(self.check_python())
        report.items.append(self.check_git())
        report.items.append(self.check_cli("Claude", self.config.cli.claude.command))
        report.items.append(self.check_cli("Codex", self.config.cli.codex.command))
        report.items.append(self.check_cli("Gemini", self.config.cli.gemini.command))
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
            if item.passed:
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
            f"  - Run {r.id}: Task='{r.task}', Status={r.status}, "
            f"Started={r.created_at.isoformat()}"
            for r in active_runs
        )
        return (
            f"Project: {ctx.project_name} (ID: {ctx.project_id[:12]})\n"
            f"Active runs ({len(active_runs)}):\n{runs_summary}"
        )

    async def run_planning(
        self, task_description: str, project_path: Path | None = None
    ) -> PlanningOutcome:
        """Run interactive Claude-based planning for a task through TaskProfile generation."""
        self.initialize()
        ctx = discover_project(explicit_path=project_path)
        workflow = PlanningWorkflow(
            db_manager=self.db_manager,
            agent_registry=self.agent_registry,
            console_ui=self.ui,
        )
        return await workflow.run(task_description, ctx)
