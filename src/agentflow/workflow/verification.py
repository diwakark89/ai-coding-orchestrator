"""Deterministic build/test/lint verification.

Verification success is determined solely by process exit codes — never by an AI's claim
that tests passed (AGENTS.md invariant #6). Commands come from project-controlled
routing.yaml configuration, split into argv arrays and executed without a shell.
"""

import hashlib
import os
import re
import shlex
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentflow.config.models import VerificationGroup
from agentflow.errors import ConfigurationError, ProcessExecutionError
from agentflow.persistence.database import DatabaseManager
from agentflow.process.executor import ProcessExecutor
from agentflow.ui.console import ConsoleUI
from agentflow.workflow.verification_scope import (
    ChangeSnapshot,
    GroupSelection,
    select_groups,
    snapshot_changes,
)


class VerificationStatus(str, Enum):
    """Overall outcome of a verification pass."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMED_OUT = "TIMED_OUT"
    INTERMITTENT = "INTERMITTENT"


class CommandResult(BaseModel):
    """Captured result of a single verification command execution."""

    model_config = ConfigDict(extra="ignore")

    command: str
    exit_code: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: datetime
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Return True if the command exited 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out


class VerificationResult(BaseModel):
    """Aggregate result of running every applicable verification group's commands."""

    model_config = ConfigDict(extra="ignore")

    status: VerificationStatus
    groups_run: list[str] = Field(default_factory=list)
    command_results: list[CommandResult] = Field(default_factory=list)
    selection_reasons: dict[str, list[str]] = Field(default_factory=dict)
    suppressed_groups: dict[str, str] = Field(default_factory=dict)
    rerun_groups: list[str] = Field(default_factory=list)
    cached_groups: list[str] = Field(default_factory=list)
    documentation_only_paths: list[str] = Field(default_factory=list)
    failed_group: str | None = None
    phase: str = "initial"
    reproduction_result: CommandResult | None = None
    executed_command_count: int = 0

    @property
    def success(self) -> bool:
        """Return True only if every command in the pass exited 0."""
        return self.status == VerificationStatus.PASSED

    @property
    def first_failure(self) -> CommandResult | None:
        """The first command that did not succeed, if any."""
        for result in self.command_results:
            if not result.success:
                return result
        return None


def describe_verification_failure(result: VerificationResult) -> str:
    """Summarize a failed command without copying arbitrary test output into run state."""
    failing = result.first_failure
    if failing is None:
        return f"Verification ended in state {result.status.value} without a failing command."

    description = f"Verification command '{failing.command}' failed (exit code {failing.exit_code})"
    if failing.timed_out:
        return f"{description}: timed out."

    output = f"{failing.stdout}\n{failing.stderr}"
    details: list[str] = []
    if result.status == VerificationStatus.ERROR and "Python virtualenv interpreter" in output:
        details.append("Python virtualenv interpreter is missing or unusable")
    collection = re.search(r"\b(\d+) errors? during collection\b", output, re.IGNORECASE)
    if collection:
        details.append(f"{collection.group(1)} errors during pytest collection")

    missing_module = re.search(r"No module named ['\"]([A-Za-z_][A-Za-z0-9_.-]*)['\"]", output)
    if missing_module:
        details.append(f"ModuleNotFoundError: missing module '{missing_module.group(1)}'")
    else:
        missing_script = re.search(r"Missing script: ['\"]([A-Za-z0-9:_-]+)['\"]", output)
        if missing_script:
            details.append(f"missing npm script '{missing_script.group(1)}'")

    if not details:
        exception = re.search(
            r"\b(ModuleNotFoundError|ImportError|SyntaxError|TypeError|AssertionError|"
            r"FileNotFoundError|ConnectionError|TimeoutError)\b",
            output,
        )
        if exception:
            details.append(exception.group(1))

    failed_test = re.search(r"(?:^|\n)(?:FAILED|FAIL)\s+([A-Za-z0-9_./\\:#\[\]-]{1,180})", output)
    if failed_test:
        details.append(f"failed test '{failed_test.group(1)}'")
    if result.reproduction_result is not None:
        details.append(
            "bounded reproduction passed; original failure is intermittent"
            if result.reproduction_result.success
            else "bounded reproduction failed again"
        )

    return f"{description}: {'; '.join(details)}." if details else f"{description}."


# Appended to every prompt for an agent that edits the worktree. Coding CLIs otherwise run
# whole test suites on their own, duplicating the orchestrator's deterministic pass, and
# may "fix" setup failures by editing test tooling instead of reporting them.
AGENT_VERIFICATION_GUIDANCE = (
    "Verification:\n"
    "- AgentFlow runs the project's configured build and test commands itself after you "
    "finish; only their exit codes decide success.\n"
    "- Do not run full test suites or full builds (for example a whole Maven reactor, every "
    "pytest suite, or the entire Jest suite). If you need feedback, run only the narrowest "
    "check for the code you changed, such as a single test file or test class.\n"
    "- Do not change test scripts, build or dependency configuration, or environment setup "
    "just to make verification pass. If tests cannot run because of the environment (missing "
    "modules, tools, or scripts), stop and report that as a blocker."
)


@runtime_checkable
class VerificationRunnerLike(Protocol):
    """The subset of VerificationRunner's interface other workflows depend on.

    Lets ReviewWorkflow, DocumentationWorkflow, and RepairWorkflow accept any object with a
    compatible `run()` (real or scripted-for-testing) without coupling to the concrete class.
    """

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        verification_config: dict[str, VerificationGroup] | None,
        log_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult: ...


async def reverify_after_edit(
    runner: VerificationRunnerLike,
    run_id: str,
    worktree_path: Path,
    verification_config: dict[str, VerificationGroup] | None,
    log_dir: Path | None = None,
) -> VerificationResult:
    """Re-verify after a repair, review fix, or doc edit, reusing unaffected passes if possible.

    Runners that only implement `run()` (e.g. scripted test doubles) get a full pass.
    """
    incremental = getattr(runner, "run_incremental", None)
    if incremental is None:
        return await runner.run(run_id, worktree_path, verification_config, log_dir=log_dir)
    result: VerificationResult = await incremental(
        run_id, worktree_path, verification_config, log_dir=log_dir
    )
    return result


def detect_applicable_groups(
    repository_path: Path, verification_config: dict[str, VerificationGroup]
) -> list[str]:
    """Return group names (config order) whose detect marker files are present.

    A group with an empty detect list has no gating condition and is always applicable.
    """
    applicable = []
    for name, group in verification_config.items():
        if not group.detect or any((repository_path / marker).exists() for marker in group.detect):
            applicable.append(name)
    return applicable


class VerificationRunner:
    """Detects applicable verification groups, runs their commands, and persists the results."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        executor: ProcessExecutor | None = None,
        console_ui: ConsoleUI | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.executor = executor or ProcessExecutor()
        self.ui = console_ui or ConsoleUI()
        self._snapshot: ChangeSnapshot | None = None
        self._run_id: str | None = None
        self._worktree: Path | None = None
        self._group_results: dict[str, list[CommandResult]] = {}
        self._reproduced: dict[tuple[str, str, str], CommandResult] = {}
        self._last_failed_group: str | None = None

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        verification_config: dict[str, VerificationGroup] | None,
        log_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Verify groups touched by current worktree changes; reset any prior cache."""
        self._run_id = run_id
        self._worktree = worktree_path.resolve()
        self._group_results.clear()
        self._reproduced.clear()
        self._last_failed_group = None
        self._snapshot = await snapshot_changes(self.executor, worktree_path)
        cfg = verification_config or {}
        applicable = detect_applicable_groups(worktree_path, cfg)
        selection = select_groups(
            cfg, applicable, self._snapshot.paths, reliable=self._snapshot.reliable
        )
        return await self._execute_selection(
            run_id, worktree_path, cfg, selection, "initial", log_dir, timeout_seconds
        )

    async def run_incremental(
        self,
        run_id: str,
        worktree_path: Path,
        verification_config: dict[str, VerificationGroup] | None,
        log_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Re-verify after an agent edit, rerunning only groups whose inputs may have changed.

        A retained pass is reused only when no path changed since that pass selects its
        group, so it is exactly as trustworthy as a fresh run of the same unchanged files.
        The previously failed group runs first so a persisting failure surfaces quickly.
        """
        if self._run_id != run_id or self._worktree != worktree_path.resolve():
            return await self.run(
                run_id, worktree_path, verification_config, log_dir, timeout_seconds
            )
        cfg = verification_config or {}
        snapshot = await snapshot_changes(self.executor, worktree_path)
        previous = self._snapshot
        changed = snapshot.changed_since(previous) if previous else snapshot.paths
        self._snapshot = snapshot
        applicable = detect_applicable_groups(worktree_path, cfg)
        selection = select_groups(cfg, applicable, snapshot.paths, reliable=snapshot.reliable)
        rerun = (
            set(select_groups(cfg, applicable, changed, reliable=True).groups)
            if changed and snapshot.reliable
            else set()
        )
        if not snapshot.reliable or (previous is not None and not previous.reliable):
            rerun.update(selection.groups)
        return await self._execute_selection(
            run_id,
            worktree_path,
            cfg,
            selection,
            "incremental",
            log_dir,
            timeout_seconds,
            force_groups=rerun,
        )

    async def reproduce_failure(
        self,
        run_id: str,
        worktree_path: Path,
        verification_config: dict[str, VerificationGroup] | None,
        result: VerificationResult,
        log_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Run the failed command once more; retain both outcomes as evidence."""
        group_name = result.failed_group
        failure = result.first_failure
        cfg = verification_config or {}
        if group_name is None or failure is None or group_name not in cfg:
            return result
        fingerprints = self._snapshot.fingerprints if self._snapshot else {}
        state_hash = hashlib.sha256(repr(sorted(fingerprints.items())).encode()).hexdigest()
        key = (group_name, failure.command, state_hash)
        if key in self._reproduced:
            result.reproduction_result = self._reproduced[key]
            if result.reproduction_result.success:
                result.status = VerificationStatus.INTERMITTENT
            return result
        cwd = self._group_cwd(worktree_path, cfg[group_name], group_name)
        retry, _ = await self._execute_configured_command(
            run_id,
            cwd,
            cfg[group_name].working_directory,
            group_name,
            failure.command,
            log_dir,
            timeout_seconds,
        )
        self._reproduced[key] = retry
        result.reproduction_result = retry
        if retry.success:
            result.status = VerificationStatus.INTERMITTENT
        self.db_manager.record_event(
            run_id=run_id,
            stage="VERIFYING",
            event="VERIFICATION_REPRODUCTION",
            attributes={"reproduced": not retry.success, "command_count": 1},
        )
        return result

    async def _execute_selection(
        self,
        run_id: str,
        worktree_path: Path,
        config: dict[str, VerificationGroup],
        selection: GroupSelection,
        phase: str,
        log_dir: Path | None,
        timeout_seconds: float | None,
        force_groups: set[str] | None = None,
    ) -> VerificationResult:
        required = selection.groups
        force = force_groups or set()
        rerun: list[str] = []
        cached: list[str] = []
        executed_count = 0
        self.ui.print_info(
            f"Verification {phase}: "
            + (
                ", ".join(f"{name} ({'; '.join(selection.reasons[name])})" for name in required)
                or "no applicable groups"
            )
        )
        if selection.documentation_only:
            self.ui.print_info(
                f"Documentation-only changes need no verification group: "
                f"{len(selection.documentation_only)} file(s)"
            )
        scheduled = [
            name
            for name in required
            if name in force
            or name not in self._group_results
            or not all(result.success for result in self._group_results[name])
        ]
        reused = [name for name in required if name not in scheduled]
        # Fail fast: a failure that survived the agent's edit most likely persists.
        order = list(required)
        if self._last_failed_group in scheduled:
            order.remove(self._last_failed_group)
            order.insert(0, self._last_failed_group)
        self.ui.print_info(
            f"Rerunning groups: {', '.join(scheduled) or 'none'}; "
            f"retaining passes: {', '.join(reused) or 'none'}"
        )
        if selection.suppressed:
            self.ui.print_info(
                "Covered duplicate groups: "
                + ", ".join(
                    f"{name} by {covering}" for name, covering in selection.suppressed.items()
                )
            )
        for group_name in order:
            if (
                group_name not in force
                and group_name in self._group_results
                and all(result.success for result in self._group_results[group_name])
            ):
                cached.append(group_name)
                continue
            rerun.append(group_name)
            group_results: list[CommandResult] = []
            group_cwd = self._group_cwd(worktree_path, config[group_name], group_name)
            for command in config[group_name].commands:
                cmd_result, command_status = await self._execute_configured_command(
                    run_id,
                    group_cwd,
                    config[group_name].working_directory,
                    group_name,
                    command,
                    log_dir,
                    timeout_seconds,
                )
                group_results.append(cmd_result)
                executed_count += 1
                self._group_results[group_name] = group_results
                if command_status != VerificationStatus.PASSED:
                    self._last_failed_group = group_name
                    result = self._build_result(
                        command_status,
                        selection,
                        rerun,
                        cached,
                        group_name,
                        phase,
                        executed_count,
                    )
                    self._record_pass(run_id, result)
                    return result
            self._group_results[group_name] = group_results
        self._last_failed_group = None
        result = self._build_result(
            VerificationStatus.PASSED, selection, rerun, cached, None, phase, executed_count
        )
        self._record_pass(run_id, result)
        return result

    def _build_result(
        self,
        status: VerificationStatus,
        selection: GroupSelection,
        rerun: list[str],
        cached: list[str],
        failed_group: str | None,
        phase: str,
        executed_count: int,
    ) -> VerificationResult:
        return VerificationResult(
            status=status,
            groups_run=selection.groups,
            command_results=[
                result for name in selection.groups for result in self._group_results.get(name, [])
            ],
            selection_reasons=selection.reasons,
            suppressed_groups=selection.suppressed,
            rerun_groups=list(rerun),
            cached_groups=list(cached),
            documentation_only_paths=list(selection.documentation_only),
            failed_group=failed_group,
            phase=phase,
            executed_command_count=executed_count,
        )

    @staticmethod
    def _group_cwd(worktree_path: Path, group: VerificationGroup, name: str) -> Path:
        directory = group.working_directory
        if Path(directory).is_absolute() or re.match(r"^[A-Za-z]:", directory):
            raise ConfigurationError(
                f"Verification group '{name}' working_directory must be relative."
            )
        group_cwd = (worktree_path / directory).resolve()
        if not group_cwd.is_relative_to(worktree_path.resolve()) or not group_cwd.is_dir():
            raise ConfigurationError(
                f"Verification group '{name}' working_directory must be an existing "
                "directory inside the worktree."
            )
        return group_cwd

    async def _execute_configured_command(
        self,
        run_id: str,
        group_cwd: Path,
        directory: str,
        group_name: str,
        command: str,
        log_dir: Path | None,
        timeout_seconds: float | None,
    ) -> tuple[CommandResult, VerificationStatus]:
        args = shlex.split(command, posix=(os.name != "nt"))
        uses_component_python = self._is_pytest_command(args)
        if uses_component_python:
            interpreter, error = self._component_python(run_id, directory, group_name)
            if error is not None:
                now = datetime.now(timezone.utc)
                result = CommandResult(
                    command=command,
                    exit_code=127,
                    stdout="",
                    stderr=error,
                    started_at=now,
                    completed_at=now,
                )
                self._persist(run_id, result, log_dir)
                return result, VerificationStatus.ERROR
            assert interpreter is not None
            args = (
                [str(interpreter), "-m", "pytest", *args[1:]]
                if args[0] == "pytest"
                else [str(interpreter), *args[1:]]
            )
        result = await self._run_command(
            run_id,
            group_cwd,
            command,
            log_dir,
            timeout_seconds,
            args=args,
            component_python=uses_component_python,
        )
        if result.timed_out:
            return result, VerificationStatus.TIMED_OUT
        if result.exit_code == 127:
            return result, VerificationStatus.ERROR
        if uses_component_python and re.search(r"No module named ['\"]pytest['\"]", result.stderr):
            return result, VerificationStatus.ERROR
        return result, VerificationStatus.PASSED if result.success else VerificationStatus.FAILED

    @staticmethod
    def _is_pytest_command(args: list[str]) -> bool:
        return bool(args) and (args[0] == "pytest" or args[:3] == ["python", "-m", "pytest"])

    def _component_python(
        self, run_id: str, directory: str, group_name: str
    ) -> tuple[Path | None, str | None]:
        run = self.db_manager.get_run(run_id)
        project = self.db_manager.get_project(run.project_id) if run else None
        if project is None:
            return None, (
                f"Python virtualenv interpreter for group '{group_name}' is unavailable: "
                "original project was not found."
            )
        component = Path(project.repository_path).resolve() / directory
        candidates = [
            component / env / Path(*relative)
            for env in (".venv", "venv")
            for relative in (
                ("Scripts", "python.exe"),
                ("bin", "python"),
            )
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate, None
        return None, (
            f"Python virtualenv interpreter for group '{group_name}' is missing or unusable "
            f"in original project component '{component}'. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )

    def _record_pass(self, run_id: str, result: VerificationResult) -> None:
        """Record a safe summary event for one full deterministic verification pass."""
        self.db_manager.record_event(
            run_id=run_id,
            stage="VERIFYING",
            event="VERIFICATION_COMPLETED",
            attributes={
                "success": result.success,
                "status": result.status.value,
                "phase": result.phase,
                "selected_group_count": len(result.groups_run),
                "cached_group_count": len(result.cached_groups),
                "command_count": result.executed_command_count,
            },
        )

    async def _run_command(
        self,
        run_id: str,
        worktree_path: Path,
        command: str,
        log_dir: Path | None,
        timeout_seconds: float | None,
        args: list[str] | None = None,
        component_python: bool = False,
    ) -> CommandResult:
        """Execute one configured command via argv (never a shell) and persist its result."""
        # posix=False on Windows: POSIX-mode shlex treats backslashes as escapes and would
        # mangle Windows paths (e.g. "C:\tools\pytest.exe") embedded in a command string.
        args = args if args is not None else shlex.split(command, posix=(os.name != "nt"))
        try:
            async with self.ui.animate_stage(f"Verifying ({command})"):
                proc_res = await self.executor.run(
                    cmd_args=args, cwd=worktree_path, timeout=timeout_seconds
                )
        except ProcessExecutionError as e:
            now = datetime.now(timezone.utc)
            cmd_result = CommandResult(
                command=command,
                exit_code=127 if component_python else 1,
                stdout="",
                stderr=(
                    f"Python virtualenv interpreter is unusable: {e}"
                    if component_python
                    else str(e)
                ),
                started_at=now,
                completed_at=now,
            )
            self._persist(run_id, cmd_result, log_dir)
            return cmd_result

        stderr = proc_res.stderr
        if component_python and proc_res.exit_code == 127:
            stderr = f"Python virtualenv interpreter is unusable: {stderr}"
        cmd_result = CommandResult(
            command=command,
            exit_code=proc_res.exit_code,
            stdout=proc_res.stdout,
            stderr=stderr,
            started_at=proc_res.started_at,
            completed_at=proc_res.completed_at,
            timed_out=proc_res.timed_out,
        )
        self._persist(run_id, cmd_result, log_dir)
        return cmd_result

    def _persist(self, run_id: str, cmd_result: CommandResult, log_dir: Path | None) -> None:
        """Write per-command logs (if log_dir given) and record a verification_runs row."""
        stdout_path: str | None = None
        stderr_path: str | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            index = len(self.db_manager.list_verification_runs(run_id)) + 1
            stdout_file = log_dir / f"{index:02d}.stdout.log"
            stderr_file = log_dir / f"{index:02d}.stderr.log"
            stdout_file.write_text(cmd_result.stdout, encoding="utf-8")
            stderr_file.write_text(cmd_result.stderr, encoding="utf-8")
            stdout_path = str(stdout_file)
            stderr_path = str(stderr_file)

        self.db_manager.record_verification_run(
            str(uuid.uuid4()),
            run_id,
            cmd_result.command,
            cmd_result.exit_code,
            cmd_result.started_at,
            cmd_result.completed_at,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def write_artifact(self, project_root: Path, run_id: str, result: VerificationResult) -> Path:
        """Persist the human/machine-readable verification.json artifact for a run."""
        run_dir = project_root / ".ai-orchestrator" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "verification.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path
