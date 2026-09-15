"""Deterministic build/test/lint verification.

Verification success is determined solely by process exit codes — never by an AI's claim
that tests passed (AGENTS.md invariant #6). Commands come from project-controlled
routing.yaml configuration, split into argv arrays and executed without a shell.
"""

import os
import shlex
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentflow.config.models import VerificationGroup
from agentflow.errors import ProcessExecutionError
from agentflow.persistence.database import DatabaseManager
from agentflow.process.executor import ProcessExecutor


class VerificationStatus(str, Enum):
    """Overall outcome of a verification pass."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMED_OUT = "TIMED_OUT"


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
        self, db_manager: DatabaseManager, executor: ProcessExecutor | None = None
    ) -> None:
        self.db_manager = db_manager
        self.executor = executor or ProcessExecutor()

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        verification_config: dict[str, VerificationGroup] | None,
        log_dir: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Run every applicable group's commands sequentially, stopping at the first failure."""
        cfg = verification_config or {}
        applicable = detect_applicable_groups(worktree_path, cfg)
        command_results: list[CommandResult] = []

        for group_name in applicable:
            for command in cfg[group_name].commands:
                cmd_result = await self._run_command(
                    run_id, worktree_path, command, log_dir, timeout_seconds
                )
                command_results.append(cmd_result)

                if cmd_result.timed_out:
                    result = VerificationResult(
                        status=VerificationStatus.TIMED_OUT,
                        groups_run=applicable,
                        command_results=command_results,
                    )
                    self._record_pass(run_id, result)
                    return result
                if not cmd_result.success:
                    result = VerificationResult(
                        status=VerificationStatus.FAILED,
                        groups_run=applicable,
                        command_results=command_results,
                    )
                    self._record_pass(run_id, result)
                    return result

        result = VerificationResult(
            status=VerificationStatus.PASSED,
            groups_run=applicable,
            command_results=command_results,
        )
        self._record_pass(run_id, result)
        return result

    def _record_pass(self, run_id: str, result: VerificationResult) -> None:
        """Record a safe summary event for one full deterministic verification pass."""
        self.db_manager.record_event(
            run_id=run_id,
            stage="VERIFYING",
            event="VERIFICATION_COMPLETED",
            attributes={
                "success": result.success,
                "status": result.status.value,
                "command_count": len(result.command_results),
            },
        )

    async def _run_command(
        self,
        run_id: str,
        worktree_path: Path,
        command: str,
        log_dir: Path | None,
        timeout_seconds: float | None,
    ) -> CommandResult:
        """Execute one configured command via argv (never a shell) and persist its result."""
        # posix=False on Windows: POSIX-mode shlex treats backslashes as escapes and would
        # mangle Windows paths (e.g. "C:\tools\pytest.exe") embedded in a command string.
        args = shlex.split(command, posix=(os.name != "nt"))
        try:
            proc_res = await self.executor.run(
                cmd_args=args, cwd=worktree_path, timeout=timeout_seconds
            )
        except ProcessExecutionError as e:
            now = datetime.now(timezone.utc)
            cmd_result = CommandResult(
                command=command,
                exit_code=1,
                stdout="",
                stderr=str(e),
                started_at=now,
                completed_at=now,
            )
            self._persist(run_id, cmd_result, log_dir)
            return cmd_result

        cmd_result = CommandResult(
            command=command,
            exit_code=proc_res.exit_code,
            stdout=proc_res.stdout,
            stderr=proc_res.stderr,
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
