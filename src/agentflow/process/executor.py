"""Safe subprocess execution using asyncio.create_subprocess_exec."""

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentflow.errors import ProcessExecutionError


class ProcessResult(BaseModel):
    """Execution result contract for safe subprocess execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: datetime
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Return True if process completed normally with exit code 0."""
        return self.exit_code == 0 and not self.timed_out


class ProcessExecutor:
    """Reusable asynchronous subprocess executor adhering to safety invariants."""

    def __init__(self, default_cwd: Path | None = None) -> None:
        self.default_cwd = default_cwd

    async def run(
        self,
        cmd_args: Sequence[str | Path],
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_data: str | bytes | None = None,
    ) -> ProcessResult:
        """Execute command via asyncio.create_subprocess_exec without shell interpolation."""
        if not cmd_args:
            raise ProcessExecutionError("Cannot execute empty command arguments.")

        str_args = [str(arg) for arg in cmd_args]
        work_dir: Path | None = None
        if cwd:
            work_dir = Path(cwd).resolve()
            if not work_dir.exists():
                raise ProcessExecutionError(f"Working directory does not exist: {work_dir}")
            if not work_dir.is_dir():
                raise ProcessExecutionError(f"Working directory is not a directory: {work_dir}")
        else:
            work_dir = self.default_cwd

        # Resolve binary path via shutil.which if not a path with separators
        executable = str_args[0]
        if os.sep not in executable and (os.altsep is None or os.altsep not in executable):
            resolved_bin = shutil.which(executable)
            if resolved_bin:
                executable = resolved_bin

        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Encode input data if provided
        stdin_bytes: bytes | None = None
        if input_data is not None:
            if isinstance(input_data, str):
                stdin_bytes = input_data.encode("utf-8")
            else:
                stdin_bytes = input_data

        started_at = datetime.now(timezone.utc)
        timed_out = False
        stdout_bytes = b""
        stderr_bytes = b""
        exit_code = -1

        try:
            proc = await asyncio.create_subprocess_exec(
                executable,
                *str_args[1:],
                cwd=work_dir,
                env=process_env,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                if timeout is not None and timeout > 0:
                    out, err = await asyncio.wait_for(
                        proc.communicate(input=stdin_bytes),
                        timeout=timeout,
                    )
                else:
                    out, err = await proc.communicate(input=stdin_bytes)

                stdout_bytes = out or b""
                stderr_bytes = err or b""
                exit_code = proc.returncode if proc.returncode is not None else -1
            except (TimeoutError, asyncio.TimeoutError):
                timed_out = True
                try:
                    proc.kill()
                    out, err = await proc.communicate()
                    stdout_bytes = out or b""
                    stderr_bytes = err or b""
                except Exception:
                    pass
                exit_code = -1

        except FileNotFoundError as e:
            completed_at = datetime.now(timezone.utc)
            return ProcessResult(
                command=str_args,
                exit_code=127,
                stdout="",
                stderr=f"Executable not found: {str_args[0]} ({e})",
                started_at=started_at,
                completed_at=completed_at,
                timed_out=False,
            )
        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            raise ProcessExecutionError(f"Failed to execute subprocess '{str_args[0]}': {e}") from e

        completed_at = datetime.now(timezone.utc)

        return ProcessResult(
            command=str_args,
            exit_code=exit_code,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            started_at=started_at,
            completed_at=completed_at,
            timed_out=timed_out,
        )

    def run_sync(
        self,
        cmd_args: Sequence[str | Path],
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_data: str | bytes | None = None,
    ) -> ProcessResult:
        """Synchronous wrapper around async run."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Create a separate thread or new task if loop is running
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.run(cmd_args, cwd=cwd, env=env, timeout=timeout, input_data=input_data),
                ).result()
        else:
            return asyncio.run(
                self.run(cmd_args, cwd=cwd, env=env, timeout=timeout, input_data=input_data)
            )
