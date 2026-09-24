"""Tests for deterministic verification execution (Phase 6)."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.config.models import VerificationGroup
from agentflow.errors import ConfigurationError, ProcessExecutionError
from agentflow.persistence.database import DatabaseManager
from agentflow.process.executor import ProcessResult
from agentflow.workflow.verification import (
    CommandResult,
    VerificationResult,
    VerificationRunner,
    VerificationStatus,
    describe_verification_failure,
    detect_applicable_groups,
)

PY = sys.executable


def test_verification_failure_summary_explains_pytest_collection_without_raw_output():
    """Blocked runs identify the command and safe failure cause without a traceback dump."""
    now = datetime.now(timezone.utc)
    failing = CommandResult(
        command="pytest",
        exit_code=2,
        stdout=(
            "private fixture value\n"
            "ERROR ai-engine/tests - ModuleNotFoundError: No module named 'ai'\n"
            "Interrupted: 14 errors during collection"
        ),
        stderr="",
        started_at=now,
        completed_at=now,
    )
    result = VerificationResult(status=VerificationStatus.FAILED, command_results=[failing])

    summary = describe_verification_failure(result)

    assert "pytest" in summary
    assert "exit code 2" in summary
    assert "14 errors during pytest collection" in summary
    assert "missing module 'ai'" in summary
    assert "private fixture value" not in summary


def test_verification_failure_summary_explains_missing_npm_script():
    """A missing configured script is reported as such, not as an architecture problem."""
    now = datetime.now(timezone.utc)
    failing = CommandResult(
        command="npm test",
        exit_code=1,
        stdout="",
        stderr='npm error Missing script: "test"',
        started_at=now,
        completed_at=now,
    )
    result = VerificationResult(status=VerificationStatus.FAILED, command_results=[failing])

    assert "missing npm script 'test'" in describe_verification_failure(result)


def _passing_command() -> str:
    return f"{PY} -c pass"


def _write_script(tmp_path: Path, name: str, content: str) -> Path:
    """Write a standalone script and return its path.

    Avoids embedding a quoted, multi-word `-c` snippet in a command string: on Windows,
    verification commands are split with posix=False (to keep backslash paths intact), and
    non-POSIX shlex does not group quoted whitespace the way POSIX shlex does.
    """
    script = tmp_path / name
    script.write_text(content, encoding="utf-8")
    return script


def _failing_command(tmp_path: Path) -> str:
    script = _write_script(
        tmp_path, "fail_script.py", "import sys\nsys.stderr.write('boom')\nsys.exit(1)\n"
    )
    return f"{PY} {script}"


def _sleep_command(tmp_path: Path) -> str:
    script = _write_script(tmp_path, "sleep_script.py", "import time\ntime.sleep(5)\n")
    return f"{PY} {script}"


def _new_runner(tmp_path: Path) -> VerificationRunner:
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(tmp_path))
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")
    return VerificationRunner(db)


# --- 1. Group detection -------------------------------------------------------------------


def test_detect_applicable_groups_matches_on_marker_file(tmp_path: Path):
    """A group is applicable when at least one of its detect markers exists."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = {
        "python": VerificationGroup(detect=["pyproject.toml"], commands=["pytest"]),
        "frontend": VerificationGroup(detect=["package.json"], commands=["npm test"]),
    }
    applicable = detect_applicable_groups(tmp_path, config)
    assert applicable == ["python"]


def test_detect_applicable_groups_empty_detect_is_always_applicable(tmp_path: Path):
    """A group with no detect markers configured is always applicable."""
    config = {"always": VerificationGroup(detect=[], commands=["true"])}
    assert detect_applicable_groups(tmp_path, config) == ["always"]


def test_detect_applicable_groups_none_match(tmp_path: Path):
    """No groups are applicable when none of their detect markers exist."""
    config = {"python": VerificationGroup(detect=["pyproject.toml"], commands=["pytest"])}
    assert detect_applicable_groups(tmp_path, config) == []


# --- 2. VerificationRunner: pass/fail/timeout scenarios ------------------------------------


@pytest.mark.asyncio
async def test_all_commands_pass(tmp_path: Path):
    """A verification pass where every command exits 0 is PASSED."""
    (tmp_path / "marker").write_text("", encoding="utf-8")
    runner = _new_runner(tmp_path)
    config = {
        "grp": VerificationGroup(
            detect=["marker"], commands=[_passing_command(), _passing_command()]
        )
    }

    result = await runner.run("run_1", tmp_path, config)

    assert result.status == VerificationStatus.PASSED
    assert result.success is True
    assert len(result.command_results) == 2
    assert all(c.success for c in result.command_results)


@pytest.mark.asyncio
async def test_failure_stops_the_pass(tmp_path: Path):
    """A failing command stops the pass; later commands in the same group are not run."""
    (tmp_path / "marker").write_text("", encoding="utf-8")
    runner = _new_runner(tmp_path)
    config = {
        "grp": VerificationGroup(
            detect=["marker"], commands=[_failing_command(tmp_path), _passing_command()]
        )
    }

    result = await runner.run("run_1", tmp_path, config)

    assert result.status == VerificationStatus.FAILED
    assert result.success is False
    assert len(result.command_results) == 1
    assert result.first_failure is not None
    assert result.first_failure.exit_code == 1


@pytest.mark.asyncio
async def test_verification_timeout_marks_timed_out(tmp_path: Path):
    """A command exceeding its timeout is reported as TIMED_OUT."""
    (tmp_path / "marker").write_text("", encoding="utf-8")
    runner = _new_runner(tmp_path)
    config = {"grp": VerificationGroup(detect=["marker"], commands=[_sleep_command(tmp_path)])}

    result = await runner.run("run_1", tmp_path, config, timeout_seconds=0.2)

    assert result.status == VerificationStatus.TIMED_OUT
    assert result.command_results[0].timed_out is True


@pytest.mark.asyncio
async def test_only_applicable_groups_run(tmp_path: Path):
    """Groups whose detect markers are absent are skipped entirely."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    runner = _new_runner(tmp_path)
    config = {
        "python": VerificationGroup(detect=["pyproject.toml"], commands=[_passing_command()]),
        "frontend": VerificationGroup(
            detect=["package.json"], commands=[_failing_command(tmp_path)]
        ),
    }

    result = await runner.run("run_1", tmp_path, config)

    assert result.groups_run == ["python"]
    assert result.status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_command_runs_in_configured_subdirectory(tmp_path: Path):
    """A nested Python project's pytest configuration must be resolved from its own root."""
    service = tmp_path / "service"
    service.mkdir()
    (service / "pyproject.toml").write_text("", encoding="utf-8")
    script = _write_script(
        service, "check_cwd.py", "from pathlib import Path\nassert Path.cwd().name == 'service'\n"
    )
    runner = _new_runner(tmp_path)
    config = {
        "service": VerificationGroup(
            detect=["service/pyproject.toml"],
            working_directory="service",
            commands=[f"{PY} {script.name}"],
        )
    }

    result = await runner.run("run_1", tmp_path, config)

    assert result.status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_command_rejects_working_directory_outside_worktree(tmp_path: Path):
    (tmp_path / "marker").write_text("", encoding="utf-8")
    runner = _new_runner(tmp_path)
    config = {
        "unsafe": VerificationGroup(
            detect=["marker"], working_directory="..", commands=[_passing_command()]
        )
    }

    with pytest.raises(ConfigurationError, match="inside the worktree"):
        await runner.run("run_1", tmp_path, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["pytest -q", "python -m pytest -q"])
@pytest.mark.parametrize("venv_path", [("Scripts", "python.exe"), ("bin", "python")])
async def test_pytest_uses_original_component_venv_for_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    venv_path: tuple[str, str],
):
    original = tmp_path / "original"
    worktree = tmp_path / "worktree"
    component = original / "ai-engine"
    worktree_component = worktree / "ai-engine"
    worktree_component.mkdir(parents=True)
    (worktree_component / "pyproject.toml").write_text("", encoding="utf-8")
    interpreter = component / ".venv" / Path(*venv_path)
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"test executable")
    interpreter.chmod(0o755)
    unrelated = original / ".venv"
    unrelated.mkdir()
    monkeypatch.setenv("VIRTUAL_ENV", str(unrelated))
    monkeypatch.setenv("PATH", str(unrelated / "Scripts"))

    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(original))
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")
    calls = []

    async def fake_run(cmd_args, cwd, timeout=None):
        if cmd_args[0] != "git":
            calls.append((cmd_args, cwd))
        now = datetime.now(timezone.utc)
        return ProcessResult(
            command=list(cmd_args),
            exit_code=0,
            stdout="",
            stderr="",
            started_at=now,
            completed_at=now,
        )

    runner = VerificationRunner(db)
    monkeypatch.setattr(runner.executor, "run", fake_run)
    config = {
        "ai-engine": VerificationGroup(
            detect=["ai-engine/pyproject.toml"],
            working_directory="ai-engine",
            commands=[command],
        )
    }

    result = await runner.run("run_1", worktree, config)

    assert result.status == VerificationStatus.PASSED
    assert calls == [([str(interpreter), "-m", "pytest", "-q"], worktree_component)]


@pytest.mark.asyncio
async def test_pytest_missing_component_venv_reports_error_without_path_fallback(tmp_path: Path):
    original = tmp_path / "original"
    worktree = tmp_path / "worktree"
    (worktree / "data-processor").mkdir(parents=True)
    (worktree / "data-processor" / "requirements.txt").write_text("", encoding="utf-8")
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(original))
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")
    runner = VerificationRunner(db)
    config = {
        "data-processor": VerificationGroup(
            detect=["data-processor/requirements.txt"],
            working_directory="data-processor",
            commands=["pytest"],
        )
    }

    result = await runner.run("run_1", worktree, config)

    assert result.status == VerificationStatus.ERROR
    assert result.first_failure is not None
    assert "Python virtualenv interpreter" in result.first_failure.stderr
    assert "data-processor" in result.first_failure.stderr
    assert "missing or unusable" in describe_verification_failure(result)


@pytest.mark.asyncio
async def test_pytest_unusable_component_interpreter_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    original = tmp_path / "original"
    worktree = tmp_path / "worktree"
    (worktree / "service").mkdir(parents=True)
    interpreter = original / "service" / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"broken")
    interpreter.chmod(0o755)
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(original))
    db.create_run(run_id="run_1", project_id="proj_1", task="Do a thing")
    runner = VerificationRunner(db)

    async def fail_run(cmd_args, cwd, timeout=None):
        raise ProcessExecutionError("invalid executable")

    monkeypatch.setattr(runner.executor, "run", fail_run)
    result = await runner.run(
        "run_1",
        worktree,
        {
            "service": VerificationGroup(
                working_directory="service",
                commands=["pytest"],
            )
        },
    )

    assert result.status == VerificationStatus.ERROR
    assert result.first_failure is not None
    assert "Python virtualenv interpreter is unusable" in result.first_failure.stderr


# --- 3. Persistence -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_runs_are_persisted_to_db(tmp_path: Path):
    """Each command execution is recorded as a verification_runs DB row."""
    (tmp_path / "marker").write_text("", encoding="utf-8")
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(tmp_path))
    db.create_run(run_id="run_db", project_id="proj_1", task="Do a thing")
    runner = VerificationRunner(db)
    config = {"grp": VerificationGroup(detect=["marker"], commands=[_passing_command()])}

    await runner.run("run_db", tmp_path, config)

    records = db.list_verification_runs("run_db")
    assert len(records) == 1
    assert records[0].exit_code == 0
    events = db.list_events("run_db")
    assert len(events) == 1
    assert events[0].event == "VERIFICATION_COMPLETED"
    assert events[0].attributes == {
        "command_count": 1,
        "status": "PASSED",
        "success": True,
        "phase": "initial",
        "selected_group_count": 1,
        "cached_group_count": 0,
    }


@pytest.mark.asyncio
async def test_verification_logs_written_when_log_dir_given(tmp_path: Path):
    """Per-command stdout/stderr logs are written when a log_dir is supplied."""
    (tmp_path / "marker").write_text("", encoding="utf-8")
    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    db.upsert_project("proj_1", "Project One", str(tmp_path))
    db.create_run(run_id="run_logs", project_id="proj_1", task="Do a thing")
    runner = VerificationRunner(db)
    log_dir = tmp_path / "verification"
    config = {"grp": VerificationGroup(detect=["marker"], commands=[_passing_command()])}

    await runner.run("run_logs", tmp_path, config, log_dir=log_dir)

    records = db.list_verification_runs("run_logs")
    assert records[0].stdout_path is not None
    assert Path(records[0].stdout_path).exists()


def test_write_artifact_persists_verification_json(tmp_path: Path):
    """write_artifact() persists a machine-readable verification.json under the run directory."""
    from agentflow.workflow.verification import VerificationResult

    db = DatabaseManager(tmp_path / "test.db")
    db.initialize()
    runner = VerificationRunner(db)

    result = VerificationResult(status=VerificationStatus.PASSED, groups_run=[], command_results=[])
    path = runner.write_artifact(tmp_path, "run_artifact", result)

    assert path.exists()
    assert path.name == "verification.json"
    assert path.parent == tmp_path / ".ai-orchestrator" / "runs" / "run_artifact"
