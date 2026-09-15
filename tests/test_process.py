"""Unit tests for safe process execution."""

import sys
from pathlib import Path

import pytest

from agentflow.errors import ProcessExecutionError
from agentflow.process.executor import ProcessExecutor


@pytest.mark.asyncio
async def test_process_stdout_capture():
    """Verify stdout capture and exit code 0."""
    executor = ProcessExecutor()
    res = await executor.run([sys.executable, "-c", "print('hello from subprocess')"])
    assert res.exit_code == 0
    assert "hello from subprocess" in res.stdout
    assert res.stderr == ""
    assert res.timed_out is False
    assert res.success is True


@pytest.mark.asyncio
async def test_process_stderr_capture():
    """Verify stderr capture and non-zero exit code."""
    executor = ProcessExecutor()
    res = await executor.run(
        [sys.executable, "-c", "import sys; sys.stderr.write('error occurred\\n'); sys.exit(42)"]
    )
    assert res.exit_code == 42
    assert "error occurred" in res.stderr
    assert res.stdout == ""
    assert res.timed_out is False
    assert res.success is False


@pytest.mark.asyncio
async def test_process_timeout_handling():
    """Verify process timeout handling terminates process and records timed_out=True."""
    executor = ProcessExecutor()
    res = await executor.run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=0.3,
    )
    assert res.timed_out is True
    assert res.exit_code != 0
    assert res.success is False


@pytest.mark.asyncio
async def test_process_working_directory(tmp_path: Path):
    """Verify working directory is correctly set for subprocess."""
    test_file = tmp_path / "marker.txt"
    test_file.write_text("found", encoding="utf-8")

    executor = ProcessExecutor()
    res = await executor.run(
        [sys.executable, "-c", "from pathlib import Path; print(Path('marker.txt').read_text())"],
        cwd=tmp_path,
    )
    assert res.exit_code == 0
    assert "found" in res.stdout.strip()


@pytest.mark.asyncio
async def test_process_environment_variables():
    """Verify custom environment variables are passed to subprocess."""
    executor = ProcessExecutor()
    res = await executor.run(
        [sys.executable, "-c", "import os; print(os.environ.get('AGENTFLOW_TEST_KEY'))"],
        env={"AGENTFLOW_TEST_KEY": "special_value_42"},
    )
    assert res.exit_code == 0
    assert "special_value_42" in res.stdout.strip()


@pytest.mark.asyncio
async def test_process_arguments_with_spaces_no_shell_split():
    """Arguments with spaces are passed intact without shell splitting."""
    executor = ProcessExecutor()
    arg_with_spaces = "string with spaces and 'quotes' & symbols"
    res = await executor.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", arg_with_spaces]
    )
    assert res.exit_code == 0
    assert res.stdout.strip() == arg_with_spaces


@pytest.mark.asyncio
async def test_process_empty_args_raises_error():
    """Executing empty command arguments raises ProcessExecutionError."""
    executor = ProcessExecutor()
    with pytest.raises(ProcessExecutionError, match="empty command"):
        await executor.run([])


@pytest.mark.asyncio
async def test_process_missing_executable():
    """Executing non-existent executable returns exit code 127."""
    executor = ProcessExecutor()
    res = await executor.run(["non_existent_binary_12345_xyz"])
    assert res.exit_code == 127
    assert res.success is False
    assert "not found" in res.stderr.lower()


def test_process_run_sync_wrapper():
    """Test the synchronous run_sync wrapper."""
    executor = ProcessExecutor()
    res = executor.run_sync([sys.executable, "-c", "print('sync output')"])
    assert res.exit_code == 0
    assert "sync output" in res.stdout


@pytest.mark.asyncio
async def test_process_invalid_cwd_raises_error(tmp_path: Path):
    """Specifying a non-existent cwd raises ProcessExecutionError."""
    executor = ProcessExecutor()
    nonexistent_cwd = tmp_path / "not_a_dir_at_all"
    with pytest.raises(ProcessExecutionError, match="Working directory does not exist"):
        await executor.run([sys.executable, "--version"], cwd=nonexistent_cwd)


@pytest.mark.asyncio
async def test_process_stdin_input():
    """Verify stdin data is correctly piped to subprocess."""
    executor = ProcessExecutor()
    # String input
    res_str = await executor.run(
        [sys.executable, "-c", "import sys; print(f'got:{sys.stdin.read()}')"],
        input_data="payload_data_123",
    )
    assert res_str.exit_code == 0
    assert "got:payload_data_123" in res_str.stdout

    # Bytes input
    res_bytes = await executor.run(
        [sys.executable, "-c", "import sys; print(f'bytes:{sys.stdin.read()}')"],
        input_data=b"binary_payload_456",
    )
    assert res_bytes.exit_code == 0
    assert "bytes:binary_payload_456" in res_bytes.stdout
