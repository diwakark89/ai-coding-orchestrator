"""Unit and integration tests for AI agent CLI adapters and registry."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.agents import (
    EXCLUDED_MODELS,
    AgentAdapterRegistry,
    AgentRequest,
    AgentResult,
    AgentRole,
    AntigravityAdapter,
    ClaudeAdapter,
    CodexAdapter,
    GeminiAdapter,
    Provider,
    create_default_registry,
    validate_model_allowed,
)
from agentflow.application import Application
from agentflow.config.models import GlobalConfig
from agentflow.errors import (
    AdapterNotFoundError,
    UnsupportedCapabilityError,
)
from agentflow.process.executor import ProcessExecutor, ProcessResult

# ---------------------------------------------------------------------------
# 1. Enums, Models & Validation
# ---------------------------------------------------------------------------


def test_provider_enum():
    """Verify Provider enum values and case-insensitive parsing."""
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.GOOGLE.value == "google"

    assert Provider.from_string("anthropic") == Provider.ANTHROPIC
    assert Provider.from_string("ANTHROPIC") == Provider.ANTHROPIC
    assert Provider.from_string("OpenAI") == Provider.OPENAI
    assert Provider.from_string("google") == Provider.GOOGLE

    with pytest.raises(ValueError, match="Unknown provider"):
        Provider.from_string("unknown_provider")


def test_validate_model_allowed():
    """Verify forbidden V1 models are rejected and valid models pass."""
    # Forbidden models must be rejected
    for forbidden in EXCLUDED_MODELS:
        with pytest.raises(ValueError, match="explicitly excluded"):
            validate_model_allowed(forbidden)
        with pytest.raises(ValueError, match="explicitly excluded"):
            validate_model_allowed(forbidden.upper())

    # Permitted models must pass
    validate_model_allowed("Claude Sonnet 5")
    validate_model_allowed("sonnet")
    validate_model_allowed("claude-opus-5-5")
    validate_model_allowed("GPT-6 Luna")
    validate_model_allowed("gpt-6-sol")
    validate_model_allowed("Gemini 3.8 Flash")


def test_agent_request_model_validation(tmp_path: Path):
    """AgentRequest validates model and non-empty prompt."""
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Design architecture",
        repository_path=tmp_path,
        model="Claude Sonnet 5",
    )
    assert req.model == "Claude Sonnet 5"
    assert req.effective_working_directory == tmp_path
    assert req.worktree_path is None

    # Empty prompt rejected
    with pytest.raises(ValueError, match="prompt cannot be empty"):
        AgentRequest(
            role=AgentRole.PLANNER,
            prompt="   ",
            repository_path=tmp_path,
            model="Claude Sonnet 5",
        )

    # Excluded model rejected
    with pytest.raises(ValueError, match="explicitly excluded"):
        AgentRequest(
            role=AgentRole.IMPLEMENTER,
            prompt="Write code",
            repository_path=tmp_path,
            model="GPT-5.4 Mini",
        )


def test_agent_request_working_directory_override(tmp_path: Path):
    """working_directory overrides repository_path when set."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    req = AgentRequest(
        role=AgentRole.STANDARD_CODER,
        prompt="Implement feature",
        repository_path=tmp_path,
        working_directory=worktree,
        model="GPT-6 Sol",
    )
    assert req.effective_working_directory == worktree
    assert req.worktree_path == worktree


def test_agent_result_properties():
    """AgentResult properties success and duration_seconds compute accurately."""
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 15, tzinfo=timezone.utc)

    success_res = AgentResult(
        provider=Provider.ANTHROPIC,
        model="sonnet",
        exit_code=0,
        started_at=start,
        completed_at=end,
        timed_out=False,
    )
    assert success_res.success is True
    assert success_res.duration_seconds == 15.0

    fail_res = AgentResult(
        provider=Provider.OPENAI,
        model="gpt-6-sol",
        exit_code=1,
        started_at=start,
        completed_at=end,
        timed_out=False,
    )
    assert fail_res.success is False

    timeout_res = AgentResult(
        provider=Provider.GOOGLE,
        model="gemini-3.8-flash",
        exit_code=0,
        started_at=start,
        completed_at=end,
        timed_out=True,
    )
    assert timeout_res.success is False


# ---------------------------------------------------------------------------
# 2. Adapter Capabilities
# ---------------------------------------------------------------------------


def test_adapter_capabilities_reporting():
    """Each adapter reports distinct, accurate capability metadata."""
    claude = ClaudeAdapter()
    assert claude.capabilities.supports_resume is True
    assert claude.capabilities.supports_read_only_mode is True
    assert claude.capabilities.supports_structured_output is True
    assert claude.capabilities.supports_model_selection is True

    codex = CodexAdapter()
    assert codex.capabilities.supports_resume is False
    assert codex.capabilities.supports_read_only_mode is False
    assert codex.capabilities.supports_structured_output is True
    assert codex.capabilities.supports_model_selection is True

    gemini = GeminiAdapter()
    assert gemini.capabilities.supports_resume is True
    assert gemini.capabilities.supports_read_only_mode is True
    assert gemini.capabilities.supports_structured_output is True
    assert gemini.capabilities.supports_model_selection is True


# ---------------------------------------------------------------------------
# 3. ClaudeAdapter Tests
# ---------------------------------------------------------------------------


def test_claude_adapter_build_args(tmp_path: Path):
    """ClaudeAdapter constructs arguments correctly with model alias and flags."""
    adapter = ClaudeAdapter(command="claude")

    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Explain design",
        repository_path=tmp_path,
        model="Claude Sonnet 5",
        read_only=True,
    )
    args = adapter.build_args(req)
    assert args == [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        "sonnet",
        "--permission-mode",
        "plan",
        "Explain design",
    ]


def test_claude_adapter_build_args_resume(tmp_path: Path):
    """ClaudeAdapter adds --resume <session_id> when session_id provided."""
    adapter = ClaudeAdapter(command="claude")
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Next question",
        repository_path=tmp_path,
        model="Claude Opus 5.5",
        read_only=False,
    )
    args = adapter.build_args(req, session_id="session-xyz-123")
    assert "--resume" in args
    idx = args.index("--resume")
    assert args[idx + 1] == "session-xyz-123"
    assert args[args.index("--model") + 1] == "claude-opus-5-5"


@pytest.mark.parametrize(
    "model",
    ["Claude Opus 5.5", "claude-opus-5-5", "Claude Opus", "Claude Opus 5", "claude-opus-5"],
)
def test_claude_adapter_pins_opus_to_5_5(tmp_path: Path, model: str):
    """All Opus names, including legacy Opus 5 ones, pin to the exact Opus 5.5 model ID."""
    adapter = ClaudeAdapter(command="claude")
    req = AgentRequest(
        role=AgentRole.ARCHITECTURE_PLANNER,
        prompt="Plan",
        repository_path=tmp_path,
        model=model,
    )
    args = adapter.build_args(req)
    assert args[args.index("--model") + 1] == "claude-opus-5-5"


@pytest.mark.asyncio
async def test_claude_adapter_start_json_output(tmp_path: Path, monkeypatch):
    """ClaudeAdapter parses JSON response and extracts text and session ID."""
    mock_json = '{"type": "result", "result": "Here is the plan.", "session_id": "sess-456-uuid"}'
    start_time = datetime.now(timezone.utc)

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=mock_json,
            stderr="",
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = ClaudeAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Create plan",
        repository_path=tmp_path,
        model="sonnet",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.provider == Provider.ANTHROPIC
    assert result.model == "sonnet"
    assert result.text == "Here is the plan."
    assert result.session_id == "sess-456-uuid"
    assert len(result.raw_events) == 1


@pytest.mark.asyncio
async def test_claude_adapter_resume(tmp_path: Path, monkeypatch):
    """ClaudeAdapter resumes session successfully and captures updated session ID."""
    mock_json = '{"type": "result", "result": "Answers recorded.", "session_id": "sess-456-uuid"}'

    captured_cmd = []

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        captured_cmd.extend(cmd_args)
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=mock_json,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = ClaudeAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Answer: Option B",
        repository_path=tmp_path,
        model="sonnet",
    )
    result = await adapter.resume("sess-456-uuid", req)

    assert result.success is True
    assert "--resume" in captured_cmd
    assert result.text == "Answers recorded."


@pytest.mark.asyncio
async def test_claude_adapter_empty_session_id_resume_raises_error(tmp_path: Path):
    """Calling resume with an empty session ID raises ValueError."""
    adapter = ClaudeAdapter()
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Next question",
        repository_path=tmp_path,
        model="sonnet",
    )
    with pytest.raises(ValueError, match="session_id cannot be empty"):
        await adapter.resume("   ", req)


# ---------------------------------------------------------------------------
# 4. CodexAdapter Tests
# ---------------------------------------------------------------------------


def test_codex_adapter_build_args(tmp_path: Path):
    """CodexAdapter constructs exec argument array with model selection."""
    adapter = CodexAdapter(command="codex")
    req = AgentRequest(
        role=AgentRole.LIGHTWEIGHT_CODER,
        prompt="Fix import error",
        repository_path=tmp_path,
        model="GPT-6 Luna",
    )
    args = adapter.build_args(req)
    assert args == [
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-6-luna",
        "Fix import error",
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("GPT-6 Luna", "gpt-6-luna"),
        ("GPT-6 Sol", "gpt-6-sol"),
        ("GPT-5.6 Luna", "gpt-6-luna"),
        ("GPT-5.6 Terra", "gpt-6-sol"),
        ("terra", "gpt-6-sol"),
    ],
)
def test_codex_adapter_upgrades_legacy_gpt_names(tmp_path: Path, model: str, expected: str):
    """Current and legacy GPT-5.6 names resolve to the GPT-6 model IDs."""
    adapter = CodexAdapter(command="codex")
    req = AgentRequest(
        role=AgentRole.STANDARD_CODER,
        prompt="Implement",
        repository_path=tmp_path,
        model=model,
    )
    args = adapter.build_args(req)
    assert args[args.index("--model") + 1] == expected


@pytest.mark.asyncio
async def test_codex_adapter_resume_unsupported(tmp_path: Path):
    """CodexAdapter explicitly raises UnsupportedCapabilityError on resume."""
    adapter = CodexAdapter()
    req = AgentRequest(
        role=AgentRole.STANDARD_CODER,
        prompt="Continue fixing",
        repository_path=tmp_path,
        model="GPT-6 Sol",
    )
    with pytest.raises(UnsupportedCapabilityError, match="does not support session resume"):
        await adapter.resume("some-session-id", req)


@pytest.mark.asyncio
async def test_codex_adapter_start_jsonl_output(tmp_path: Path, monkeypatch):
    """CodexAdapter parses streaming JSONL output and distinct error output."""
    jsonl_output = (
        '{"type": "item_created", "item": "file.py"}\n'
        '{"type": "message", "content": "Code generated successfully."}\n'
    )

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=jsonl_output,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = CodexAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.STANDARD_CODER,
        prompt="Implement auth controller",
        repository_path=tmp_path,
        model="gpt-6-sol",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.provider == Provider.OPENAI
    assert result.model == "gpt-6-sol"
    assert "Code generated successfully." in result.text
    assert len(result.raw_events) == 2


@pytest.mark.asyncio
async def test_codex_adapter_cli_error_preservation(tmp_path: Path, monkeypatch):
    """CodexAdapter preserves CLI error in stderr and non-zero exit code."""

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=2,
            stdout="",
            stderr="Flag --invalid is unrecognized",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = CodexAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.LIGHTWEIGHT_CODER,
        prompt="Run check",
        repository_path=tmp_path,
        model="gpt-6-luna",
    )
    result = await adapter.start(req)

    assert result.success is False
    assert result.exit_code == 2
    assert "Flag --invalid is unrecognized" in result.stderr


# ---------------------------------------------------------------------------
# 5. GeminiAdapter Tests
# ---------------------------------------------------------------------------


def test_gemini_adapter_build_args(tmp_path: Path):
    """GeminiAdapter constructs arguments with model, output-format, and read-only."""
    adapter = GeminiAdapter(command="gemini")
    req = AgentRequest(
        role=AgentRole.DEFAULT_REVIEWER,
        prompt="Review diff",
        repository_path=tmp_path,
        model="Gemini 3.8 Flash",
        read_only=True,
    )
    args = adapter.build_args(req)
    assert args == [
        "gemini",
        "--output-format",
        "json",
        "--model",
        "gemini-3.8-flash",
        "--read-only",
        "Review diff",
    ]


def test_gemini_adapter_build_args_resume(tmp_path: Path):
    """GeminiAdapter constructs resume arguments with session_id."""
    adapter = GeminiAdapter(command="gemini")
    req = AgentRequest(
        role=AgentRole.DOCUMENTER,
        prompt="Update documentation",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    args = adapter.build_args(req, session_id="gemini-sess-789")
    assert "--resume" in args
    assert args[args.index("--resume") + 1] == "gemini-sess-789"


@pytest.mark.asyncio
async def test_gemini_adapter_start_candidate_parsing(tmp_path: Path, monkeypatch):
    """GeminiAdapter correctly parses standard Gemini candidate structure."""
    gemini_json = (
        '{"candidates": [{"content": {"parts": [{"text": "LGTM. No issues found."}]}}], '
        '"session_id": "gemini-sess-789"}'
    )

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=gemini_json,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = GeminiAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_REVIEWER,
        prompt="Review changes",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.provider == Provider.GOOGLE
    assert result.model == "gemini-3.8-flash"
    assert result.text == "LGTM. No issues found."
    assert result.session_id == "gemini-sess-789"
    assert len(result.raw_events) == 1


# ---------------------------------------------------------------------------
# 5b. AntigravityAdapter Tests
# ---------------------------------------------------------------------------


def test_antigravity_adapter_capabilities():
    """AntigravityAdapter has no read-only-mode equivalent flag, unlike GeminiAdapter."""
    agy = AntigravityAdapter()
    assert agy.provider == Provider.GOOGLE
    assert agy.capabilities.supports_resume is True
    assert agy.capabilities.supports_read_only_mode is False
    assert agy.capabilities.supports_structured_output is True
    assert agy.capabilities.supports_model_selection is True


def test_antigravity_adapter_build_args_read_only(tmp_path: Path):
    """AntigravityAdapter passes the prompt via -p and uses --sandbox for read-only requests."""
    adapter = AntigravityAdapter(command="agy")
    req = AgentRequest(
        role=AgentRole.DEFAULT_REVIEWER,
        prompt="Review diff",
        repository_path=tmp_path,
        model="Gemini 3.8 Flash",
        read_only=True,
    )
    args = adapter.build_args(req)
    assert args == [
        "agy",
        "-p",
        "Review diff",
        "--output-format",
        "json",
        "--model",
        "gemini-3.8-flash",
        "--sandbox",
    ]


def test_antigravity_adapter_build_args_unattended(tmp_path: Path):
    """A non-read-only AntigravityAdapter request passes --dangerously-skip-permissions."""
    adapter = AntigravityAdapter(command="agy")
    req = AgentRequest(
        role=AgentRole.IMPLEMENTER,
        prompt="Implement the change",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    args = adapter.build_args(req)
    assert "--dangerously-skip-permissions" in args
    assert "--sandbox" not in args


def test_antigravity_adapter_build_args_resume(tmp_path: Path):
    """AntigravityAdapter resumes via --conversation, not GeminiAdapter's --resume."""
    adapter = AntigravityAdapter(command="agy")
    req = AgentRequest(
        role=AgentRole.DOCUMENTER,
        prompt="Update documentation",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    args = adapter.build_args(req, session_id="agy-sess-789")
    assert "--conversation" in args
    assert args[args.index("--conversation") + 1] == "agy-sess-789"
    assert "--resume" not in args


@pytest.mark.asyncio
async def test_antigravity_adapter_start_reuses_gemini_parsing(tmp_path: Path, monkeypatch):
    """AntigravityAdapter reuses GeminiAdapter's schema-agnostic JSON response parsing."""
    agy_json = (
        '{"candidates": [{"content": {"parts": [{"text": "LGTM. No issues found."}]}}], '
        '"session_id": "agy-sess-789"}'
    )

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=agy_json,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = AntigravityAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_REVIEWER,
        prompt="Review changes",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.provider == Provider.GOOGLE
    assert result.text == "LGTM. No issues found."
    assert result.session_id == "agy-sess-789"


# ---------------------------------------------------------------------------
# 6. Adapter Registry Tests
# ---------------------------------------------------------------------------


def test_registry_registration_and_lookup():
    """AgentAdapterRegistry registers and retrieves adapters by enum and string."""
    registry = AgentAdapterRegistry()
    claude = ClaudeAdapter()
    codex = CodexAdapter()

    registry.register(Provider.ANTHROPIC, claude)
    registry.register(Provider.OPENAI, codex)

    assert registry.has(Provider.ANTHROPIC) is True
    assert registry.has("anthropic") is True
    assert registry.has("openai") is True
    assert registry.has(Provider.GOOGLE) is False

    assert registry.get(Provider.ANTHROPIC) is claude
    assert registry.get("anthropic") is claude
    assert registry.get("openai") is codex

    with pytest.raises(AdapterNotFoundError, match="No adapter registered"):
        registry.get(Provider.GOOGLE)

    with pytest.raises(AdapterNotFoundError, match="Unknown provider"):
        registry.get("invalid_provider")


def test_create_default_registry():
    """create_default_registry populates all three standard adapters from config."""
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "cli": {
                "claude": {"command": "custom-claude"},
                "codex": {"command": "custom-codex"},
                "gemini": {"command": "custom-gemini"},
            },
        }
    )
    registry = create_default_registry(config=cfg)

    assert len(registry.registered_providers) == 3
    claude = registry.get(Provider.ANTHROPIC)
    assert isinstance(claude, ClaudeAdapter)
    assert claude.command == "custom-claude"

    codex = registry.get(Provider.OPENAI)
    assert isinstance(codex, CodexAdapter)
    assert codex.command == "custom-codex"

    gemini = registry.get(Provider.GOOGLE)
    assert isinstance(gemini, GeminiAdapter)
    assert gemini.command == "custom-gemini"


def test_create_default_registry_antigravity_dialect():
    """dialect: antigravity registers an AntigravityAdapter for Provider.GOOGLE, not
    GeminiAdapter."""
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "cli": {"gemini": {"command": "agy", "dialect": "antigravity"}},
        }
    )
    registry = create_default_registry(config=cfg)

    google_adapter = registry.get(Provider.GOOGLE)
    assert isinstance(google_adapter, AntigravityAdapter)
    assert google_adapter.command == "agy"


def test_application_contains_agent_registry():
    """Application DI container automatically initializes agent_registry."""
    app = Application()
    assert app.agent_registry is not None
    assert app.agent_registry.has(Provider.ANTHROPIC) is True
    assert app.agent_registry.has(Provider.OPENAI) is True
    assert app.agent_registry.has(Provider.GOOGLE) is True


# ---------------------------------------------------------------------------
# 7. Subprocess Safety & Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_prompt_with_dangerous_characters(tmp_path: Path, monkeypatch):
    """Prompts with quotes, newlines, and semicolon are passed safely as single argument."""
    dangerous_prompt = 'echo "hacked"; rm -rf /; `whoami` & \n\t "test"'
    captured_args = []

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        captured_args.extend(cmd_args)
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout="{}",
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = ClaudeAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.PLANNER,
        prompt=dangerous_prompt,
        repository_path=tmp_path,
        model="sonnet",
    )
    await adapter.start(req)

    # Prompt must be the final argument exactly as provided without string interpolation
    assert captured_args[-1] == dangerous_prompt


@pytest.mark.asyncio
async def test_adapter_timeout_preservation(tmp_path: Path, monkeypatch):
    """Subprocess timeout sets timed_out=True on AgentResult."""

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=-1,
            stdout="",
            stderr="Execution timed out",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=True,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = GeminiAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.REVIEWER,
        prompt="Review large PR",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
        timeout_seconds=0.1,
    )
    result = await adapter.start(req)

    assert result.timed_out is True
    assert result.success is False
    assert result.exit_code == -1


@pytest.mark.asyncio
async def test_claude_adapter_plain_text_with_regex_session_id(tmp_path: Path, monkeypatch):
    """ClaudeAdapter extracts session ID via regex when output is plain text."""
    plain_output = "Response generated.\nSession ID: claude-sess-9988-aabb\nDone."

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=plain_output,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = ClaudeAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_PLANNER,
        prompt="Explain design",
        repository_path=tmp_path,
        model="sonnet",
        extra_args=["--verbose"],
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.session_id == "claude-sess-9988-aabb"
    assert "Response generated" in result.text


@pytest.mark.asyncio
async def test_codex_adapter_plain_text_output(tmp_path: Path, monkeypatch):
    """CodexAdapter handles non-JSON plain text stdout cleanly."""
    plain_output = "Build passed, no errors found."

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=plain_output,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = CodexAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.LIGHTWEIGHT_CODER,
        prompt="Check code",
        repository_path=tmp_path,
        model="gpt-6-luna",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.text == plain_output
    assert result.session_id is None


@pytest.mark.asyncio
async def test_gemini_adapter_plain_text_regex_session_id(tmp_path: Path, monkeypatch):
    """GeminiAdapter extracts session ID via regex when output is plain text."""
    plain_output = "Review complete.\nsession: gemini-sess-5544-ccdd\nAll clear."

    async def mock_run(cmd_args, cwd=None, env=None, timeout=None, input_data=None):
        return ProcessResult(
            command=[str(a) for a in cmd_args],
            exit_code=0,
            stdout=plain_output,
            stderr="",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        )

    executor = ProcessExecutor()
    monkeypatch.setattr(executor, "run", mock_run)

    adapter = GeminiAdapter(executor=executor)
    req = AgentRequest(
        role=AgentRole.DEFAULT_REVIEWER,
        prompt="Review changes",
        repository_path=tmp_path,
        model="gemini-3.8-flash",
    )
    result = await adapter.start(req)

    assert result.success is True
    assert result.session_id == "gemini-sess-5544-ccdd"
    assert "Review complete" in result.text


def test_doctor_check_cli_version_failure(monkeypatch):
    """Doctor check_cli marks check as warning when --version returns non-zero exit code."""
    app = Application()

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/custom-cli")
    monkeypatch.setattr(
        app.executor,
        "run_sync",
        lambda *args, **kwargs: ProcessResult(
            command=["custom-cli", "--version"],
            exit_code=1,
            stdout="",
            stderr="Version flag not supported",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            timed_out=False,
        ),
    )

    item = app.check_cli("Custom", "custom-cli")
    assert item.passed is False
    assert item.critical is False
    assert item.is_warning is True
    assert "failed '--version' check" in item.details
