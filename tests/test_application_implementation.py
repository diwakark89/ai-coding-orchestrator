"""End-to-end integration tests for Application.run_implementation (Milestone 5: Phases 5-6).

Exercises the full pipeline against a real Git fixture repository: plan -> route -> implement
inside an isolated worktree -> verify with real subprocess commands -> repair if needed.
Only the AI agent CLIs (Claude for planning, Codex for implementation/repair) are scripted;
everything else (Git, SQLite, subprocess verification) is real.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agentflow.workflow.planning as planning_module
from agentflow.agents.base import AdapterCapabilities, AgentRequest, AgentResult, Provider
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.application import Application
from agentflow.config.models import GlobalConfig
from agentflow.persistence.database import DatabaseManager
from agentflow.ui.approval import ApprovalDecision
from agentflow.workflow.states import WorkflowState
from agentflow.workflow.verification import VerificationStatus

PY = sys.executable


class ScriptedClaudeAdapter:
    """Fake Claude adapter returning a single scripted planning response."""

    def __init__(self, response: AgentResult) -> None:
        self._response = response

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=True,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        return self._response

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        return self._response


class ScriptedCodexAdapter:
    """Fake Codex adapter running a scripted sequence of (response, worktree side-effect)."""

    def __init__(self, steps: list[tuple[AgentResult, Callable[[Path], None] | None]]) -> None:
        self._steps = list(steps)
        self.calls = 0

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=False,
            supports_read_only_mode=False,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    async def start(self, request: AgentRequest) -> AgentResult:
        self.calls += 1
        response, side_effect = self._steps.pop(0)
        if side_effect is not None:
            assert request.working_directory is not None
            side_effect(request.working_directory)
        return response

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        raise NotImplementedError


def make_agent_result(provider: Provider, model: str, text: str = "done") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=provider,
        model=model,
        session_id="sess-1",
        exit_code=0,
        text=text,
        started_at=now,
        completed_at=now,
    )


def plan_ready_payload(estimated_files: int = 1, **flags: object) -> str:
    task_profile = {
        "stage": "IMPLEMENTATION",
        "technologies": ["python"],
        "affected_layers": ["backend"],
        "estimated_files": estimated_files,
        **flags,
    }
    return json.dumps(
        {
            "status": "plan_ready",
            "plan_markdown": "# Plan\n\nObjective: add a feature.\n",
            "task_profile": task_profile,
        }
    )


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit_file(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", f"add {name}"], repo)


def _write_routing_yaml(repo: Path, verification_yaml: str) -> None:
    orch_dir = repo / ".ai-orchestrator"
    orch_dir.mkdir(exist_ok=True)
    (orch_dir / "routing.yaml").write_text(
        f"version: 1\n\nproject:\n  name: demo\n\n{verification_yaml}\n", encoding="utf-8"
    )


def make_app(registry: AgentAdapterRegistry, tmp_path: Path) -> Application:
    return Application(
        config=GlobalConfig(),
        db_manager=DatabaseManager(tmp_path / "test.db"),
        agent_registry=registry,
    )


@pytest.mark.asyncio
async def test_full_pipeline_succeeds_without_repair(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Plan, route, implement, and verify succeed end to end when verification passes first try."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _commit_file(git_repo, "check_ok.py", "import sys\nsys.exit(0)\n")
    _write_routing_yaml(
        git_repo,
        "verification:\n"
        "  py:\n"
        "    detect:\n"
        "      - check_ok.py\n"
        "    commands:\n"
        f"      - '{PY} check_ok.py'\n",
    )

    registry = AgentAdapterRegistry()
    registry.register(
        Provider.ANTHROPIC,
        ScriptedClaudeAdapter(
            make_agent_result(Provider.ANTHROPIC, "sonnet", text=plan_ready_payload())
        ),
    )

    def write_feature(worktree: Path) -> None:
        (worktree / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    codex = ScriptedCodexAdapter(
        [(make_agent_result(Provider.OPENAI, "gpt-5.6-luna"), write_feature)]
    )
    registry.register(Provider.OPENAI, codex)

    app_instance = make_app(registry, tmp_path)
    outcome = await app_instance.run_implementation("Add a feature", project_path=git_repo)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.verification is not None
    assert outcome.verification.status == VerificationStatus.PASSED
    assert outcome.implementation is not None
    assert outcome.implementation.worktree is not None
    assert (outcome.implementation.worktree.path / "feature.py").exists()
    assert not (git_repo / "feature.py").exists()
    assert outcome.verification_path is not None and outcome.verification_path.exists()
    assert outcome.routing_decision is not None
    assert outcome.routing_decision.model == "GPT-5.6 Luna"
    assert codex.calls == 1

    run = app_instance.db_manager.get_run(outcome.planning_outcome.run_id)
    assert run is not None
    assert run.status == "COMPLETED"

    decisions = app_instance.db_manager.list_routing_decisions(outcome.planning_outcome.run_id)
    assert len(decisions) == 1
    verifications = app_instance.db_manager.list_verification_runs(outcome.planning_outcome.run_id)
    assert len(verifications) == 1
    assert verifications[0].exit_code == 0
    events = app_instance.db_manager.list_events(outcome.planning_outcome.run_id)
    event_types = [event.event for event in events]
    assert "ROUTING_SELECTED" in event_types
    assert "AGENT_STARTED" in event_types
    assert "AGENT_COMPLETED" in event_types
    verification_event = next(event for event in events if event.event == "VERIFICATION_COMPLETED")
    assert verification_event.attributes["success"] is True


@pytest.mark.asyncio
async def test_pipeline_repairs_a_failing_verification(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A first verification failure triggers repair; success requires a full re-verification."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    check_script = (
        "import pathlib, sys\n"
        "if (pathlib.Path(__file__).parent / 'status.flag').exists():\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    _commit_file(git_repo, "check_flag.py", check_script)
    _write_routing_yaml(
        git_repo,
        "verification:\n"
        "  py:\n"
        "    detect:\n"
        "      - check_flag.py\n"
        "    commands:\n"
        f"      - '{PY} check_flag.py'\n",
    )

    registry = AgentAdapterRegistry()
    registry.register(
        Provider.ANTHROPIC,
        ScriptedClaudeAdapter(
            make_agent_result(Provider.ANTHROPIC, "sonnet", text=plan_ready_payload())
        ),
    )

    def write_feature(worktree: Path) -> None:
        (worktree / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    def write_status_flag(worktree: Path) -> None:
        (worktree / "status.flag").write_text("ok\n", encoding="utf-8")

    codex = ScriptedCodexAdapter(
        [
            (make_agent_result(Provider.OPENAI, "gpt-5.6-luna"), write_feature),
            (make_agent_result(Provider.OPENAI, "gpt-5.6-luna"), write_status_flag),
        ]
    )
    registry.register(Provider.OPENAI, codex)

    app_instance = make_app(registry, tmp_path)
    outcome = await app_instance.run_implementation("Add a feature", project_path=git_repo)

    assert outcome.state == WorkflowState.VERIFYING
    assert outcome.verification is not None
    assert outcome.verification.status == VerificationStatus.PASSED
    assert outcome.repair is not None
    assert outcome.repair.attempts == 1
    assert codex.calls == 2

    verifications = app_instance.db_manager.list_verification_runs(outcome.planning_outcome.run_id)
    assert len(verifications) == 2
    assert verifications[0].exit_code == 1
    assert verifications[1].exit_code == 0
