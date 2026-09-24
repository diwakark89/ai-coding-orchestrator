"""End-to-end integration tests for Application.run_pipeline (Milestone 6: Phases 7-9).

Exercises the full pipeline against a real Git fixture repository: plan -> route -> implement
-> verify -> review -> document -> final approval -> COMPLETED. Only the AI agent CLIs are
scripted (Claude for planning, Codex for implementation, Gemini for review/documentation);
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
from agentflow.ui.approval import ApprovalDecision, FinalApprovalDecision
from agentflow.workflow.states import WorkflowState

PY = sys.executable


class ScriptedAdapter:
    """Fake AgentAdapter running a scripted sequence of (response, worktree side-effect) steps."""

    def __init__(
        self, provider: Provider, steps: list[tuple[AgentResult, Callable[[Path], None] | None]]
    ) -> None:
        self._provider = provider
        self._steps = list(steps)
        self.calls = 0

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=True,
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
        return await self.start(request)


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


def plan_ready_payload() -> str:
    return json.dumps(
        {
            "status": "plan_ready",
            "plan_markdown": "# Plan\n\nObjective: add a health check.\n",
            "task_profile": {
                "stage": "IMPLEMENTATION",
                "technologies": ["python"],
                "affected_layers": ["backend"],
                "estimated_files": 1,
            },
        }
    )


def review_payload(
    status: str = "APPROVED", findings: list[dict[str, object]] | None = None
) -> str:
    return json.dumps({"status": status, "findings": findings or []})


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit_file(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content, encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", f"add {name}"], repo)


def _write_routing_yaml(repo: Path) -> None:
    orch_dir = repo / ".ai-orchestrator"
    orch_dir.mkdir(exist_ok=True)
    (orch_dir / "routing.yaml").write_text(
        "version: 1\n\n"
        "project:\n"
        "  name: demo\n\n"
        "verification:\n"
        "  py:\n"
        "    detect:\n"
        "      - check_ok.py\n"
        "    commands:\n"
        f"      - '{PY} check_ok.py'\n\n"
        "documentation:\n"
        "  enabled: true\n"
        "  candidate_files:\n"
        "    - architecture.md\n",
        encoding="utf-8",
    )


def make_app(registry: AgentAdapterRegistry, tmp_path: Path) -> Application:
    return Application(
        config=GlobalConfig(),
        db_manager=DatabaseManager(tmp_path / "test.db"),
        agent_registry=registry,
    )


def make_registry() -> tuple[
    AgentAdapterRegistry, ScriptedAdapter, ScriptedAdapter, ScriptedAdapter
]:
    def write_feature(worktree: Path) -> None:
        (worktree / "feature.py").write_text("print('hi')\n", encoding="utf-8")

    def write_architecture_doc(worktree: Path) -> None:
        (worktree / "architecture.md").write_text(
            "# Architecture\n\nHealth check added.\n", encoding="utf-8"
        )

    claude = ScriptedAdapter(
        Provider.ANTHROPIC,
        [(make_agent_result(Provider.ANTHROPIC, "sonnet", plan_ready_payload()), None)],
    )
    codex = ScriptedAdapter(
        Provider.OPENAI, [(make_agent_result(Provider.OPENAI, "luna"), write_feature)]
    )
    gemini = ScriptedAdapter(
        Provider.GOOGLE,
        [
            (make_agent_result(Provider.GOOGLE, "flash", review_payload()), None),
            (make_agent_result(Provider.GOOGLE, "flash"), write_architecture_doc),
        ],
    )

    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, claude)
    registry.register(Provider.OPENAI, codex)
    registry.register(Provider.GOOGLE, gemini)
    return registry, claude, codex, gemini


@pytest.mark.asyncio
async def test_full_pipeline_reaches_completed(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Plan, route, implement, verify, review, document, and approve reach COMPLETED end to end."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _commit_file(git_repo, "check_ok.py", "import sys\nsys.exit(0)\n")
    _write_routing_yaml(git_repo)

    registry, claude, codex, gemini = make_registry()
    app_instance = make_app(registry, tmp_path)

    outcome = await app_instance.run_pipeline(
        "Add a health check endpoint",
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.MERGE,
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert outcome.review is not None
    assert outcome.review.state == WorkflowState.REVIEW_APPROVED
    assert outcome.documentation is not None
    assert outcome.documentation.enabled is True
    assert "architecture.md" in outcome.documentation.updated_files
    assert outcome.final_summary is not None
    assert outcome.final_summary_path is not None and outcome.final_summary_path.exists()
    assert "architecture.md" in outcome.final_summary.render_markdown()

    run_id = outcome.implementation_outcome.planning_outcome.run_id
    run = app_instance.db_manager.get_run(run_id)
    assert run is not None
    assert run.state == "COMPLETED"
    assert run.status == "COMPLETED"

    decisions = app_instance.db_manager.list_decisions(run_id)
    assert any(d.question == "final_approval" and d.answer == "merge" for d in decisions)
    merged = [d.answer for d in decisions if d.question == "merged_commit"]
    assert len(merged) == 1

    # One squashed commit on the branch the run started from, with every changed file.
    head = subprocess.run(
        ["git", "log", "-1", "--format=%H%n%s%n%b"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert head.startswith(merged[0])
    assert f"AgentFlow run {run_id}" in head
    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert sorted(committed) == sorted(outcome.final_summary.changed_files)
    implementation = outcome.implementation_outcome.implementation
    assert implementation is not None and implementation.worktree is not None
    assert implementation.worktree.base_branch is not None
    assert not implementation.worktree.path.exists()
    branches = subprocess.run(
        ["git", "branch", "--list", f"agentflow/{run_id}"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert branches.strip() == ""

    assert claude.calls == 1
    assert codex.calls == 1
    assert gemini.calls == 2


@pytest.mark.asyncio
async def test_pipeline_cancelled_at_final_approval(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Cancelling at final approval ends the run CANCELLED even though every gate passed."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _commit_file(git_repo, "check_ok.py", "import sys\nsys.exit(0)\n")
    _write_routing_yaml(git_repo)

    registry, _, _, _ = make_registry()
    app_instance = make_app(registry, tmp_path)

    outcome = await app_instance.run_pipeline(
        "Add a health check endpoint",
        project_path=git_repo,
        final_approval_prompt=lambda: FinalApprovalDecision.CANCEL,
    )

    assert outcome.state == WorkflowState.CANCELLED
    run_id = outcome.implementation_outcome.planning_outcome.run_id
    run = app_instance.db_manager.get_run(run_id)
    assert run is not None
    assert run.state == "CANCELLED"
    assert run.status == "CANCELLED"


@pytest.mark.asyncio
async def test_final_gate_shows_diff_then_merges(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Choosing `diff` shows the worktree diff and re-prompts; `merge` then lands the run."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    _commit_file(git_repo, "check_ok.py", "import sys\nsys.exit(0)\n")
    _write_routing_yaml(git_repo)
    registry, *_ = make_registry()
    app_instance = make_app(registry, tmp_path)
    shown: list[str] = []

    async def fake_show_diff(worktree_path: Path, diff_text: str) -> None:
        shown.append(diff_text)

    monkeypatch.setattr(app_instance, "_show_worktree_diff", fake_show_diff)
    choices = iter([FinalApprovalDecision.DIFF, FinalApprovalDecision.MERGE])

    outcome = await app_instance.run_pipeline(
        "Add a health check endpoint",
        project_path=git_repo,
        final_approval_prompt=lambda: next(choices),
    )

    assert outcome.state == WorkflowState.COMPLETED
    assert len(shown) == 1 and shown[0].strip()
    run_id = outcome.implementation_outcome.planning_outcome.run_id
    answers = [
        d.answer
        for d in app_instance.db_manager.list_decisions(run_id)
        if d.question in ("final_approval", "merged_commit")
    ]
    assert answers[0] == "merge" and len(answers) == 2
