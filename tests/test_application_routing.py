"""Integration tests for Application.run_routing (Phase 4 planning -> routing wiring)."""

import json
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


class ScriptedAdapter:
    """Fake AgentAdapter returning a pre-scripted sequence of AgentResults."""

    def __init__(self, responses: list[AgentResult]) -> None:
        self._responses = list(responses)

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
        return self._responses.pop(0)

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        return self._responses.pop(0)


def make_result(text: str, session_id: str = "sess-1") -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        provider=Provider.ANTHROPIC,
        model="sonnet",
        session_id=session_id,
        exit_code=0,
        text=text,
        started_at=now,
        completed_at=now,
    )


@pytest.mark.asyncio
async def test_run_routing_persists_decision_json_and_db_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """run_routing classifies a task, computes routing, and persists both artifact and DB record."""
    monkeypatch.setattr(
        planning_module, "ask_plan_approval", lambda console_instance=None: ApprovalDecision.APPROVE
    )
    (tmp_path / ".git").mkdir()

    payload = json.dumps(
        {
            "status": "plan_ready",
            "plan_markdown": "# Plan\n\nObjective: add a secured endpoint.\n",
            "task_profile": {
                "stage": "IMPLEMENTATION",
                "technologies": ["python"],
                "affected_layers": ["backend"],
                "estimated_files": 2,
                "authorization": True,
            },
        }
    )
    adapter = ScriptedAdapter([make_result(payload)])
    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, adapter)

    app_instance = Application(
        config=GlobalConfig(),
        db_manager=DatabaseManager(":memory:"),
        agent_registry=registry,
    )

    outcome = await app_instance.run_routing("Add a secured endpoint", project_path=tmp_path)

    assert outcome.planning_outcome.state == WorkflowState.TASK_CLASSIFIED
    assert outcome.decision is not None
    assert outcome.decision.model == "GPT-5.6 Terra"
    assert outcome.decision.matched_rule == "implementation.force-standard"
    assert outcome.decision_path is not None
    assert outcome.decision_path.exists()

    persisted = json.loads(outcome.decision_path.read_text(encoding="utf-8"))
    assert persisted["model"] == "GPT-5.6 Terra"

    records = app_instance.db_manager.list_routing_decisions(outcome.planning_outcome.run_id)
    assert len(records) == 1
    assert records[0].model == "GPT-5.6 Terra"


@pytest.mark.asyncio
async def test_run_routing_returns_no_decision_when_planning_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """run_routing does not attempt to route when planning does not reach TASK_CLASSIFIED."""
    (tmp_path / ".git").mkdir()
    blocked_payload = json.dumps({"status": "blocked", "blocker_reason": "Task is impossible."})
    adapter = ScriptedAdapter([make_result(blocked_payload)])
    registry = AgentAdapterRegistry()
    registry.register(Provider.ANTHROPIC, adapter)

    app_instance = Application(
        config=GlobalConfig(),
        db_manager=DatabaseManager(":memory:"),
        agent_registry=registry,
    )

    outcome = await app_instance.run_routing("Do the impossible", project_path=tmp_path)

    assert outcome.planning_outcome.state == WorkflowState.BLOCKED
    assert outcome.decision is None
    assert outcome.decision_path is None
