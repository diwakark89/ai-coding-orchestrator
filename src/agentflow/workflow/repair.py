"""Deterministic failure classification and bounded repair/escalation loop.

Verification failures are classified with fixed keyword/command patterns (never AI
classification), routed to the cheapest capable repair tier, and escalated
(Luna -> Sol -> Sonnet) only after each tier repeatedly fails. Success is only ever
declared after a full re-run of the configured verification sequence.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agentflow.agents.base import AgentRequest, AgentRole
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import LimitsConfig, VerificationGroup
from agentflow.git.lock import WorktreeLock
from agentflow.observability.events import record_agent_completed, record_agent_started
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.rules import ModelsConfig
from agentflow.task.profile import TaskProfile
from agentflow.ui.console import ConsoleUI
from agentflow.workflow.states import WorkflowState, transition_run_state
from agentflow.workflow.verification import (
    CommandResult,
    VerificationResult,
    VerificationRunnerLike,
)


class FailureCategory(str, Enum):
    """Deterministic classification of a verification failure."""

    LINT = "lint"
    FORMATTING = "formatting"
    MISSING_IMPORT = "missing_import"
    TYPE_MISMATCH = "type_mismatch"
    COMPILATION = "compilation"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    TRANSACTION = "transaction"
    DATABASE = "database"
    UNKNOWN = "unknown"


# Appropriate for the lightweight tier (Luna) per phased-implementation-plan.md Phase 6 Step 6.
SIMPLE_CATEGORIES = frozenset(
    {
        FailureCategory.LINT,
        FailureCategory.FORMATTING,
        FailureCategory.MISSING_IMPORT,
        FailureCategory.TYPE_MISMATCH,
        FailureCategory.COMPILATION,
    }
)

_LINT_TOOLS = ("ruff", "eslint", "flake8", "pylint", "lint")
_FORMAT_TOOLS = ("black", "prettier", "gofmt", "format")
_MISSING_IMPORT_KEYWORDS = (
    "modulenotfounderror",
    "importerror",
    "cannot find symbol",
    "unresolved import",
    "no module named",
)
_TYPE_MISMATCH_KEYWORDS = ("typeerror", "type mismatch", "incompatible type")
_COMPILATION_KEYWORDS = ("syntaxerror", "compilation error", "compile error", "error: expected")
_TRANSACTION_KEYWORDS = ("deadlock", "transaction", "rollback")
_DATABASE_KEYWORDS = ("database", "sqlstate", "connection refused", "sql error")


def classify_failure(command: str, stdout: str, stderr: str) -> FailureCategory:
    """Classify a failing verification command using fixed deterministic patterns."""
    text = f"{stdout}\n{stderr}".lower()
    cmd_lower = command.lower()

    if any(tool in cmd_lower for tool in _LINT_TOOLS):
        return FailureCategory.LINT
    if any(tool in cmd_lower for tool in _FORMAT_TOOLS):
        return FailureCategory.FORMATTING
    if any(kw in text for kw in _MISSING_IMPORT_KEYWORDS):
        return FailureCategory.MISSING_IMPORT
    if any(kw in text for kw in _TYPE_MISMATCH_KEYWORDS):
        return FailureCategory.TYPE_MISMATCH
    if any(kw in text for kw in _COMPILATION_KEYWORDS):
        return FailureCategory.COMPILATION
    if any(kw in text for kw in _TRANSACTION_KEYWORDS):
        return FailureCategory.TRANSACTION
    if any(kw in text for kw in _DATABASE_KEYWORDS):
        return FailureCategory.DATABASE
    if "integration" in cmd_lower:
        return FailureCategory.INTEGRATION_TEST
    if "test" in cmd_lower or "assertionerror" in text:
        return FailureCategory.UNIT_TEST
    return FailureCategory.UNKNOWN


class RepairTier(str, Enum):
    """Model tier used for a repair attempt."""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    ESCALATION = "escalation"


_TIER_ALIASES: dict[RepairTier, str] = {
    RepairTier.LIGHTWEIGHT: "implementation.lightweight",
    RepairTier.STANDARD: "implementation.standard",
    RepairTier.ESCALATION: "implementation.escalation",
}
_TIER_ROLES: dict[RepairTier, AgentRole] = {
    RepairTier.LIGHTWEIGHT: AgentRole.LIGHTWEIGHT_CODER,
    RepairTier.STANDARD: AgentRole.STANDARD_CODER,
    RepairTier.ESCALATION: AgentRole.IMPLEMENTATION_ESCALATION,
}


@dataclass
class RepairOutcome:
    """Terminal result of the repair loop: either verification now passes, or it is blocked."""

    state: WorkflowState
    verification_result: VerificationResult
    attempts: int
    final_tier: RepairTier | None = None
    blocker_reason: str | None = None


def _build_repair_prompt(
    plan_markdown: str,
    task_profile: TaskProfile,
    failing: CommandResult,
    category: FailureCategory,
) -> str:
    """Build a repair prompt describing the failure and instructing a scoped, minimal fix."""
    return (
        "You are AgentFlow's repair agent, operating inside an isolated Git worktree that "
        "already contains a prior implementation attempt.\n\n"
        f"Approved plan:\n{plan_markdown}\n\n"
        f"TaskProfile:\n{task_profile.model_dump_json(indent=2)}\n\n"
        f"Verification command failed: {failing.command}\n"
        f"Classified failure category: {category.value}\n\n"
        f"stdout:\n{failing.stdout[-4000:]}\n\n"
        f"stderr:\n{failing.stderr[-4000:]}\n\n"
        "Fix only what is necessary to make verification pass. Do not redesign the approved "
        "architecture. If the failure indicates an architectural problem rather than an "
        "implementation bug, stop and report the blocker instead of attempting a workaround."
    )


class RepairWorkflow:
    """Bounded verify/repair/re-verify loop, escalating Luna -> Sol -> Sonnet."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        agent_registry: AgentAdapterRegistry,
        verification_runner: VerificationRunnerLike,
        models_config: ModelsConfig,
        limits: LimitsConfig | None = None,
        console_ui: ConsoleUI | None = None,
        retired_models: Mapping[str, str] | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.agent_registry = agent_registry
        self.verification_runner = verification_runner
        self.models_config = models_config
        self.limits = limits or LimitsConfig()
        self.ui = console_ui or ConsoleUI()
        self.retired_models = retired_models

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        verification_config: dict[str, VerificationGroup] | None,
        initial_result: VerificationResult,
    ) -> RepairOutcome:
        """Repair and re-verify until verification passes or bounded escalation is exhausted."""
        result = initial_result
        lightweight_failures = 0
        standard_failures = 0
        escalation_attempted = False
        attempts = 0
        tier = RepairTier.LIGHTWEIGHT

        while not result.success:
            attempts += 1
            failing = result.first_failure
            if failing is None:
                break  # ERROR/TIMED_OUT with no isolated failing command; nothing to classify.
            category = classify_failure(failing.command, failing.stdout, failing.stderr)

            if (
                category not in SIMPLE_CATEGORIES
                or lightweight_failures >= self.limits.lightweight_verification_failures
            ):
                if standard_failures >= self.limits.standard_failures:
                    if escalation_attempted:
                        reason = (
                            "Repair escalated to Claude Sonnet 5 but verification still fails; "
                            "suspected architecture blocker."
                        )
                        transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
                        return RepairOutcome(
                            state=WorkflowState.BLOCKED,
                            verification_result=result,
                            attempts=attempts,
                            final_tier=RepairTier.ESCALATION,
                            blocker_reason=reason,
                        )
                    tier = RepairTier.ESCALATION
                    escalation_attempted = True
                else:
                    tier = RepairTier.STANDARD
            else:
                tier = RepairTier.LIGHTWEIGHT

            transition_run_state(
                self.db_manager,
                run_id,
                WorkflowState.REPAIRING,
                f"Repair attempt {attempts} ({tier.value}) for category '{category.value}'",
            )

            await self._attempt_repair(
                run_id,
                worktree_path,
                project_context,
                plan_markdown,
                task_profile,
                failing,
                category,
                tier,
            )

            transition_run_state(
                self.db_manager,
                run_id,
                WorkflowState.VERIFYING,
                f"Re-verifying after {tier.value} repair",
            )
            result = await self.verification_runner.run(
                run_id,
                worktree_path,
                verification_config,
                log_dir=self._verification_log_dir(project_context, run_id),
            )

            if not result.success:
                if tier == RepairTier.LIGHTWEIGHT:
                    lightweight_failures += 1
                elif tier == RepairTier.STANDARD:
                    standard_failures += 1

        return RepairOutcome(
            state=WorkflowState.VERIFYING,
            verification_result=result,
            attempts=attempts,
            final_tier=tier,
        )

    async def _attempt_repair(
        self,
        run_id: str,
        worktree_path: Path,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        failing: CommandResult,
        category: FailureCategory,
        tier: RepairTier,
    ) -> None:
        """Invoke the tier's adapter with a lock held, and record the resulting agent session."""
        ref = self.models_config.resolve(_TIER_ALIASES[tier], retired=self.retired_models)
        lock = WorktreeLock(worktree_path)
        lock.acquire()
        try:
            adapter = self.agent_registry.get(ref.provider_enum)
            request = AgentRequest(
                role=_TIER_ROLES[tier],
                prompt=_build_repair_prompt(plan_markdown, task_profile, failing, category),
                repository_path=project_context.root_path,
                working_directory=worktree_path,
                model=ref.model,
                read_only=False,
            )
            record_agent_started(
                self.db_manager,
                run_id,
                WorkflowState.REPAIRING.value,
                ref.provider_enum.value,
                ref.model,
            )
            async with self.ui.animate_stage(f"Repairing: {ref.model}"):
                result = await adapter.start(request)
        finally:
            lock.release()

        self.db_manager.record_agent_session(
            str(uuid.uuid4()),
            run_id,
            WorkflowState.REPAIRING.value,
            ref.provider_enum.value,
            ref.model,
            result.session_id,
        )
        record_agent_completed(self.db_manager, run_id, WorkflowState.REPAIRING.value, result)

    @staticmethod
    def _verification_log_dir(project_context: ProjectContext, run_id: str) -> Path:
        return project_context.root_path / ".ai-orchestrator" / "runs" / run_id / "verification"
