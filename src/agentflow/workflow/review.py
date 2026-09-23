"""Independent, read-only AI review with a bounded mandatory-fix cycle.

Reviewer selection is deterministic (TDD §21 / routing.review via the Phase 4 routing engine)
— never chosen by the reviewer itself. CRITICAL/HIGH findings are always mandatory, MEDIUM is
configurable, LOW is informational only; this decision is computed by the orchestrator from
severities, never trusted from the AI's own self-reported status. The reviewer never modifies
files directly: mandatory fixes are sent to the original implementation worker, and every fix
requires a full verification re-run before another review pass.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentflow.agents.base import AgentRequest, AgentRole
from agentflow.agents.parser import parse_model_into_schema
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import LimitsConfig, VerificationGroup
from agentflow.errors import ReviewBlockedError, StructuredParsingError
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeManager
from agentflow.observability.events import (
    record_agent_completed,
    record_agent_started,
    record_review_completed,
)
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.decision import RoutingDecision
from agentflow.routing.engine import route
from agentflow.routing.rules import ModelsConfig, RoleOverride, RoutingRulesConfig
from agentflow.task.profile import Stage, TaskProfile
from agentflow.ui.console import ConsoleUI
from agentflow.workflow.states import WorkflowState, transition_run_state
from agentflow.workflow.verification import VerificationResult, VerificationRunnerLike


class ReviewSeverity(str, Enum):
    """Severity of a single review finding."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


MANDATORY_SEVERITIES = frozenset({ReviewSeverity.CRITICAL, ReviewSeverity.HIGH})

_REVIEW_ROLE_BY_ALIAS: dict[str, AgentRole] = {
    "review.default": AgentRole.DEFAULT_REVIEWER,
    "review.deep": AgentRole.DEEP_REVIEWER,
    "review.architecture": AgentRole.ARCHITECTURE_REVIEWER,
}
_IMPLEMENTATION_ROLE_BY_ALIAS: dict[str, AgentRole] = {
    "implementation.lightweight": AgentRole.LIGHTWEIGHT_CODER,
    "implementation.standard": AgentRole.STANDARD_CODER,
    "implementation.escalation": AgentRole.IMPLEMENTATION_ESCALATION,
}


class ReviewFinding(BaseModel):
    """A single structured review finding (TDD §34)."""

    model_config = ConfigDict(extra="ignore")

    severity: ReviewSeverity
    category: str
    file: str
    line: int | None = None
    problem: str
    recommendation: str


class ReviewReport(BaseModel):
    """Structured wire-protocol response expected from the reviewer CLI."""

    model_config = ConfigDict(extra="forbid")

    status: str
    findings: list[ReviewFinding] = Field(default_factory=list)


@dataclass
class ReviewOutcome:
    """Terminal result of the review workflow."""

    state: WorkflowState
    findings: list[ReviewFinding] = field(default_factory=list)
    cycles: int = 0
    reviewer_model: str | None = None
    review_path: Path | None = None
    findings_path: Path | None = None
    verification_result: VerificationResult | None = None
    blocker_reason: str | None = None


def mandatory_findings(
    findings: list[ReviewFinding], medium_is_mandatory: bool
) -> list[ReviewFinding]:
    """Findings the orchestrator (never the AI itself) determines must be fixed."""
    severities = set(MANDATORY_SEVERITIES)
    if medium_is_mandatory:
        severities.add(ReviewSeverity.MEDIUM)
    return [f for f in findings if f.severity in severities]


def _build_review_prompt(
    plan_markdown: str,
    task_profile: TaskProfile,
    diff_text: str,
    verification_result: VerificationResult,
) -> str:
    verification_summary = f"Status: {verification_result.status.value}\n" + "\n".join(
        f"- {c.command}: exit_code={c.exit_code}" for c in verification_result.command_results
    )
    return (
        "You are AgentFlow's independent code reviewer, operating strictly read-only. Do not "
        "modify any files.\n\n"
        f"Approved plan:\n{plan_markdown}\n\n"
        f"TaskProfile:\n{task_profile.model_dump_json(indent=2)}\n\n"
        f"Final diff:\n{diff_text[:20000]}\n\n"
        f"Verification results:\n{verification_summary}\n\n"
        "Respond with a single JSON object and nothing else, matching:\n"
        '{"status": "APPROVED" | "CHANGES_REQUIRED", "findings": ['
        '{"severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW", "category": "...", '
        '"file": "...", "line": <int or null>, "problem": "...", "recommendation": "..."}'
        "]}\n\n"
        "Only report genuine problems; an empty findings list with status APPROVED is the "
        "correct response for code with no issues."
    )


def _build_retry_prompt(error: str) -> str:
    return (
        "Your previous response could not be parsed as valid JSON matching the required "
        f"schema. Parsing error: {error}\n\n"
        "Respond again with ONLY a single valid JSON object in the shape described "
        "previously. Do not include any explanation outside the JSON."
    )


def _build_review_fix_prompt(
    plan_markdown: str, task_profile: TaskProfile, findings: list[ReviewFinding]
) -> str:
    listed = "\n\n".join(
        f"- [{f.severity.value}] {f.file}"
        + (f":{f.line}" if f.line is not None else "")
        + f" ({f.category})\n  Problem: {f.problem}\n  Recommendation: {f.recommendation}"
        for f in findings
    )
    return (
        "You are AgentFlow's implementation agent, fixing mandatory issues raised by an "
        "independent code review.\n\n"
        f"Approved plan:\n{plan_markdown}\n\n"
        f"TaskProfile:\n{task_profile.model_dump_json(indent=2)}\n\n"
        f"Mandatory findings to resolve:\n{listed}\n\n"
        "Fix only these findings. Do not redesign the approved architecture. If a finding "
        "indicates an architectural problem you cannot resolve without redesigning the "
        "approach, stop and report the blocker instead of attempting a workaround."
    )


class ReviewWorkflow:
    """Selects a reviewer deterministically, reviews, and drives a bounded mandatory-fix loop."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        agent_registry: AgentAdapterRegistry,
        verification_runner: VerificationRunnerLike,
        worktree_manager: WorktreeManager,
        models_config: ModelsConfig,
        routing_rules: RoutingRulesConfig,
        limits: LimitsConfig | None = None,
        console_ui: ConsoleUI | None = None,
        max_malformed_retries: int = 2,
        role_override: RoleOverride | None = None,
        retired_models: Mapping[str, str] | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.agent_registry = agent_registry
        self.verification_runner = verification_runner
        self.worktree_manager = worktree_manager
        self.models_config = models_config
        self.routing_rules = routing_rules
        self.limits = limits or LimitsConfig()
        self.ui = console_ui or ConsoleUI()
        self.max_malformed_retries = max_malformed_retries
        self.role_override = role_override
        self.retired_models = retired_models

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        verification_config: dict[str, VerificationGroup] | None,
        implementation_decision: RoutingDecision,
        verification_result: VerificationResult,
    ) -> ReviewOutcome:
        """Review the current diff, resolving mandatory findings within a bounded fix budget."""
        current_verification = verification_result
        fix_attempts = 0
        cycle = 0
        review_profile = task_profile.model_copy(update={"stage": Stage.REVIEW})

        while True:
            cycle += 1
            decision = route(
                review_profile,
                models=self.models_config,
                routing_rules=self.routing_rules,
                user_override=(
                    self.role_override.for_stage(Stage.REVIEW) if self.role_override else None
                ),
                retired_models=self.retired_models,
            )
            self.db_manager.record_routing_decision(
                str(uuid.uuid4()),
                run_id,
                decision.stage.value,
                review_profile.model_dump_json(),
                decision.complexity_score,
                decision.complexity.value,
                decision.matched_rule,
                decision.provider.value,
                decision.model,
                decision.reason,
            )

            diff = await self.worktree_manager.capture_changes(worktree_path)
            try:
                report = await self._review_once(
                    run_id,
                    decision,
                    project_context,
                    worktree_path,
                    plan_markdown,
                    task_profile,
                    diff.diff_text,
                    current_verification,
                )
            except ReviewBlockedError as e:
                transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, str(e))
                return ReviewOutcome(
                    state=WorkflowState.BLOCKED,
                    cycles=cycle,
                    blocker_reason=str(e),
                    verification_result=current_verification,
                )

            mandatory = mandatory_findings(
                report.findings, self.routing_rules.review.medium_is_mandatory
            )
            review_path, findings_path = self._write_review_artifacts(
                project_context, run_id, report, mandatory
            )
            severity_counts: dict[str, int] = {}
            for finding in report.findings:
                severity = finding.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            record_review_completed(
                self.db_manager,
                run_id,
                decision.provider.value,
                decision.model,
                severity_counts,
            )

            if not mandatory:
                transition_run_state(
                    self.db_manager, run_id, WorkflowState.REVIEW_APPROVED, "No mandatory findings"
                )
                return ReviewOutcome(
                    state=WorkflowState.REVIEW_APPROVED,
                    findings=report.findings,
                    cycles=cycle,
                    reviewer_model=decision.model,
                    review_path=review_path,
                    findings_path=findings_path,
                    verification_result=current_verification,
                )

            if fix_attempts >= self.limits.review_fix_cycles:
                reason = (
                    f"Review did not converge within {self.limits.review_fix_cycles} "
                    "fix cycle(s); mandatory findings remain."
                )
                transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
                return ReviewOutcome(
                    state=WorkflowState.BLOCKED,
                    findings=report.findings,
                    cycles=cycle,
                    reviewer_model=decision.model,
                    review_path=review_path,
                    findings_path=findings_path,
                    blocker_reason=reason,
                    verification_result=current_verification,
                )

            transition_run_state(
                self.db_manager,
                run_id,
                WorkflowState.REVIEW_FIXING,
                f"{len(mandatory)} mandatory finding(s)",
            )
            try:
                await self._apply_fix(
                    run_id,
                    worktree_path,
                    project_context,
                    plan_markdown,
                    task_profile,
                    mandatory,
                    implementation_decision,
                )
            except ReviewBlockedError as e:
                transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, str(e))
                return ReviewOutcome(
                    state=WorkflowState.BLOCKED,
                    findings=report.findings,
                    cycles=cycle,
                    blocker_reason=str(e),
                    verification_result=current_verification,
                )
            fix_attempts += 1

            transition_run_state(
                self.db_manager, run_id, WorkflowState.VERIFYING, "Re-verifying after review fix"
            )
            current_verification = await self.verification_runner.run(
                run_id, worktree_path, verification_config
            )
            if not current_verification.success:
                reason = "Review fix broke verification; a full re-verification did not pass."
                transition_run_state(self.db_manager, run_id, WorkflowState.BLOCKED, reason)
                return ReviewOutcome(
                    state=WorkflowState.BLOCKED,
                    findings=report.findings,
                    cycles=cycle,
                    blocker_reason=reason,
                    verification_result=current_verification,
                )

            transition_run_state(
                self.db_manager, run_id, WorkflowState.REVIEWING, "Re-reviewing after fix"
            )

    async def _review_once(
        self,
        run_id: str,
        decision: RoutingDecision,
        project_context: ProjectContext,
        worktree_path: Path,
        plan_markdown: str,
        task_profile: TaskProfile,
        diff_text: str,
        verification_result: VerificationResult,
    ) -> ReviewReport:
        adapter = self.agent_registry.get(decision.provider)
        role = _REVIEW_ROLE_BY_ALIAS.get(decision.role, AgentRole.REVIEWER)
        prompt = _build_review_prompt(plan_markdown, task_profile, diff_text, verification_result)

        last_error = ""
        for attempt in range(self.max_malformed_retries + 1):
            request = AgentRequest(
                role=role,
                prompt=prompt,
                repository_path=project_context.root_path,
                working_directory=worktree_path,
                model=decision.model,
                read_only=True,
            )
            record_agent_started(
                self.db_manager,
                run_id,
                WorkflowState.REVIEWING.value,
                adapter.provider.value,
                decision.model,
            )
            result = await adapter.start(request)
            self.db_manager.record_agent_session(
                str(uuid.uuid4()),
                run_id,
                WorkflowState.REVIEWING.value,
                adapter.provider.value,
                decision.model,
                result.session_id,
            )
            record_agent_completed(self.db_manager, run_id, WorkflowState.REVIEWING.value, result)
            if not result.success:
                detail = result.stderr.strip() or result.text.strip() or "no output"
                raise ReviewBlockedError(f"Reviewer exited with code {result.exit_code}: {detail}")

            try:
                return parse_model_into_schema(result.text, ReviewReport)
            except StructuredParsingError as e:
                last_error = str(e)
                if attempt >= self.max_malformed_retries:
                    break
                prompt = _build_retry_prompt(last_error)

        raise ReviewBlockedError(
            f"Reviewer returned malformed structured output after "
            f"{self.max_malformed_retries + 1} attempt(s): {last_error}"
        )

    async def _apply_fix(
        self,
        run_id: str,
        worktree_path: Path,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        mandatory: list[ReviewFinding],
        implementation_decision: RoutingDecision,
    ) -> None:
        lock = WorktreeLock(worktree_path)
        lock.acquire()
        try:
            adapter = self.agent_registry.get(implementation_decision.provider)
            role = _IMPLEMENTATION_ROLE_BY_ALIAS.get(
                implementation_decision.role, AgentRole.IMPLEMENTER
            )
            request = AgentRequest(
                role=role,
                prompt=_build_review_fix_prompt(plan_markdown, task_profile, mandatory),
                repository_path=project_context.root_path,
                working_directory=worktree_path,
                model=implementation_decision.model,
                read_only=False,
            )
            record_agent_started(
                self.db_manager,
                run_id,
                WorkflowState.REVIEW_FIXING.value,
                implementation_decision.provider.value,
                implementation_decision.model,
            )
            result = await adapter.start(request)
        finally:
            lock.release()

        self.db_manager.record_agent_session(
            str(uuid.uuid4()),
            run_id,
            WorkflowState.REVIEW_FIXING.value,
            implementation_decision.provider.value,
            implementation_decision.model,
            result.session_id,
        )
        record_agent_completed(self.db_manager, run_id, WorkflowState.REVIEW_FIXING.value, result)
        if not result.success:
            detail = result.stderr.strip() or result.text.strip() or "no output"
            raise ReviewBlockedError(
                f"Review-fix agent exited with code {result.exit_code}: {detail}"
            )

    def _write_review_artifacts(
        self,
        project_context: ProjectContext,
        run_id: str,
        report: ReviewReport,
        mandatory: list[ReviewFinding],
    ) -> tuple[Path, Path]:
        """Persist review.md (human) and review-findings.json (machine) for a run."""
        run_dir = project_context.root_path / ".ai-orchestrator" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        lines = [f"# Review\n\nStatus: {report.status}\n"]
        if not report.findings:
            lines.append("No findings.\n")
        for f in report.findings:
            marker = "MANDATORY" if f in mandatory else "informational"
            location = f"{f.file}:{f.line}" if f.line is not None else f.file
            lines.append(
                f"## [{f.severity.value}] {location} ({marker})\n\n"
                f"**Category:** {f.category}\n\n"
                f"**Problem:** {f.problem}\n\n"
                f"**Recommendation:** {f.recommendation}\n"
            )
        review_path = run_dir / "review.md"
        review_path.write_text("\n".join(lines), encoding="utf-8")

        findings_path = run_dir / "review-findings.json"
        findings_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return review_path, findings_path
