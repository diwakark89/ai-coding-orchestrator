"""Post-verification documentation sync (Phase 8).

Documentation is updated only after implementation and review are stable, must describe the
final implementation rather than blindly copy the original proposal, and is itself a writer
stage bound by the same single-writer lock as implementation and review fixes.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.agents.base import AgentRequest, AgentRole
from agentflow.agents.registry import AgentAdapterRegistry
from agentflow.config.models import DocumentationConfig, VerificationGroup
from agentflow.git.lock import WorktreeLock
from agentflow.git.worktree import WorktreeManager
from agentflow.observability.events import record_agent_completed, record_agent_started
from agentflow.persistence.database import DatabaseManager
from agentflow.project.context import ProjectContext
from agentflow.routing.engine import route
from agentflow.routing.rules import ModelsConfig, RoutingRulesConfig
from agentflow.task.profile import Stage, TaskProfile
from agentflow.ui.console import ConsoleUI
from agentflow.workflow.review import ReviewFinding
from agentflow.workflow.verification import VerificationResult, VerificationRunnerLike


@dataclass
class DocumentationOutcome:
    """Result of the documentation workflow: whether it ran, and what it changed."""

    enabled: bool
    updated_files: list[str] = field(default_factory=list)
    summary_path: Path | None = None
    verification_result: VerificationResult | None = None
    blocked: bool = False
    blocker_reason: str | None = None


def _build_documentation_prompt(
    plan_markdown: str,
    task_profile: TaskProfile,
    diff_text: str,
    verification_result: VerificationResult,
    review_findings: list[ReviewFinding],
    candidate_files: list[str],
) -> str:
    findings_text = (
        "\n".join(f"- [{f.severity.value}] {f.file}: {f.problem}" for f in review_findings)
        or "(none)"
    )
    files_text = "\n".join(f"- {f}" for f in candidate_files)
    return (
        "You are AgentFlow's documentation agent, operating inside an isolated Git worktree.\n\n"
        f"Candidate documentation files (only touch these):\n{files_text}\n\n"
        f"Approved plan:\n{plan_markdown}\n\n"
        f"TaskProfile:\n{task_profile.model_dump_json(indent=2)}\n\n"
        "Final diff (this is the ACTUAL implementation to describe -- not the original "
        f"proposal, which may have changed during implementation/review):\n{diff_text[:20000]}\n\n"
        f"Verification status: {verification_result.status.value}\n\n"
        f"Final review findings:\n{findings_text}\n\n"
        "Update only the candidate documentation files listed above to accurately describe "
        "the final implementation. If a candidate file does not yet exist, you may create it. "
        "Do not modify source code or any file outside the candidate list."
    )


class DocumentationWorkflow:
    """Syncs project documentation to the final implementation, once, under the writer lock."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        agent_registry: AgentAdapterRegistry,
        worktree_manager: WorktreeManager,
        verification_runner: VerificationRunnerLike,
        models_config: ModelsConfig,
        routing_rules: RoutingRulesConfig,
        console_ui: ConsoleUI | None = None,
    ) -> None:
        self.db_manager = db_manager
        self.agent_registry = agent_registry
        self.worktree_manager = worktree_manager
        self.verification_runner = verification_runner
        self.models_config = models_config
        self.routing_rules = routing_rules
        self.ui = console_ui or ConsoleUI()

    async def run(
        self,
        run_id: str,
        worktree_path: Path,
        project_context: ProjectContext,
        plan_markdown: str,
        task_profile: TaskProfile,
        diff_text: str,
        verification_result: VerificationResult,
        review_findings: list[ReviewFinding],
        documentation_config: DocumentationConfig | None,
        verification_config: dict[str, VerificationGroup] | None,
    ) -> DocumentationOutcome:
        """Update configured candidate documentation files to describe the final implementation."""
        config = documentation_config or DocumentationConfig(enabled=True, candidate_files=[])
        if not config.enabled or not config.candidate_files:
            return DocumentationOutcome(enabled=False)

        doc_profile = task_profile.model_copy(update={"stage": Stage.DOCUMENTATION})
        decision = route(doc_profile, models=self.models_config, routing_rules=self.routing_rules)
        self.db_manager.record_routing_decision(
            str(uuid.uuid4()),
            run_id,
            decision.stage.value,
            doc_profile.model_dump_json(),
            decision.complexity_score,
            decision.complexity.value,
            decision.matched_rule,
            decision.provider.value,
            decision.model,
            decision.reason,
        )

        lock = WorktreeLock(worktree_path)
        lock.acquire()
        try:
            adapter = self.agent_registry.get(decision.provider)
            request = AgentRequest(
                role=AgentRole.DOCUMENTER,
                prompt=_build_documentation_prompt(
                    plan_markdown,
                    task_profile,
                    diff_text,
                    verification_result,
                    review_findings,
                    config.candidate_files,
                ),
                repository_path=project_context.root_path,
                working_directory=worktree_path,
                model=decision.model,
                read_only=False,
            )
            record_agent_started(
                self.db_manager, run_id, "DOCUMENTING", adapter.provider.value, decision.model
            )
            result = await adapter.start(request)
        finally:
            lock.release()

        self.db_manager.record_agent_session(
            str(uuid.uuid4()),
            run_id,
            "DOCUMENTING",
            adapter.provider.value,
            decision.model,
            result.session_id,
        )
        record_agent_completed(self.db_manager, run_id, "DOCUMENTING", result)

        if not result.success:
            detail = result.stderr.strip() or result.text.strip() or "no output"
            reason = f"Documentation agent exited with code {result.exit_code}: {detail}"
            return DocumentationOutcome(enabled=True, blocked=True, blocker_reason=reason)

        diff = await self.worktree_manager.capture_changes(worktree_path)

        verification_after: VerificationResult | None = None
        if verification_config:
            verification_after = await self.verification_runner.run(
                run_id, worktree_path, verification_config
            )
            if not verification_after.success:
                reason = "Documentation changes broke verification."
                return DocumentationOutcome(
                    enabled=True,
                    updated_files=diff.changed_files,
                    blocked=True,
                    blocker_reason=reason,
                    verification_result=verification_after,
                )

        summary_path = self._write_summary(project_context, run_id, result.text, diff.changed_files)
        return DocumentationOutcome(
            enabled=True,
            updated_files=diff.changed_files,
            summary_path=summary_path,
            verification_result=verification_after,
        )

    def _write_summary(
        self,
        project_context: ProjectContext,
        run_id: str,
        agent_text: str,
        updated_files: list[str],
    ) -> Path:
        """Persist documentation-summary.md describing what documentation changed."""
        run_dir = project_context.root_path / ".ai-orchestrator" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "documentation-summary.md"
        changed = "\n".join(f"- {f}" for f in updated_files) or "- (none)"
        content = (
            f"# Documentation Summary\n\n## Agent Output\n\n{agent_text}\n\n"
            f"## Updated Files\n\n{changed}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path
