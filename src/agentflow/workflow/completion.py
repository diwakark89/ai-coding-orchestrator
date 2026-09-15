"""Final run summary and artifact (Phase 9).

A complete run must reach READY_FOR_APPROVAL only once plan approval, implementation,
deterministic verification, mandatory review resolution, and required documentation have all
succeeded. From there, only a human decides completion -- V1 never auto-pushes, auto-merges,
or deploys.
"""

from dataclasses import dataclass, field
from pathlib import Path

from agentflow.persistence.models import RoutingDecisionRecord
from agentflow.workflow.review import ReviewFinding
from agentflow.workflow.verification import VerificationResult


@dataclass
class FinalSummary:
    """Everything a human needs to decide whether to approve a run's completion."""

    task: str
    plan_markdown: str
    routing_decisions: list[RoutingDecisionRecord] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    verification_result: VerificationResult | None = None
    review_findings: list[ReviewFinding] = field(default_factory=list)
    documentation_updated_files: list[str] = field(default_factory=list)
    outstanding_warnings: list[str] = field(default_factory=list)
    worktree_path: Path | None = None
    branch_name: str | None = None

    def render_markdown(self) -> str:
        """Render the final summary as human-readable Markdown."""
        lines: list[str] = [
            "# Final Summary",
            "",
            "## Task",
            "",
            self.task,
            "",
            "## Plan",
            "",
            self.plan_markdown,
            "",
            "## Selected Models & Routing Reasons",
            "",
        ]
        if self.routing_decisions:
            lines.extend(
                f"- **{d.stage}** ({d.matched_rule}): {d.provider}/{d.model} — {d.reason}"
                for d in self.routing_decisions
            )
        else:
            lines.append("- (none)")

        lines += ["", "## Files Changed", ""]
        if self.changed_files:
            lines.extend(f"- {f}" for f in self.changed_files)
        else:
            lines.append("- (none)")

        lines += ["", "## Verification Status", ""]
        lines.append(self.verification_result.status.value if self.verification_result else "N/A")

        lines += ["", "## Review Status", ""]
        if self.review_findings:
            lines.extend(
                f"- [{f.severity.value}] {f.file}: {f.problem}" for f in self.review_findings
            )
        else:
            lines.append("No findings.")

        lines += ["", "## Documentation Changes", ""]
        if self.documentation_updated_files:
            lines.extend(f"- {f}" for f in self.documentation_updated_files)
        else:
            lines.append("- (none)")

        lines += ["", "## Outstanding Warnings", ""]
        if self.outstanding_warnings:
            lines.extend(f"- {w}" for w in self.outstanding_warnings)
        else:
            lines.append("- (none)")

        lines += [
            "",
            "## Worktree",
            "",
            str(self.worktree_path) if self.worktree_path else "N/A",
            "",
            "## Branch",
            "",
            self.branch_name or "N/A",
        ]
        return "\n".join(lines)


def write_final_summary_artifact(project_root: Path, run_id: str, summary: FinalSummary) -> Path:
    """Persist final-summary.md for a run."""
    run_dir = project_root / ".ai-orchestrator" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "final-summary.md"
    path.write_text(summary.render_markdown(), encoding="utf-8")
    return path
