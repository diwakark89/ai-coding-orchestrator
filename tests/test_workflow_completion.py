"""Unit tests for the final run summary artifact (Phase 9)."""

from datetime import datetime, timezone
from pathlib import Path

from agentflow.persistence.models import RoutingDecisionRecord
from agentflow.workflow.completion import FinalSummary, write_final_summary_artifact
from agentflow.workflow.review import ReviewFinding, ReviewSeverity
from agentflow.workflow.verification import VerificationResult, VerificationStatus


def make_routing_decision_record() -> RoutingDecisionRecord:
    return RoutingDecisionRecord(
        id="rd_1",
        run_id="run_1",
        stage="IMPLEMENTATION",
        task_profile_json="{}",
        complexity_score=0,
        complexity_level="LOW",
        matched_rule="low-complexity",
        provider="openai",
        model="GPT-5.6 Luna",
        reason="No risk flags.",
        created_at=datetime.now(timezone.utc),
    )


def test_final_summary_renders_all_required_sections():
    """render_markdown() includes every section required by Phase 9 step 1."""
    summary = FinalSummary(
        task="Add a health check endpoint",
        plan_markdown="# Plan\n\nObjective: ...",
        routing_decisions=[make_routing_decision_record()],
        changed_files=["src/app.py"],
        verification_result=VerificationResult(
            status=VerificationStatus.PASSED, groups_run=["py"], command_results=[]
        ),
        review_findings=[
            ReviewFinding(
                severity=ReviewSeverity.LOW,
                category="style",
                file="src/app.py",
                line=10,
                problem="Minor style nit.",
                recommendation="Consider renaming.",
            )
        ],
        documentation_updated_files=["architecture.md"],
        outstanding_warnings=["Gemini CLI not installed."],
        worktree_path=Path("/tmp/worktree"),
        branch_name="agentflow/RUN-001",
    )

    markdown = summary.render_markdown()

    assert "Add a health check endpoint" in markdown
    assert "Objective: ..." in markdown
    assert "GPT-5.6 Luna" in markdown
    assert "src/app.py" in markdown
    assert "PASSED" in markdown
    assert "Minor style nit." in markdown
    assert "architecture.md" in markdown
    assert "Gemini CLI not installed." in markdown
    assert "worktree" in markdown.lower()
    assert "agentflow/RUN-001" in markdown


def test_final_summary_handles_empty_sections():
    """render_markdown() renders sensible placeholders when nothing is present."""
    summary = FinalSummary(task="Add a feature", plan_markdown="# Plan")

    markdown = summary.render_markdown()

    assert "N/A" in markdown
    assert "(none)" in markdown
    assert "No findings." in markdown


def test_write_final_summary_artifact_persists_file(tmp_path: Path):
    """write_final_summary_artifact() persists final-summary.md under the run directory."""
    summary = FinalSummary(task="Add a feature", plan_markdown="# Plan")

    path = write_final_summary_artifact(tmp_path, "run_1", summary)

    assert path.exists()
    assert path.name == "final-summary.md"
    assert path.parent == tmp_path / ".ai-orchestrator" / "runs" / "run_1"
    assert "Add a feature" in path.read_text(encoding="utf-8")
