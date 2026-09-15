"""Tests for Phase 11's local, informational routing analytics."""

from pathlib import Path

from agentflow.observability.metrics import StatisticsService
from agentflow.persistence.database import DatabaseManager


def _seed_run(
    db_manager: DatabaseManager,
    run_id: str,
    model: str,
    matched_rule: str,
    first_pass_success: bool,
    severity: str | None = None,
) -> None:
    """Persist one complete, non-sensitive run shape for deterministic reporting tests."""
    db_manager.create_run(run_id, "project_1", "Synthetic task")
    db_manager.record_routing_decision(
        f"route_{run_id}",
        run_id,
        "IMPLEMENTATION",
        '{"stage":"IMPLEMENTATION"}',
        2,
        "LOW",
        matched_rule,
        "openai",
        model,
        "Synthetic deterministic reason",
    )
    stage = db_manager.create_stage(f"stage_{run_id}", run_id, "implementation", attempt_count=1)
    db_manager.update_stage(stage.id, status="PASSED", completed=True)
    db_manager.record_event(
        run_id,
        "VERIFYING",
        "VERIFICATION_COMPLETED",
        attributes={"success": first_pass_success, "status": "PASSED", "command_count": 1},
    )
    if severity is not None:
        db_manager.record_event(
            run_id,
            "REVIEWING",
            "REVIEW_COMPLETED",
            provider="google",
            model="Gemini 3.8 Flash",
            attributes={"finding_count": 1, f"severity_{severity.lower()}": 1},
        )


def test_statistics_are_reproducible_from_persisted_records(tmp_path: Path):
    """Model/rule/review metrics are derived deterministically from SQLite records only."""
    db_manager = DatabaseManager(tmp_path / "stats.db")
    db_manager.initialize()
    db_manager.upsert_project("project_1", "Demo", str(tmp_path))
    _seed_run(
        db_manager,
        "RUN-LUNA",
        "GPT-5.6 Luna",
        "complexity.low",
        first_pass_success=True,
        severity="HIGH",
    )
    _seed_run(
        db_manager,
        "RUN-TERRA",
        "GPT-5.6 Terra",
        "hard-risk.force-standard",
        first_pass_success=False,
    )
    db_manager.record_decision("escalation_1", "RUN-TERRA", "architecture_escalation", "test")

    report = StatisticsService(db_manager).build_report("project_1")

    assert report.total_runs == 2
    by_model = {metric.model: metric for metric in report.models}
    assert by_model["GPT-5.6 Luna"].tasks == 1
    assert by_model["GPT-5.6 Luna"].routing_selections == 1
    assert by_model["GPT-5.6 Luna"].first_pass_rate == 1.0
    assert by_model["GPT-5.6 Terra"].first_pass_rate == 0.0
    assert by_model["GPT-5.6 Terra"].escalations == 1
    assert by_model["Gemini 3.8 Flash"].review_findings == 1
    assert report.routing_rule_frequency == {
        "complexity.low": 1,
        "hard-risk.force-standard": 1,
    }
    assert report.review_findings_by_severity == {"HIGH": 1}
    assert "implementation" in report.average_stage_duration_seconds


def test_statistics_do_not_require_implementation_routes(tmp_path: Path):
    """A project with only planning data reports zero models without failing or changing state."""
    db_manager = DatabaseManager(tmp_path / "stats.db")
    db_manager.initialize()
    db_manager.upsert_project("project_1", "Demo", str(tmp_path))
    db_manager.create_run("RUN-PLANNING", "project_1", "Synthetic task")

    report = StatisticsService(db_manager).build_report("project_1")

    assert report.total_runs == 1
    assert report.models == []
