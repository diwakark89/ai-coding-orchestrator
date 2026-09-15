"""Unit tests for SQLite database persistence and schema migrations."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.errors import PersistenceError
from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.migrations import MIGRATIONS
from agentflow.persistence.models import RunStatus


def test_sqlite_initialization(tmp_path: Path):
    """DatabaseManager initializes database file, WAL mode, and migrations table."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    assert db_path.exists()

    with mgr.connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "schema_migrations" in tables
        assert "projects" in tables
        assert "runs" in tables


def test_migrations_idempotency(tmp_path: Path):
    """Calling initialize() multiple times does not fail or duplicate migrations."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    # Call again
    mgr.initialize()

    with mgr.connection() as conn:
        cursor = conn.execute("SELECT count(*) FROM schema_migrations;")
        count = cursor.fetchone()[0]
        assert count == len(MIGRATIONS)


def test_project_crud(tmp_path: Path):
    """Test upsert, get, and list project operations."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    # Create project
    proj = mgr.upsert_project(
        project_id="proj_1",
        name="Project One",
        repository_path="/path/to/one",
        config_hash="abc123hash",
    )
    assert proj.id == "proj_1"
    assert proj.name == "Project One"

    # Get project
    retrieved = mgr.get_project("proj_1")
    assert retrieved is not None
    assert retrieved.id == "proj_1"
    assert retrieved.name == "Project One"
    assert retrieved.repository_path == "/path/to/one"

    # Upsert with updated name
    updated = mgr.upsert_project(
        project_id="proj_1",
        name="Project One Renamed",
        repository_path="/path/to/one",
    )
    assert updated.name == "Project One Renamed"

    # List projects
    mgr.upsert_project("proj_2", "Project Two", "/path/to/two")
    all_projects = mgr.list_projects()
    assert len(all_projects) == 2


def test_run_crud(tmp_path: Path):
    """Test creating, getting, querying active runs, and updating run status."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    mgr.upsert_project("proj_1", "Project One", "/path/to/one")

    # Create run
    run1 = mgr.create_run(
        run_id="run_101",
        project_id="proj_1",
        task="Build authentication feature",
        status=RunStatus.PENDING.value,
    )
    assert run1.id == "run_101"
    assert run1.status == RunStatus.PENDING.value

    # Get active runs
    active = mgr.get_active_runs("proj_1")
    assert len(active) == 1
    assert active[0].id == "run_101"

    # Update run status
    mgr.update_run_status("run_101", RunStatus.COMPLETED.value)
    updated_run = mgr.get_run("run_101")
    assert updated_run is not None
    assert updated_run.status == RunStatus.COMPLETED.value

    # Check active runs is now empty
    assert len(mgr.get_active_runs("proj_1")) == 0


def test_foreign_key_constraint(tmp_path: Path):
    """Runs must reference existing project_id when foreign keys are enabled."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    # Attempting to create run with non-existent project_id must fail
    with pytest.raises(sqlite3.IntegrityError):
        mgr.create_run(
            run_id="run_orphan",
            project_id="nonexistent_proj",
            task="Orphaned task",
        )


def test_upsert_project_coalesce_preserves_values_in_return_model(tmp_path: Path):
    """Upserting project with None for optional fields must retain previous values in model."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    initial = mgr.upsert_project(
        project_id="p_coalesce",
        name="Original Name",
        repository_path="/original/path",
        config_hash="initial_hash",
    )
    assert initial.name == "Original Name"
    assert initial.config_hash == "initial_hash"

    # Now update without providing name or config_hash
    updated = mgr.upsert_project(
        project_id="p_coalesce",
        name=None,
        repository_path="/new/path",
        config_hash=None,
    )
    assert updated.name == "Original Name"
    assert updated.config_hash == "initial_hash"
    assert updated.repository_path == "/new/path"

    # Also verify from a fresh query
    fetched = mgr.get_project("p_coalesce")
    assert fetched is not None
    assert fetched.name == "Original Name"
    assert fetched.config_hash == "initial_hash"


def test_run_created_with_new_workflow_state(tmp_path: Path):
    """A newly created run defaults to workflow state NEW."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")

    run = mgr.create_run(run_id="run_state_1", project_id="proj_1", task="Do a thing")
    assert run.state == "NEW"

    fetched = mgr.get_run("run_state_1")
    assert fetched is not None
    assert fetched.state == "NEW"


def test_update_run_state_persists_transition_history(tmp_path: Path):
    """update_run_state updates the run's current state and records transition history."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_state_2", project_id="proj_1", task="Do a thing")

    updated = mgr.update_run_state("run_state_2", "PROJECT_READY", reason="Project discovered")
    assert updated.state == "PROJECT_READY"

    updated = mgr.update_run_state("run_state_2", "PLANNING", reason="Planning started")
    assert updated.state == "PLANNING"

    transitions = mgr.list_state_transitions("run_state_2")
    assert [t.to_state for t in transitions] == ["PROJECT_READY", "PLANNING"]
    assert transitions[0].from_state == "NEW"
    assert transitions[1].from_state == "PROJECT_READY"
    assert transitions[1].reason == "Planning started"


def test_update_run_state_missing_run_raises(tmp_path: Path):
    """update_run_state raises PersistenceError for an unknown run ID."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    with pytest.raises(PersistenceError):
        mgr.update_run_state("does_not_exist", "PLANNING")


def test_record_and_list_agent_sessions(tmp_path: Path):
    """Agent sessions can be recorded and listed chronologically for a run."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_sessions", project_id="proj_1", task="Do a thing")

    mgr.record_agent_session(
        "sess_1", "run_sessions", "PLANNING", "anthropic", "Claude Sonnet 5", "cli-session-abc"
    )
    mgr.record_agent_session(
        "sess_2", "run_sessions", "PLANNING", "anthropic", "Claude Opus 5", None
    )

    sessions = mgr.list_agent_sessions("run_sessions")
    assert [s.id for s in sessions] == ["sess_1", "sess_2"]
    assert sessions[0].cli_session_id == "cli-session-abc"
    assert sessions[1].cli_session_id is None


def test_record_and_list_decisions(tmp_path: Path):
    """User decisions (question/answer pairs) can be recorded and listed for a run."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_decisions", project_id="proj_1", task="Do a thing")

    mgr.record_decision("dec_1", "run_decisions", "Use REST or GraphQL?", "REST")

    decisions = mgr.list_decisions("run_decisions")
    assert len(decisions) == 1
    assert decisions[0].question == "Use REST or GraphQL?"
    assert decisions[0].answer == "REST"


def test_record_and_list_routing_decisions(tmp_path: Path):
    """Routing decisions can be recorded and listed chronologically for a run."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_routing", project_id="proj_1", task="Do a thing")

    mgr.record_routing_decision(
        "route_1",
        "run_routing",
        "IMPLEMENTATION",
        '{"stage": "IMPLEMENTATION"}',
        2,
        "LOW",
        "implementation.force-standard",
        "openai",
        "GPT-5.6 Terra",
        "Authorization requires the standard tier.",
    )

    decisions = mgr.list_routing_decisions("run_routing")
    assert len(decisions) == 1
    assert decisions[0].model == "GPT-5.6 Terra"
    assert decisions[0].matched_rule == "implementation.force-standard"


def test_create_and_update_stage(tmp_path: Path):
    """Stages track per-attempt status, incrementing attempt_count and marking completion."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_stage", project_id="proj_1", task="Do a thing")

    stage = mgr.create_stage("stage_1", "run_stage", "implementation")
    assert stage.status == "IN_PROGRESS"
    assert stage.attempt_count == 0
    assert stage.completed_at is None

    updated = mgr.update_stage("stage_1", increment_attempt=True)
    assert updated.attempt_count == 1

    updated = mgr.update_stage("stage_1", status="FAILED", increment_attempt=True)
    assert updated.attempt_count == 2
    assert updated.status == "FAILED"

    completed = mgr.update_stage("stage_1", status="PASSED", completed=True)
    assert completed.status == "PASSED"
    assert completed.completed_at is not None


def test_update_stage_missing_raises(tmp_path: Path):
    """update_stage raises PersistenceError for an unknown stage ID."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()

    with pytest.raises(PersistenceError):
        mgr.update_stage("does_not_exist", status="FAILED")


def test_list_stages_chronological(tmp_path: Path):
    """list_stages returns a run's stages in chronological order."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_stages", project_id="proj_1", task="Do a thing")

    mgr.create_stage("stage_a", "run_stages", "implementation")
    mgr.create_stage("stage_b", "run_stages", "repair.lightweight")

    stages = mgr.list_stages("run_stages")
    assert [s.id for s in stages] == ["stage_a", "stage_b"]


def test_record_and_list_verification_runs(tmp_path: Path):
    """Verification command executions can be recorded and listed chronologically."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_verify", project_id="proj_1", task="Do a thing")

    now = datetime.now(timezone.utc)
    mgr.record_verification_run(
        "vr_1", "run_verify", "pytest", 0, now, now, stdout_path="/tmp/out.log"
    )
    mgr.record_verification_run("vr_2", "run_verify", "ruff check .", 1, now, now)

    records = mgr.list_verification_runs("run_verify")
    assert len(records) == 2
    assert records[0].command == "pytest"
    assert records[0].exit_code == 0
    assert records[1].exit_code == 1
    assert records[1].stdout_path is None


def test_record_and_list_safe_observability_events(tmp_path: Path):
    """Structured local events retain operational metadata without task or agent output."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize()
    mgr.upsert_project("proj_1", "Project One", "/path/to/one")
    mgr.create_run(run_id="run_events", project_id="proj_1", task="Do a thing")

    event = mgr.record_event(
        run_id="run_events",
        stage="IMPLEMENTING",
        event="AGENT_COMPLETED",
        provider="openai",
        model="GPT-5.6 Luna",
        attributes={"exit_code": 0, "duration_seconds": 1.25, "timed_out": False},
    )

    assert event.attributes["exit_code"] == 0
    events = mgr.list_events("run_events")
    assert len(events) == 1
    assert events[0].event == "AGENT_COMPLETED"
    assert events[0].provider == "openai"
    assert events[0].attributes == {
        "duration_seconds": 1.25,
        "exit_code": 0,
        "timed_out": False,
    }


def test_database_manager_in_memory():
    """DatabaseManager supports :memory: connection without path resolution crashes."""
    mgr = DatabaseManager(":memory:")
    mgr.initialize()

    proj = mgr.upsert_project("mem_p1", "Memory Project", "/mem/repo")
    assert proj.id == "mem_p1"
    assert proj.name == "Memory Project"

    fetched = mgr.get_project("mem_p1")
    assert fetched is not None
    assert fetched.name == "Memory Project"
    mgr.close()
