"""Unit tests for Application orchestration and Doctor checks."""

from pathlib import Path

from agentflow.application import Application, DoctorCheckItem, DoctorReport
from agentflow.config.models import GlobalConfig
from agentflow.persistence.database import DatabaseManager


def test_doctor_report_exit_code():
    """Exit code is 0 when no critical failures exist, 1 when critical fails."""
    healthy_report = DoctorReport(
        items=[
            DoctorCheckItem(name="Python", passed=True, critical=True),
            DoctorCheckItem(name="Codex CLI", passed=False, critical=False, is_warning=True),
        ]
    )
    assert healthy_report.has_critical_failures is False
    assert healthy_report.exit_code == 0

    unhealthy_report = DoctorReport(
        items=[
            DoctorCheckItem(name="Python", passed=False, critical=True),
        ]
    )
    assert unhealthy_report.has_critical_failures is True
    assert unhealthy_report.exit_code == 1


def test_application_doctor_run(tmp_path: Path):
    """Application run_doctor executes and generates comprehensive report."""
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "storage": {"database": str(tmp_path / "app.db")},
            "worktrees": {"root": str(tmp_path / "worktrees")},
            "logging": {"root": str(tmp_path / "logs")},
        }
    )
    app = Application(config=cfg)
    report = app.run_doctor(project_path=tmp_path)

    assert len(report.items) >= 7
    # Python check must pass in our test environment
    py_check = next(i for i in report.items if i.name == "Python runtime")
    assert py_check.passed is True

    # Storage check must pass in tmp_path
    storage_check = next(i for i in report.items if i.name == "Global data storage")
    assert storage_check.passed is True


def test_application_status_message(tmp_path: Path):
    """Application status reports no active runs initially, and reports active runs when created."""
    cfg = GlobalConfig.model_validate(
        {
            "version": 1,
            "storage": {"database": str(tmp_path / "app.db")},
            "worktrees": {"root": str(tmp_path / "worktrees")},
            "logging": {"root": str(tmp_path / "logs")},
        }
    )
    db = DatabaseManager(tmp_path / "app.db")
    app = Application(config=cfg, db_manager=db)

    # Empty status
    msg = app.get_status_message(project_path=tmp_path)
    assert "No active runs found" in msg

    # Add active run
    app.initialize()
    from agentflow.project.discovery import discover_project

    ctx = discover_project(explicit_path=tmp_path)
    db.upsert_project(ctx.project_id, ctx.project_name, str(ctx.root_path))
    db.create_run("RUN-100", ctx.project_id, "Sample test run", "RUNNING")

    msg_with_run = app.get_status_message(project_path=tmp_path)
    assert "Active runs (1)" in msg_with_run
    assert "RUN-100" in msg_with_run


def test_doctor_python_version_failure(monkeypatch):
    """Doctor marks Python check as failed and sets exit code 1 if Python < 3.12."""
    app = Application()
    monkeypatch.setattr("sys.version_info", (3, 11, 5, "final", 0))

    py_check = app.check_python()
    assert py_check.passed is False
    assert py_check.critical is True

    report = app.run_doctor()
    assert report.has_critical_failures is True
    assert report.exit_code == 1


def test_doctor_git_missing_failure(monkeypatch):
    """Doctor marks Git check as failed and sets exit code 1 if git is not in PATH."""
    app = Application()
    monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "git" else "/bin/other")

    git_check = app.check_git()
    assert git_check.passed is False
    assert git_check.critical is True


def test_doctor_invalid_routing_config_failure(tmp_path: Path):
    """Doctor marks invalid routing.yaml as critical failure with exit code 1."""
    (tmp_path / ".git").mkdir()
    orch_dir = tmp_path / ".ai-orchestrator"
    orch_dir.mkdir()
    (orch_dir / "routing.yaml").write_text("version: 99\n", encoding="utf-8")

    app = Application()
    report = app.run_doctor(project_path=tmp_path)

    config_check = next(i for i in report.items if i.name == "Routing configuration")
    assert config_check.passed is False
    assert config_check.critical is True
    assert "Unsupported configuration version: 99" in config_check.details
    assert report.has_critical_failures is True
    assert report.exit_code == 1


def test_doctor_explicit_nonexistent_project_failure(tmp_path: Path):
    """Doctor with non-existent explicit project path marks critical failure."""
    nonexistent = tmp_path / "does_not_exist"
    app = Application()
    report = app.run_doctor(project_path=nonexistent)

    git_check = next(i for i in report.items if i.name == "Git repository")
    assert git_check.passed is False
    assert git_check.critical is True
    assert report.has_critical_failures is True
    assert report.exit_code == 1


def test_doctor_git_command_execution_error(monkeypatch):
    """Doctor gracefully catches exceptions during git execution and marks check failed."""
    app = Application()
    monkeypatch.setattr(
        app.executor,
        "run_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Unexpected subprocess crash")),
    )

    git_check = app.check_git()
    assert git_check.passed is False
    assert git_check.critical is True
    assert "Command execution error" in git_check.details
