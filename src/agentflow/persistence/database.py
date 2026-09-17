"""SQLite database manager and entity repositories."""

import json
import sqlite3
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from agentflow.errors import PersistenceError
from agentflow.persistence.migrations import apply_migrations
from agentflow.persistence.models import (
    AgentSessionRecord,
    CliAvailabilityRecord,
    DecisionRecord,
    EventRecord,
    ProjectRecord,
    RoutingDecisionRecord,
    RunRecord,
    RunStateTransitionRecord,
    RunStatus,
    StageRecord,
    VerificationRunRecord,
)


class DatabaseManager:
    """Manages SQLite connection lifecycle, schema migrations, and entity queries."""

    def __init__(self, db_path: Path | str) -> None:
        self.is_memory = str(db_path) == ":memory:" or str(db_path).startswith("file:")
        self.db_path = Path(":memory:") if self.is_memory else Path(db_path).resolve()
        self._memory_hold: sqlite3.Connection | None = None
        if self.is_memory:
            # Keep one anchor connection open so in-memory DB survives connection closing
            self._memory_hold = sqlite3.connect("file::memory:?cache=shared", uri=True)

    def close(self) -> None:
        """Close persistent anchor connections."""
        if self._memory_hold is not None:
            try:
                self._memory_hold.close()
            except Exception:
                pass
            self._memory_hold = None

    def __del__(self) -> None:
        self.close()

    def initialize(self) -> None:
        """Create storage directory and apply pending schema migrations."""
        try:
            if not self.is_memory:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self.connection() as conn:
                if not self.is_memory:
                    try:
                        conn.execute("PRAGMA journal_mode = WAL;")
                    except sqlite3.OperationalError:
                        # In-memory or certain file systems might not support WAL
                        pass
                conn.execute("PRAGMA foreign_keys = ON;")
                apply_migrations(conn)
        except Exception as e:
            raise PersistenceError(
                f"Failed to initialize SQLite database at '{self.db_path}': {e}"
            ) from e

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active SQLite connection with foreign keys enabled."""
        connect_target = "file::memory:?cache=shared" if self.is_memory else str(self.db_path)
        conn = sqlite3.connect(
            connect_target,
            timeout=15.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            uri=self.is_memory,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_project(
        self,
        project_id: str,
        name: str | None,
        repository_path: str,
        config_hash: str | None = None,
    ) -> ProjectRecord:
        """Insert or update a project record, updating last_used_at."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.execute("SELECT created_at FROM projects WHERE id = ?;", (project_id,))
            row = cursor.fetchone()
            if row is None:
                created_at = now
                conn.execute(
                    """
                    INSERT INTO projects (
                        id, name, repository_path, config_hash, created_at, last_used_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (project_id, name, repository_path, config_hash, created_at, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE projects
                    SET name = coalesce(?, name),
                        repository_path = ?,
                        config_hash = coalesce(?, config_hash),
                        last_used_at = ?
                    WHERE id = ?;
                    """,
                    (name, repository_path, config_hash, now, project_id),
                )

            # Retrieve actual updated row to reflect coalesced values faithfully
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?;", (project_id,))
            updated_row = cursor.fetchone()

        return ProjectRecord(
            id=updated_row["id"],
            name=updated_row["name"],
            repository_path=updated_row["repository_path"],
            config_hash=updated_row["config_hash"],
            created_at=datetime.fromisoformat(updated_row["created_at"]),
            last_used_at=datetime.fromisoformat(updated_row["last_used_at"]),
        )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        """Retrieve project by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM projects WHERE id = ?;", (project_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return ProjectRecord(
                id=row["id"],
                name=row["name"],
                repository_path=row["repository_path"],
                config_hash=row["config_hash"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used_at=datetime.fromisoformat(row["last_used_at"]),
            )

    def get_cli_availability(self, provider: str) -> CliAvailabilityRecord | None:
        """Retrieve the most recently recorded `doctor` check for one provider CLI."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM cli_availability WHERE provider = ?;", (provider,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return CliAvailabilityRecord(
                provider=row["provider"],
                command=row["command"],
                available=bool(row["available"]),
                last_checked_at=datetime.fromisoformat(row["last_checked_at"]),
            )

    def upsert_cli_availability(
        self, provider: str, command: str, available: bool, checked_at: str
    ) -> None:
        """Record the outcome of the latest `doctor` check for one provider CLI."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO cli_availability (provider, command, available, last_checked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    command = excluded.command,
                    available = excluded.available,
                    last_checked_at = excluded.last_checked_at;
                """,
                (provider, command, int(available), checked_at),
            )

    def list_projects(self) -> list[ProjectRecord]:
        """List all tracked projects ordered by last used."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM projects ORDER BY last_used_at DESC;")
            records = []
            for row in cursor.fetchall():
                records.append(
                    ProjectRecord(
                        id=row["id"],
                        name=row["name"],
                        repository_path=row["repository_path"],
                        config_hash=row["config_hash"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        last_used_at=datetime.fromisoformat(row["last_used_at"]),
                    )
                )
            return records

    def create_run(
        self,
        run_id: str,
        project_id: str,
        task: str,
        status: str = RunStatus.PENDING.value,
        state: str = "NEW",
    ) -> RunRecord:
        """Create a new run entity in the database."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (id, project_id, task, status, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (run_id, project_id, task, status, state, now, now),
            )

        return RunRecord(
            id=run_id,
            project_id=project_id,
            task=task,
            status=status,
            state=state,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve run by ID."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM runs WHERE id = ?;", (run_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return RunRecord(
                id=row["id"],
                project_id=row["project_id"],
                task=row["task"],
                status=row["status"],
                state=row["state"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def get_active_runs(self, project_id: str | None = None) -> list[RunRecord]:
        """Retrieve runs currently in PENDING or RUNNING status."""
        query = (
            "SELECT * FROM runs WHERE status IN (?, ?) "
            + ("AND project_id = ? " if project_id else "")
            + "ORDER BY created_at DESC;"
        )
        params: list[str] = [RunStatus.PENDING.value, RunStatus.RUNNING.value]
        if project_id:
            params.append(project_id)

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            runs = []
            for row in cursor.fetchall():
                runs.append(
                    RunRecord(
                        id=row["id"],
                        project_id=row["project_id"],
                        task=row["task"],
                        status=row["status"],
                        state=row["state"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return runs

    def list_runs(self, project_id: str | None = None, limit: int = 20) -> list[RunRecord]:
        """List runs optionally filtered by project."""
        query = (
            "SELECT * FROM runs "
            + ("WHERE project_id = ? " if project_id else "")
            + "ORDER BY created_at DESC LIMIT ?;"
        )
        params: list[object] = [project_id, limit] if project_id else [limit]

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            runs = []
            for row in cursor.fetchall():
                runs.append(
                    RunRecord(
                        id=row["id"],
                        project_id=row["project_id"],
                        task=row["task"],
                        status=row["status"],
                        state=row["state"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return runs

    def update_run_status(self, run_id: str, status: str) -> None:
        """Update status and updated_at timestamp for a run."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?;",
                (status, now, run_id),
            )
            if cursor.rowcount == 0:
                raise PersistenceError(f"Run with ID '{run_id}' not found.")

    def update_run_state(self, run_id: str, to_state: str, reason: str | None = None) -> RunRecord:
        """Persist a workflow-state transition for a run and record transition history."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cursor = conn.execute("SELECT state FROM runs WHERE id = ?;", (run_id,))
            row = cursor.fetchone()
            if row is None:
                raise PersistenceError(f"Run with ID '{run_id}' not found.")
            from_state = row["state"]

            conn.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE id = ?;",
                (to_state, now, run_id),
            )
            conn.execute(
                """
                INSERT INTO run_state_transitions (
                    id, run_id, from_state, to_state, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (str(uuid.uuid4()), run_id, from_state, to_state, reason, now),
            )
            conn.execute(
                """
                INSERT INTO events (
                    id, run_id, stage, event, provider, model, attributes_json, created_at
                )
                VALUES (?, ?, ?, ?, NULL, NULL, ?, ?);
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    to_state,
                    "STATE_TRANSITIONED",
                    json.dumps({"from_state": from_state, "to_state": to_state}),
                    now,
                ),
            )

            cursor = conn.execute("SELECT * FROM runs WHERE id = ?;", (run_id,))
            updated_row = cursor.fetchone()

        return RunRecord(
            id=updated_row["id"],
            project_id=updated_row["project_id"],
            task=updated_row["task"],
            status=updated_row["status"],
            state=updated_row["state"],
            created_at=datetime.fromisoformat(updated_row["created_at"]),
            updated_at=datetime.fromisoformat(updated_row["updated_at"]),
        )

    def list_state_transitions(self, run_id: str) -> list[RunStateTransitionRecord]:
        """List workflow-state transition history for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM run_state_transitions WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [
                RunStateTransitionRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    from_state=row["from_state"],
                    to_state=row["to_state"],
                    reason=row["reason"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def record_agent_session(
        self,
        session_id: str,
        run_id: str,
        stage: str,
        provider: str,
        model: str,
        cli_session_id: str | None = None,
    ) -> AgentSessionRecord:
        """Record an individual agent CLI session invoked during a run."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_sessions (
                    id, run_id, stage, provider, model, cli_session_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (session_id, run_id, stage, provider, model, cli_session_id, now),
            )

        return AgentSessionRecord(
            id=session_id,
            run_id=run_id,
            stage=stage,
            provider=provider,
            model=model,
            cli_session_id=cli_session_id,
            created_at=datetime.fromisoformat(now),
        )

    def list_agent_sessions(self, run_id: str) -> list[AgentSessionRecord]:
        """List agent CLI sessions recorded for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM agent_sessions WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [
                AgentSessionRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    stage=row["stage"],
                    provider=row["provider"],
                    model=row["model"],
                    cli_session_id=row["cli_session_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def record_decision(
        self,
        decision_id: str,
        run_id: str,
        question: str,
        answer: str,
    ) -> DecisionRecord:
        """Record a user decision (question/answer) made during a run."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO decisions (id, run_id, question, answer, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (decision_id, run_id, question, answer, now),
            )

        return DecisionRecord(
            id=decision_id,
            run_id=run_id,
            question=question,
            answer=answer,
            created_at=datetime.fromisoformat(now),
        )

    def list_decisions(self, run_id: str) -> list[DecisionRecord]:
        """List user decisions recorded for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM decisions WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [
                DecisionRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    question=row["question"],
                    answer=row["answer"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def record_routing_decision(
        self,
        decision_id: str,
        run_id: str,
        stage: str,
        task_profile_json: str,
        complexity_score: int,
        complexity_level: str,
        matched_rule: str,
        provider: str,
        model: str,
        reason: str,
    ) -> RoutingDecisionRecord:
        """Persist a deterministic RoutingDecision for a run."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO routing_decisions (
                    id, run_id, stage, task_profile_json, complexity_score, complexity_level,
                    matched_rule, provider, model, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    decision_id,
                    run_id,
                    stage,
                    task_profile_json,
                    complexity_score,
                    complexity_level,
                    matched_rule,
                    provider,
                    model,
                    reason,
                    now,
                ),
            )

        record = RoutingDecisionRecord(
            id=decision_id,
            run_id=run_id,
            stage=stage,
            task_profile_json=task_profile_json,
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            matched_rule=matched_rule,
            provider=provider,
            model=model,
            reason=reason,
            created_at=datetime.fromisoformat(now),
        )
        self.record_event(
            run_id=run_id,
            stage=stage,
            event="ROUTING_SELECTED",
            provider=provider,
            model=model,
            attributes={
                "matched_rule": matched_rule,
                "complexity_score": complexity_score,
                "complexity_level": complexity_level,
            },
        )
        return record

    def list_routing_decisions(self, run_id: str) -> list[RoutingDecisionRecord]:
        """List routing decisions recorded for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM routing_decisions WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            )
            return [
                RoutingDecisionRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    stage=row["stage"],
                    task_profile_json=row["task_profile_json"],
                    complexity_score=row["complexity_score"],
                    complexity_level=row["complexity_level"],
                    matched_rule=row["matched_rule"],
                    provider=row["provider"],
                    model=row["model"],
                    reason=row["reason"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in cursor.fetchall()
            ]

    def create_stage(
        self,
        stage_id: str,
        run_id: str,
        stage: str,
        status: str = "IN_PROGRESS",
        attempt_count: int = 0,
    ) -> StageRecord:
        """Create a stage-attempt tracking row for a run.

        `stage` identifies the workflow stage, e.g. 'implementation' or 'repair.lightweight'.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO stages (
                    id, run_id, stage, status, attempt_count, started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL);
                """,
                (stage_id, run_id, stage, status, attempt_count, now),
            )

        return StageRecord(
            id=stage_id,
            run_id=run_id,
            stage=stage,
            status=status,
            attempt_count=attempt_count,
            started_at=datetime.fromisoformat(now),
            completed_at=None,
        )

    def update_stage(
        self,
        stage_id: str,
        status: str | None = None,
        increment_attempt: bool = False,
        completed: bool = False,
    ) -> StageRecord:
        """Update a stage's status and/or bump its attempt_count; optionally mark it completed."""
        with self.connection() as conn:
            cursor = conn.execute("SELECT * FROM stages WHERE id = ?;", (stage_id,))
            row = cursor.fetchone()
            if row is None:
                raise PersistenceError(f"Stage with ID '{stage_id}' not found.")

            new_status = status if status is not None else row["status"]
            new_attempt_count = (
                row["attempt_count"] + 1 if increment_attempt else row["attempt_count"]
            )
            new_completed_at = (
                datetime.now(timezone.utc).isoformat() if completed else row["completed_at"]
            )

            conn.execute(
                """
                UPDATE stages SET status = ?, attempt_count = ?, completed_at = ? WHERE id = ?;
                """,
                (new_status, new_attempt_count, new_completed_at, stage_id),
            )

        return StageRecord(
            id=row["id"],
            run_id=row["run_id"],
            stage=row["stage"],
            status=new_status,
            attempt_count=new_attempt_count,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(new_completed_at) if new_completed_at else None,
        )

    def list_stages(self, run_id: str) -> list[StageRecord]:
        """List stage-attempt tracking rows for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM stages WHERE run_id = ? ORDER BY started_at ASC;",
                (run_id,),
            )
            return [
                StageRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    stage=row["stage"],
                    status=row["status"],
                    attempt_count=row["attempt_count"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=(
                        datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                    ),
                )
                for row in cursor.fetchall()
            ]

    def record_verification_run(
        self,
        record_id: str,
        run_id: str,
        command: str,
        exit_code: int,
        started_at: datetime,
        completed_at: datetime,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ) -> VerificationRunRecord:
        """Persist a single verification command execution."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO verification_runs (
                    id, run_id, command, exit_code, stdout_path, stderr_path,
                    started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record_id,
                    run_id,
                    command,
                    exit_code,
                    stdout_path,
                    stderr_path,
                    started_at.isoformat(),
                    completed_at.isoformat(),
                ),
            )

        return VerificationRunRecord(
            id=record_id,
            run_id=run_id,
            command=command,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_at=started_at,
            completed_at=completed_at,
        )

    def list_verification_runs(self, run_id: str) -> list[VerificationRunRecord]:
        """List verification command executions recorded for a run in chronological order."""
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM verification_runs WHERE run_id = ? ORDER BY started_at ASC;",
                (run_id,),
            )
            return [
                VerificationRunRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    command=row["command"],
                    exit_code=row["exit_code"],
                    stdout_path=row["stdout_path"],
                    stderr_path=row["stderr_path"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]),
                )
                for row in cursor.fetchall()
            ]

    def record_event(
        self,
        run_id: str,
        stage: str | None,
        event: str,
        provider: str | None = None,
        model: str | None = None,
        attributes: Mapping[str, str | int | float | bool] | None = None,
    ) -> EventRecord:
        """Persist a safe, structured local event without task text or agent output."""
        now = datetime.now(timezone.utc).isoformat()
        event_id = str(uuid.uuid4())
        safe_attributes = dict(attributes or {})
        attributes_json = json.dumps(safe_attributes, sort_keys=True)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    id, run_id, stage, event, provider, model, attributes_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (event_id, run_id, stage, event, provider, model, attributes_json, now),
            )
        return EventRecord(
            id=event_id,
            run_id=run_id,
            stage=stage,
            event=event,
            provider=provider,
            model=model,
            attributes=safe_attributes,
            created_at=datetime.fromisoformat(now),
        )

    def list_events(self, run_id: str | None = None) -> list[EventRecord]:
        """List local observability events chronologically, optionally for one run."""
        query = "SELECT * FROM events " + ("WHERE run_id = ? " if run_id else "")
        query += "ORDER BY created_at ASC;"
        params: tuple[str, ...] = (run_id,) if run_id else ()
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            records: list[EventRecord] = []
            for row in cursor.fetchall():
                decoded = json.loads(row["attributes_json"])
                attributes = (
                    {
                        str(key): value
                        for key, value in decoded.items()
                        if isinstance(value, str | int | float | bool)
                    }
                    if isinstance(decoded, dict)
                    else {}
                )
                records.append(
                    EventRecord(
                        id=row["id"],
                        run_id=row["run_id"],
                        stage=row["stage"],
                        event=row["event"],
                        provider=row["provider"],
                        model=row["model"],
                        attributes=attributes,
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return records
