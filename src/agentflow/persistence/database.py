"""SQLite database manager and entity repositories."""

import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from agentflow.errors import PersistenceError
from agentflow.persistence.migrations import apply_migrations
from agentflow.persistence.models import (
    AgentSessionRecord,
    DecisionRecord,
    ProjectRecord,
    RunRecord,
    RunStateTransitionRecord,
    RunStatus,
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
