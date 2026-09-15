"""Database migrations and schema management for SQLite."""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from agentflow.errors import PersistenceError


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATION_001_INITIAL_SCHEMA = Migration(
    version=1,
    name="initial_schema",
    sql="""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT,
        repository_path TEXT NOT NULL,
        config_hash TEXT,
        created_at TEXT NOT NULL,
        last_used_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        task TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_runs_project_id ON runs(project_id);
    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    """,
)

MIGRATIONS: Sequence[Migration] = [
    MIGRATION_001_INITIAL_SCHEMA,
]


def ensure_migration_table(conn: sqlite3.Connection) -> None:
    """Ensure the schema_migrations tracking table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def get_applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Retrieve the set of migration versions already applied."""
    cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    return {row[0] for row in cursor.fetchall()}


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations in ascending order within individual transactions.

    Returns the number of newly applied migrations.
    """
    try:
        ensure_migration_table(conn)
        applied = get_applied_versions(conn)
        newly_applied = 0

        for migration in sorted(MIGRATIONS, key=lambda m: m.version):
            if migration.version in applied:
                continue

            with conn:
                conn.executescript(migration.sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                    (
                        migration.version,
                        migration.name,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            newly_applied += 1

        return newly_applied
    except Exception as e:
        raise PersistenceError(f"Failed to apply database migrations: {e}") from e
