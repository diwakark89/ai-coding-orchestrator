"""Safe local event helpers for AgentFlow observability.

Events intentionally include only operational metadata: identifiers, stage names, provider/model
selection, result codes, counts, and durations. Prompts, task text, agent output, diffs, file
contents, and verification output must remain in their existing scoped artifacts, never here.
"""

from agentflow.agents.base import AgentResult
from agentflow.persistence.database import DatabaseManager


def record_agent_started(
    db_manager: DatabaseManager,
    run_id: str,
    stage: str,
    provider: str,
    model: str,
) -> None:
    """Record that an agent CLI invocation has started."""
    db_manager.record_event(
        run_id=run_id,
        stage=stage,
        event="AGENT_STARTED",
        provider=provider,
        model=model,
    )


def record_agent_completed(
    db_manager: DatabaseManager,
    run_id: str,
    stage: str,
    result: AgentResult,
) -> None:
    """Record a completed agent invocation using only safe operational fields."""
    db_manager.record_event(
        run_id=run_id,
        stage=stage,
        event="AGENT_COMPLETED",
        provider=result.provider.value,
        model=result.model,
        attributes={
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": round(result.duration_seconds, 3),
        },
    )


def record_review_completed(
    db_manager: DatabaseManager,
    run_id: str,
    provider: str,
    model: str,
    severity_counts: dict[str, int],
) -> None:
    """Record only the count and severities of review findings, never their text."""
    attributes: dict[str, str | int | float | bool] = {
        "finding_count": sum(severity_counts.values()),
        **{f"severity_{severity.lower()}": count for severity, count in severity_counts.items()},
    }
    db_manager.record_event(
        run_id=run_id,
        stage="REVIEWING",
        event="REVIEW_COMPLETED",
        provider=provider,
        model=model,
        attributes=attributes,
    )
