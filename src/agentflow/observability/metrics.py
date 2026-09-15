"""Informational, reproducible local routing analytics.

This module only reads AgentFlow's SQLite records. It never changes routing rules, models, or
workflow behavior; operators remain responsible for manual `routing.yaml` changes.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agentflow.persistence.database import DatabaseManager
from agentflow.persistence.models import RunRecord
from agentflow.task.profile import Stage


@dataclass
class ModelStatistics:
    """Aggregated metrics for one deterministically selected implementation model."""

    model: str
    tasks: int = 0
    routing_selections: int = 0
    first_pass_successes: int = 0
    first_pass_attempts: int = 0
    implementation_attempts: int = 0
    verification_attempts: int = 0
    escalations: int = 0
    review_findings: int = 0

    @property
    def first_pass_rate(self) -> float | None:
        """Return the first-pass verification percentage, or None when no pass was recorded."""
        if self.first_pass_attempts == 0:
            return None
        return self.first_pass_successes / self.first_pass_attempts


@dataclass
class StatisticsReport:
    """Immutable-source analytics report rendered by `agentflow stats`."""

    total_runs: int
    models: list[ModelStatistics] = field(default_factory=list)
    routing_rule_frequency: dict[str, int] = field(default_factory=dict)
    review_findings_by_severity: dict[str, int] = field(default_factory=dict)
    average_stage_duration_seconds: dict[str, float] = field(default_factory=dict)


class StatisticsService:
    """Derive local metrics exclusively from persisted AgentFlow records."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def build_report(self, project_id: str | None = None) -> StatisticsReport:
        """Build a deterministic report for one project or all tracked projects."""
        runs = self.db_manager.list_runs(project_id=project_id, limit=100_000)
        return self._build_report(runs)

    def _build_report(self, runs: list[RunRecord]) -> StatisticsReport:
        rule_frequency: Counter[str] = Counter()
        model_runs: dict[str, set[str]] = defaultdict(set)
        metrics: dict[str, ModelStatistics] = {}
        severity_counts: Counter[str] = Counter()
        stage_durations: dict[str, list[float]] = defaultdict(list)

        for run in runs:
            decisions = self.db_manager.list_routing_decisions(run.id)
            for decision in decisions:
                rule_frequency[decision.matched_rule] += 1
                metrics.setdefault(decision.model, ModelStatistics(model=decision.model))
                metrics[decision.model].routing_selections += 1

            for stage in self.db_manager.list_stages(run.id):
                if stage.completed_at is not None:
                    stage_durations[stage.stage].append(
                        (stage.completed_at - stage.started_at).total_seconds()
                    )

            events = self.db_manager.list_events(run.id)
            for event in events:
                if event.event != "REVIEW_COMPLETED":
                    continue
                if event.model is not None:
                    metrics.setdefault(event.model, ModelStatistics(model=event.model))
                for key, value in event.attributes.items():
                    if key.startswith("severity_") and isinstance(value, int):
                        severity_counts[key.removeprefix("severity_").upper()] += value
                finding_count = event.attributes.get("finding_count")
                if event.model is not None and isinstance(finding_count, int):
                    metrics[event.model].review_findings += finding_count

            implementation_decision = next(
                (
                    decision
                    for decision in decisions
                    if decision.stage == Stage.IMPLEMENTATION.value
                ),
                None,
            )
            if implementation_decision is None:
                continue

            model = implementation_decision.model
            model_runs[model].add(run.id)

            for stage in self.db_manager.list_stages(run.id):
                if stage.stage == "implementation":
                    metrics[model].implementation_attempts += max(stage.attempt_count, 1)

            verification_events = [
                event for event in events if event.event == "VERIFICATION_COMPLETED"
            ]
            metrics[model].verification_attempts += len(verification_events)
            if verification_events:
                first = verification_events[0]
                metrics[model].first_pass_attempts += 1
                if first.attributes.get("success") is True:
                    metrics[model].first_pass_successes += 1

            metrics[model].escalations += sum(
                1
                for decision in self.db_manager.list_decisions(run.id)
                if "escalation" in decision.question
            )
        for model, run_ids in model_runs.items():
            metrics[model].tasks = len(run_ids)

        return StatisticsReport(
            total_runs=len(runs),
            models=sorted(metrics.values(), key=lambda item: item.model),
            routing_rule_frequency=dict(sorted(rule_frequency.items())),
            review_findings_by_severity=dict(sorted(severity_counts.items())),
            average_stage_duration_seconds={
                stage: sum(durations) / len(durations)
                for stage, durations in sorted(stage_durations.items())
                if durations
            },
        )
