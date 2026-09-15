"""First-match-wins rule matching helpers.

Rules configured at the same priority level are evaluated top-to-bottom; the first rule whose
condition matches wins. No scoring, weighting, or "best match" semantics are involved.
"""

from agentflow.routing.complexity import ComplexityLevel
from agentflow.routing.rules import ComplexityRule


def match_complexity_rule(
    rules: list[ComplexityRule], level: ComplexityLevel
) -> ComplexityRule | None:
    """Return the first rule (top-to-bottom) whose complexity condition matches, or None."""
    target = level.value.lower()
    for rule in rules:
        if rule.when.complexity == target:
            return rule
    return None
