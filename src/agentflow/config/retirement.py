"""Pure substitution logic for user-managed model retirements.

A retirement is a static, user-authored mapping (persisted in the global
`~/.agentflow/config.yaml`, `GlobalConfig.models.retired`) from a retired model name to
its replacement. Applying it is a deterministic lookup -- no AI model participates in the
decision -- consistent with AGENTS.md's deterministic-routing invariant.
"""

from collections.abc import Mapping

from agentflow.errors import ConfigurationError


def apply_retirements(model: str, retired: Mapping[str, str]) -> str:
    """Chain-resolve `model` through the retirement map, case-insensitively.

    Follows chains (X -> Y, Y -> Z resolves X to Z) and raises ConfigurationError if the
    chain revisits a model it has already seen, rather than looping forever.
    """
    seen = {model.strip().lower()}
    current = model
    while (key := current.strip().lower()) in retired:
        next_model = retired[key]
        next_key = next_model.strip().lower()
        if next_key in seen:
            raise ConfigurationError(
                f"Retirement cycle detected resolving '{model}': "
                f"'{current}' -> '{next_model}' revisits an already-seen model."
            )
        seen.add(next_key)
        current = next_model
    return current
