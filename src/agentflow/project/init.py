"""Starter `.ai-orchestrator/routing.yaml` generation for `agentflow init`.

Detection is a heuristic starting point only -- there is no way to reliably infer a project's
full test/build setup from marker files alone, so the generated config always needs a human
review pass before real tasks are routed against it.
"""

import json
from pathlib import Path
from typing import Literal

from agentflow.agents.base import Provider
from agentflow.config.models import (
    DocumentationConfig,
    LimitsConfig,
    ProjectConfig,
    ProjectMetaConfig,
    VerificationGroup,
)
from agentflow.errors import InitError
from agentflow.routing.complexity import DEFAULT_COMPLEXITY_CONFIG
from agentflow.routing.rules import (
    DEFAULT_ROUTING_RULES,
    DocumentationModels,
    ImplementationModels,
    ModelRef,
    ModelsConfig,
    PlannerModels,
    ReviewModels,
)

# (low-tier model, high-tier model) per provider. Google has only one model in the V1 pool, so
# its "tiers" are identical -- there is no way to give it real escalation headroom.
_PROVIDER_TIERS: dict[Provider, tuple[str, str]] = {
    Provider.ANTHROPIC: ("Claude Sonnet 5", "Claude Opus 5.5"),
    Provider.OPENAI: ("GPT-6 Luna", "GPT-6 Sol"),
    Provider.GOOGLE: ("Gemini 3.8 Flash", "Gemini 3.8 Flash"),
}

# Per-role provider preference order and low/high tier. Chosen so that, when all three providers
# are available, this reproduces DEFAULT_MODELS_CONFIG exactly (locked in by a unit test) --
# providers only get substituted when a preferred one is genuinely unavailable.
_ROLE_PREFERENCES: dict[str, tuple[list[Provider], Literal["low", "high"]]] = {
    "planner.default": ([Provider.ANTHROPIC, Provider.OPENAI, Provider.GOOGLE], "low"),
    "planner.architecture": ([Provider.ANTHROPIC, Provider.OPENAI, Provider.GOOGLE], "high"),
    "implementation.lightweight": ([Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE], "low"),
    "implementation.standard": ([Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE], "high"),
    "implementation.escalation": ([Provider.ANTHROPIC, Provider.OPENAI, Provider.GOOGLE], "low"),
    "review.default": ([Provider.GOOGLE, Provider.ANTHROPIC, Provider.OPENAI], "low"),
    "review.deep": ([Provider.ANTHROPIC, Provider.GOOGLE, Provider.OPENAI], "low"),
    "review.architecture": ([Provider.ANTHROPIC, Provider.GOOGLE, Provider.OPENAI], "high"),
    "documentation.default": ([Provider.GOOGLE, Provider.ANTHROPIC, Provider.OPENAI], "low"),
}

_SKIP_DIR_NAMES = frozenset(
    {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target"}
)

# Marker filename -> (group kind, guessed test command). Order matters: first match per
# directory wins so a directory with both package.json and pyproject.toml isn't double-counted
# under two different guessed commands for the same group.
_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("package.json", "node", "npm test"),
    ("pyproject.toml", "python", "python -m pytest"),
    ("requirements.txt", "python", "python -m pytest"),
    ("pom.xml", "java", "mvn test"),
)

_MAX_DEPTH = 3


def _marker_for(directory: Path) -> tuple[str, str, str] | None:
    """Return (marker_name, group_kind, command) for the first matching marker in `directory`."""
    for marker_name, group_kind, command in _MARKERS:
        if (directory / marker_name).exists():
            if marker_name == "package.json":
                try:
                    package = json.loads((directory / marker_name).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
                if not isinstance(scripts, dict) or not isinstance(scripts.get("test"), str):
                    continue
                if not scripts["test"].strip():
                    continue
            return marker_name, group_kind, command
    return None


def detect_verification_groups(root_path: Path) -> dict[str, VerificationGroup]:
    """Heuristically detect per-directory test groups by walking the repo for marker files.

    Walks up to `_MAX_DEPTH` directories deep, skipping dependency/build directories and any
    hidden directory. Each match becomes one `verification:` group, keyed by directory name
    (or the detected group kind at the repo root).
    """
    groups: dict[str, VerificationGroup] = {}

    def _register(name: str, marker_relative: str, command: str) -> None:
        unique_name = name
        suffix = 2
        while unique_name in groups:
            unique_name = f"{name}-{suffix}"
            suffix += 1
        working_directory = Path(marker_relative).parent.as_posix()
        groups[unique_name] = VerificationGroup(
            detect=[marker_relative],
            working_directory=working_directory,
            commands=[command],
        )

    root_marker = _marker_for(root_path)
    if root_marker is not None:
        marker_name, group_kind, command = root_marker
        _register(group_kind, marker_name, command)

    def _walk(directory: Path, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in _SKIP_DIR_NAMES:
                continue
            marker = _marker_for(entry)
            if marker is not None:
                marker_name, _group_kind, command = marker
                relative_marker = (entry / marker_name).relative_to(root_path).as_posix()
                _register(entry.name, relative_marker, command)
            _walk(entry, depth + 1)

    _walk(root_path, 1)
    return groups


def _resolve_role(role: str, providers: set[Provider]) -> ModelRef:
    """Pick a provider+model for `role` from whichever of its preferred providers is available."""
    preferences, tier = _ROLE_PREFERENCES[role]
    for provider in preferences:
        if provider in providers:
            low, high = _PROVIDER_TIERS[provider]
            return ModelRef(provider=provider.value, model=high if tier == "high" else low)
    # Unreachable as long as `providers` is non-empty -- every preference list includes all
    # three providers, so at least one is always present.
    raise AssertionError(f"No available provider for role '{role}' among {providers}.")


def build_models_config(providers: set[Provider]) -> ModelsConfig:
    """Build a ModelsConfig using only `providers`, preferring each role's usual provider.

    When all three providers are available this reproduces `DEFAULT_MODELS_CONFIG` exactly
    (see tests/test_init.py) -- providers are only substituted when a role's preferred one is
    genuinely unavailable.
    """
    if not providers:
        raise InitError("build_models_config requires at least one available provider.")
    return ModelsConfig(
        planner=PlannerModels(
            default=_resolve_role("planner.default", providers),
            architecture=_resolve_role("planner.architecture", providers),
        ),
        implementation=ImplementationModels(
            lightweight=_resolve_role("implementation.lightweight", providers),
            standard=_resolve_role("implementation.standard", providers),
            escalation=_resolve_role("implementation.escalation", providers),
        ),
        review=ReviewModels(
            default=_resolve_role("review.default", providers),
            deep=_resolve_role("review.deep", providers),
            architecture=_resolve_role("review.architecture", providers),
        ),
        documentation=DocumentationModels(
            default=_resolve_role("documentation.default", providers),
        ),
    )


def build_starter_config(root_path: Path, providers: set[Provider]) -> ProjectConfig:
    """Assemble a starter ProjectConfig for `root_path`, using only the given `providers`.

    `documentation` is left disabled: there is no reliable way to guess which docs a project
    wants kept in sync, so that section is a deliberate no-op until the user configures it.
    `routing:` stays AgentFlow's default unchanged -- its fields are alias references (e.g.
    "implementation.lightweight"), not literal providers, so they work regardless of which
    provider each alias resolves to.
    """
    verification = detect_verification_groups(root_path)
    return ProjectConfig(
        version=1,
        project=ProjectMetaConfig(name=root_path.name),
        models=build_models_config(providers),
        complexity=DEFAULT_COMPLEXITY_CONFIG,
        routing=DEFAULT_ROUTING_RULES,
        verification=verification or None,
        limits=LimitsConfig(),
        documentation=DocumentationConfig(enabled=False, candidate_files=[]),
    )
